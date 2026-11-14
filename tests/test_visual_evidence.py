from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from extract_visual_evidence import (
    EVIDENCE_EVENTS_CSV_HEADER,
    BestFrameSelection,
    EventClipRecord,
    EventRecord,
    EvidenceConfig,
    EvidenceEventRecord,
    FrameTracks,
    TrackDetection,
    extract_event_video_clip,
    extract_person_crop,
    index_track_detections,
    load_evidence_config,
    read_evidence_events_csv,
    render_clip_frame_overlay,
    render_event_snapshot,
    render_evidence_contact_sheet,
    run_visual_evidence,
    save_evidence_config,
    select_best_frame_for_event,
    verify_evidence_records,
    write_evidence_events_csv,
)
from extract_visual_evidence import (
    main as extract_visual_evidence_main,
)
from generate_dwell_events import write_events_csv
from helpers import make_video
from track_people import write_tracks_csv


def test_evidence_config_defaults_and_validation() -> None:
    cfg = EvidenceConfig()
    assert cfg.best_frame_strategy == "highest_confidence"
    assert cfg.crop_padding_ratio == pytest.approx(0.10)
    assert cfg.min_padding_px == 8
    assert cfg.jpeg_quality == 92
    assert cfg.overlay_scene_geometry is True
    assert cfg.contact_sheet_columns == 4
    assert cfg.contact_sheet_max_items == 12
    assert cfg.extract_video_clips is False
    assert cfg.clip_pre_roll_seconds == pytest.approx(1.0)
    assert cfg.clip_post_roll_seconds == pytest.approx(1.0)
    assert cfg.render_clip_telemetry is True
    assert cfg.extract_storyboards is False
    assert cfg.storyboard_stages == ("start", "peak", "end")

    with pytest.raises(ValueError, match="best_frame_strategy must be one of"):
        EvidenceConfig(best_frame_strategy="invalid_strategy")

    with pytest.raises(ValueError, match="crop_padding_ratio must be >= 0"):
        EvidenceConfig(crop_padding_ratio=-0.05)

    with pytest.raises(ValueError, match="min_padding_px must be >= 0"):
        EvidenceConfig(min_padding_px=-1)

    with pytest.raises(ValueError, match="jpeg_quality must be in"):
        EvidenceConfig(jpeg_quality=101)

    with pytest.raises(ValueError, match="contact_sheet_columns must be >= 1"):
        EvidenceConfig(contact_sheet_columns=0)

    with pytest.raises(ValueError, match="contact_sheet_max_items must be >= 1"):
        EvidenceConfig(contact_sheet_max_items=0)

    with pytest.raises(ValueError, match="clip_pre_roll_seconds must be >= 0"):
        EvidenceConfig(clip_pre_roll_seconds=-0.5)

    with pytest.raises(ValueError, match="clip_post_roll_seconds must be >= 0"):
        EvidenceConfig(clip_post_roll_seconds=-0.5)


def test_best_frame_selection_area_property() -> None:
    sel = BestFrameSelection(
        event_id="evt_0001",
        track_id=2,
        frame_id=15,
        x1=100.0,
        y1=80.0,
        x2=160.0,
        y2=220.0,
        confidence=0.87,
        strategy_used="highest_confidence",
    )
    assert sel.area == pytest.approx(8400.0)
    assert sel.used_nearest_fallback is False


def test_evidence_event_record_validation_and_to_row() -> None:
    assert len(EVIDENCE_EVENTS_CSV_HEADER) == 9
    base_event = EventRecord(
        event_id="evt_0001",
        event_type="dwell_time_event",
        track_id=3,
        zone_id="zone_center",
        start_time=2.5,
        end_time=6.25,
        duration_seconds=3.75,
    )
    rec = EvidenceEventRecord.from_event(
        base_event,
        snapshot_path="snapshots/evt_0001_3.jpg",
        crop_path="crops/evt_0001_3.jpg",
        best_frame_id=30,
        best_confidence=0.89,
    )
    assert rec.to_row() == [
        "evt_0001",
        "dwell_time_event",
        "3",
        "zone_center",
        "2.5000",
        "6.2500",
        "3.7500",
        "snapshots/evt_0001_3.jpg",
        "crops/evt_0001_3.jpg",
    ]

    with pytest.raises(ValueError, match="snapshot_path must be a non-empty string"):
        EvidenceEventRecord.from_event(
            base_event,
            snapshot_path="",
            crop_path="crops/evt_0001_3.jpg",
        )

    with pytest.raises(ValueError, match="crop_path must be a non-empty string"):
        EvidenceEventRecord.from_event(
            base_event,
            snapshot_path="snapshots/evt_0001_3.jpg",
            crop_path="   ",
        )


def test_evidence_config_yaml_round_trip(tmp_path: Path) -> None:
    cfg = EvidenceConfig(
        best_frame_strategy="midpoint",
        crop_padding_ratio=0.15,
        min_padding_px=12,
        jpeg_quality=95,
        overlay_scene_geometry=False,
        contact_sheet_columns=3,
        contact_sheet_max_items=9,
    )
    yaml_path = tmp_path / "evidence.yaml"
    save_evidence_config(cfg, yaml_path)
    loaded = load_evidence_config(yaml_path)
    assert loaded == cfg

    default_yaml = (
        Path(__file__).resolve().parents[1]
        / "cv_projects"
        / "07_visual_evidence"
        / "configs"
        / "evidence.yaml"
    )
    assert load_evidence_config(default_yaml) == EvidenceConfig()

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    assert load_evidence_config(empty_yaml) == EvidenceConfig()

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected YAML mapping"):
        load_evidence_config(bad_yaml)

    with pytest.raises(FileNotFoundError):
        load_evidence_config(tmp_path / "missing.yaml")


def test_evidence_events_csv_round_trip(tmp_path: Path) -> None:
    records = [
        EvidenceEventRecord(
            event_id="evt_0001",
            event_type="zone_entry",
            track_id=1,
            zone_id="zone_center",
            start_time=0.5,
            end_time=0.5,
            duration_seconds=0.0,
            snapshot_path="snapshots/evt_0001_1.jpg",
            crop_path="crops/evt_0001_1.jpg",
        ),
        EvidenceEventRecord(
            event_id="evt_0002",
            event_type="dwell_time_event",
            track_id=1,
            zone_id="zone_center",
            start_time=0.5,
            end_time=4.25,
            duration_seconds=3.75,
            snapshot_path="snapshots/evt_0002_1.jpg",
            crop_path="crops/evt_0002_1.jpg",
        ),
    ]
    csv_path = tmp_path / "events.csv"
    write_evidence_events_csv(records, csv_path)
    loaded = read_evidence_events_csv(csv_path)
    assert loaded == records

    # Round trip with video_clip_path and storyboard_path (11-column format)
    records_with_clips = [
        EvidenceEventRecord(
            event_id="evt_0001",
            event_type="zone_entry",
            track_id=1,
            zone_id="zone_center",
            start_time=0.5,
            end_time=0.5,
            duration_seconds=0.0,
            snapshot_path="snapshots/evt_0001_1.jpg",
            crop_path="crops/evt_0001_1.jpg",
            video_clip_path="clips/evt_0001.mp4",
            storyboard_path="storyboards/evt_0001_storyboard.jpg",
        ),
        EvidenceEventRecord(
            event_id="evt_0002",
            event_type="dwell_time_event",
            track_id=1,
            zone_id="zone_center",
            start_time=0.5,
            end_time=4.25,
            duration_seconds=3.75,
            snapshot_path="snapshots/evt_0002_1.jpg",
            crop_path="crops/evt_0002_1.jpg",
            video_clip_path="clips/evt_0002.mp4",
            storyboard_path="storyboards/evt_0002_storyboard.jpg",
        ),
    ]
    csv_clips_path = tmp_path / "events_with_clips.csv"
    write_evidence_events_csv(records_with_clips, csv_clips_path)
    loaded_clips = read_evidence_events_csv(csv_clips_path)
    assert loaded_clips == records_with_clips

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("event_id,event_type\n1,zone_entry\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected events.csv header"):
        read_evidence_events_csv(bad_header)

    with pytest.raises(FileNotFoundError):
        read_evidence_events_csv(tmp_path / "missing_events.csv")


def test_select_best_frame_for_event_all_strategies_and_fallback() -> None:
    frame_tracks = [
        FrameTracks(
            frame_id=10,
            tracks=[
                TrackDetection(
                    frame_id=10,
                    track_id=1,
                    x1=100.0,
                    y1=100.0,
                    x2=150.0,
                    y2=200.0,
                    confidence=0.75,
                )
            ],
        ),
        FrameTracks(
            frame_id=15,
            tracks=[
                TrackDetection(
                    frame_id=15,
                    track_id=1,
                    x1=105.0,
                    y1=100.0,
                    x2=165.0,
                    y2=210.0,
                    confidence=0.82,
                )
            ],
        ),
        FrameTracks(
            frame_id=20,
            tracks=[
                TrackDetection(
                    frame_id=20,
                    track_id=1,
                    x1=110.0,
                    y1=100.0,
                    x2=155.0,
                    y2=195.0,
                    confidence=0.91,
                )
            ],
        ),
    ]
    indexed = index_track_detections(frame_tracks)
    assert set(indexed[1].keys()) == {10, 15, 20}

    event = EventRecord(
        event_id="evt_0001",
        event_type="dwell_time_event",
        track_id=1,
        zone_id="zone_center",
        start_time=1.0,  # frame 10 @ 10 fps
        end_time=2.0,  # frame 20 @ 10 fps
        duration_seconds=1.0,
    )

    sel_conf = select_best_frame_for_event(
        event, indexed, fps=10.0, strategy="highest_confidence"
    )
    assert sel_conf.frame_id == 20
    assert sel_conf.confidence == pytest.approx(0.91)
    assert sel_conf.used_nearest_fallback is False

    sel_area = select_best_frame_for_event(
        event, indexed, fps=10.0, strategy="largest_area"
    )
    assert sel_area.frame_id == 15  # 60 * 110 = 6600 vs 5000 vs 4275

    sel_mid = select_best_frame_for_event(event, indexed, fps=10.0, strategy="midpoint")
    assert sel_mid.frame_id == 15

    sel_start = select_best_frame_for_event(
        event, indexed, fps=10.0, strategy="start_frame"
    )
    assert sel_start.frame_id == 10

    # Fallback when event window has no exact frame for track 1 (e.g. frame 17)
    gap_event = EventRecord(
        event_id="evt_0002",
        event_type="line_crossing",
        track_id=1,
        zone_id="line_center",
        start_time=1.7,
        end_time=1.7,
        duration_seconds=0.0,
    )
    sel_fallback = select_best_frame_for_event(
        gap_event, indexed, fps=10.0, strategy="highest_confidence"
    )
    assert sel_fallback.frame_id == 15
    assert sel_fallback.used_nearest_fallback is True

    with pytest.raises(KeyError, match="No tracked detections found for track_id=99"):
        select_best_frame_for_event(
            EventRecord(
                event_id="evt_0003",
                event_type="zone_entry",
                track_id=99,
                zone_id="zone_center",
                start_time=1.0,
                end_time=1.0,
                duration_seconds=0.0,
            ),
            indexed,
            fps=10.0,
        )


def test_extract_person_crop_interior_and_boundary_clipping() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[50:150, 100:180] = 255

    interior_sel = BestFrameSelection(
        event_id="evt_0001",
        track_id=1,
        frame_id=5,
        x1=100.0,
        y1=50.0,
        x2=180.0,
        y2=150.0,
        confidence=0.90,
        strategy_used="highest_confidence",
    )
    crop_int, clipped_int = extract_person_crop(
        frame, interior_sel, padding_ratio=0.10, min_padding_px=8
    )
    assert clipped_int is False
    # box is 80x100 -> pad_x=max(8, 8)=8, pad_y=max(8, 10)=10 -> 96x120
    assert crop_int.shape == (120, 96, 3)

    edge_sel = BestFrameSelection(
        event_id="evt_0002",
        track_id=1,
        frame_id=5,
        x1=2.0,
        y1=3.0,
        x2=60.0,
        y2=198.0,
        confidence=0.88,
        strategy_used="highest_confidence",
    )
    crop_edge, clipped_edge = extract_person_crop(
        frame, edge_sel, padding_ratio=0.10, min_padding_px=8
    )
    assert clipped_edge is True
    assert crop_edge.shape[0] == 200
    assert crop_edge.shape[1] == 68


def test_render_event_snapshot_and_contact_sheet() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    event = EventRecord(
        event_id="evt_0001",
        event_type="dwell_time_event",
        track_id=1,
        zone_id="zone_center",
        start_time=1.0,
        end_time=4.5,
        duration_seconds=3.5,
    )
    sel = BestFrameSelection(
        event_id="evt_0001",
        track_id=1,
        frame_id=18,
        x1=60.0,
        y1=50.0,
        x2=130.0,
        y2=190.0,
        confidence=0.92,
        strategy_used="highest_confidence",
    )
    snapshot = render_event_snapshot(frame, event, sel, scene_config=None)
    assert snapshot.shape == frame.shape
    assert int(snapshot.sum()) > 0

    crop, _ = extract_person_crop(frame, sel)
    sheet = render_evidence_contact_sheet(
        [(crop, "evt_0001 | dwell | T1"), (crop, "evt_0002 | entry | T1")],
        columns=2,
        thumb_size=(160, 200),
    )
    assert sheet.shape == (200 + 36, 160 * 2, 3)

    empty_sheet = render_evidence_contact_sheet([], columns=3, thumb_size=(160, 200))
    assert empty_sheet.shape == (200 + 36, 160 * 3, 3)


def test_verify_evidence_records_valid_and_missing(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    crop_dir = tmp_path / "crops"
    snap_dir.mkdir(parents=True)
    crop_dir.mkdir(parents=True)

    (snap_dir / "evt_0001_1.jpg").write_bytes(b"fake_jpeg_bytes")
    (crop_dir / "evt_0001_1.jpg").write_bytes(b"fake_crop_bytes")

    rec1 = EvidenceEventRecord(
        event_id="evt_0001",
        event_type="zone_entry",
        track_id=1,
        zone_id="zone_center",
        start_time=0.5,
        end_time=0.5,
        duration_seconds=0.0,
        snapshot_path="snapshots/evt_0001_1.jpg",
        crop_path="crops/evt_0001_1.jpg",
    )
    report_ok = verify_evidence_records([rec1], base_dir=tmp_path)
    assert report_ok["all_paths_valid"] is True
    assert report_ok["resolved_snapshots"] == 1
    assert report_ok["resolved_crops"] == 1

    rec_missing = EvidenceEventRecord(
        event_id="evt_0002",
        event_type="zone_exit",
        track_id=1,
        zone_id="zone_center",
        start_time=1.5,
        end_time=1.5,
        duration_seconds=0.0,
        snapshot_path="snapshots/evt_0002_1.jpg",
        crop_path="crops/evt_0002_1.jpg",
    )
    report_bad = verify_evidence_records([rec1, rec_missing], base_dir=tmp_path)
    assert report_bad["all_paths_valid"] is False
    assert report_bad["missing_or_empty_snapshots"] == ["snapshots/evt_0002_1.jpg"]
    assert report_bad["missing_or_empty_crops"] == ["crops/evt_0002_1.jpg"]


def test_render_clip_frame_overlay_telemetry_and_breadcrumbs() -> None:
    frame = np.full((300, 400, 3), 50, dtype=np.uint8)
    event = EventRecord(
        event_id="evt_0001",
        event_type="dwell_time_event",
        track_id=3,
        zone_id="restricted_zone",
        start_time=1.0,
        end_time=4.0,
        duration_seconds=3.0,
    )
    det = TrackDetection(
        frame_id=30,
        track_id=3,
        x1=50.0,
        y1=60.0,
        x2=120.0,
        y2=200.0,
        confidence=0.91,
    )
    traj = [(50.0, 180.0), (70.0, 190.0), (85.0, 200.0)]

    rendered = render_clip_frame_overlay(
        frame,
        event=event,
        track_detection=det,
        current_frame_id=30,
        current_timestamp=1.5,
        trajectory_points=traj,
        camera_id="cam_north",
        alert_text="LOITERING ALERT",
    )
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    # Canvas should not be identical to original blank frame
    assert not np.array_equal(rendered, frame)

    # Empty frame raises ValueError
    with pytest.raises(ValueError, match="Invalid frame shape"):
        render_clip_frame_overlay(np.zeros((0, 0, 3), dtype=np.uint8))


def test_extract_event_video_clip_creates_playable_clip(tmp_path: Path) -> None:
    video_path = make_video(tmp_path, name="source_test.avi", count=15)
    event = EventRecord(
        event_id="evt_0010",
        event_type="dwell_time_event",
        track_id=1,
        zone_id="zone_main",
        start_time=0.4,
        end_time=0.8,
        duration_seconds=0.4,
    )
    clip_path = tmp_path / "clips" / "evt_0010.mp4"
    record = extract_event_video_clip(
        video_path,
        clip_path,
        event=event,
        fps=10.0,
        total_source_frames=15,
        pre_roll_seconds=0.2,
        post_roll_seconds=0.2,
        camera_id="cam_01",
        alert_text="DWELL",
    )
    assert Path(record.video_clip_path).is_file()
    assert record.event_id == "evt_0010"
    assert record.track_id == 1
    assert record.total_frames > 0
    assert record.duration_seconds > 0.0

    # Verify clip can be opened and decoded
    cap = cv2.VideoCapture(str(clip_path))
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret is True
    assert frame is not None
    cap.release()


def test_run_visual_evidence_pipeline_and_cli_end_to_end(tmp_path: Path) -> None:
    video_path = make_video(tmp_path, name="evidence_clip.avi", count=6)
    tracks_csv = tmp_path / "tracks.csv"
    events_csv = tmp_path / "input_events.csv"
    out_dir = tmp_path / "evidence_out"

    frame_tracks = [
        FrameTracks(
            frame_id=i,
            tracks=[
                TrackDetection(
                    frame_id=i,
                    track_id=1,
                    x1=10.0,
                    y1=8.0,
                    x2=34.0,
                    y2=40.0,
                    confidence=0.70 + 0.04 * i,
                )
            ],
        )
        for i in range(6)
    ]
    write_tracks_csv(tracks_csv, frame_tracks)

    events = [
        EventRecord(
            event_id="evt_0001",
            event_type="zone_entry",
            track_id=1,
            zone_id="zone_center",
            start_time=0.1,
            end_time=0.1,
            duration_seconds=0.0,
        ),
        EventRecord(
            event_id="evt_0002",
            event_type="dwell_time_event",
            track_id=1,
            zone_id="zone_center",
            start_time=0.1,
            end_time=0.4,
            duration_seconds=0.3,
        ),
    ]
    write_events_csv(events_csv, events)

    summary = run_visual_evidence(
        source_path=video_path,
        events_csv_path=events_csv,
        tracks_csv_path=tracks_csv,
        output_dir=out_dir,
        scene_config_path=None,
    )
    assert summary["total_events"] == 2
    assert summary["verification"]["all_paths_valid"] is True
    assert (out_dir / "snapshots" / "evt_0001_1.jpg").is_file()
    assert (out_dir / "crops" / "evt_0001_1.jpg").is_file()
    assert (out_dir / "snapshots" / "evt_0002_1.jpg").is_file()
    assert (out_dir / "crops" / "evt_0002_1.jpg").is_file()
    assert (out_dir / "evidence_contact_sheet.jpg").is_file()
    assert (out_dir / "visual_evidence_summary.json").is_file()

    enriched = read_evidence_events_csv(out_dir / "events.csv")
    assert len(enriched) == 2
    assert enriched[0].snapshot_path == "snapshots/evt_0001_1.jpg"
    assert enriched[1].crop_path == "crops/evt_0002_1.jpg"

    cli_out = tmp_path / "cli_out"
    rc = extract_visual_evidence_main(
        [
            "--source",
            str(video_path),
            "--events",
            str(events_csv),
            "--tracks",
            str(tracks_csv),
            "--output-dir",
            str(cli_out),
            "--strategy",
            "midpoint",
            "--padding-ratio",
            "0.12",
            "--min-padding-px",
            "6",
            "--jpeg-quality",
            "90",
        ]
    )
    assert rc == 0
    assert (cli_out / "events.csv").is_file()
