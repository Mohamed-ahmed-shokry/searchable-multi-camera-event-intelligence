from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from extract_visual_evidence import (
    EVIDENCE_EVENTS_CSV_HEADER,
    BestFrameSelection,
    EventRecord,
    EvidenceConfig,
    EvidenceEventRecord,
    FrameTracks,
    TrackDetection,
    extract_person_crop,
    index_track_detections,
    load_evidence_config,
    read_evidence_events_csv,
    save_evidence_config,
    select_best_frame_for_event,
    write_evidence_events_csv,
)


def test_evidence_config_defaults_and_validation() -> None:
    cfg = EvidenceConfig()
    assert cfg.best_frame_strategy == "highest_confidence"
    assert cfg.crop_padding_ratio == pytest.approx(0.10)
    assert cfg.min_padding_px == 8
    assert cfg.jpeg_quality == 92
    assert cfg.overlay_scene_geometry is True
    assert cfg.contact_sheet_columns == 4
    assert cfg.contact_sheet_max_items == 12

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
