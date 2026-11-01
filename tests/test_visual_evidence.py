from __future__ import annotations

from pathlib import Path

import pytest
from extract_visual_evidence import (
    EVIDENCE_EVENTS_CSV_HEADER,
    BestFrameSelection,
    EventRecord,
    EvidenceConfig,
    EvidenceEventRecord,
    load_evidence_config,
    read_evidence_events_csv,
    save_evidence_config,
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
