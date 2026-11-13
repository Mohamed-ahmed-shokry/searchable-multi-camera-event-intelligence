"""Unit tests for Phase 11d behavioral insight detectors and domain types."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from event_intelligence.insights import (
    InsightConfig,
    InsightRecord,
    InsightType,
    Severity,
    detect_cross_camera_transitions,
    detect_loitering_insights,
    detect_speed_anomaly_insights,
    detect_stationary_and_drop_insights,
    detect_wrong_way_insights,
    generate_all_insights,
)


@dataclass
class DummyEvent:
    event_id: str
    event_type: str
    track_id: int
    zone_id: str
    start_time: float
    end_time: float
    duration: float = 0.0
    evidence_snapshot_path: str | None = None
    evidence_crop_path: str | None = None


@dataclass
class DummyPoint:
    track_id: int
    frame_id: int
    timestamp: float
    x: float
    y: float


@dataclass
class DummyTrace:
    stay_id: str
    track_id: int
    zone_id: str
    entry_timestamp: float
    exit_timestamp: float
    track_lost_in_zone: bool


class TestInsightTypesAndConfig(unittest.TestCase):
    """Test validation and serialization of InsightRecord and InsightConfig."""

    def test_insight_record_validation_and_serialization(self) -> None:
        rec = InsightRecord(
            insight_id="ins_001",
            camera_id="cam_main",
            insight_type=InsightType.LOITERING.value,
            severity=Severity.HIGH.value,
            track_id=10,
            zone_id="restricted_zone",
            start_time=5.0,
            end_time=25.0,
            duration=20.0,
            description="Loitering detected",
            evidence_snapshot_path="snapshots/snap_001.jpg",
            evidence_crop_path="crops/crop_001.jpg",
            details={"key": "val"},
        )
        d = rec.to_dict()
        self.assertEqual(d["insight_id"], "ins_001")
        self.assertEqual(d["severity"], "high")

        restored = InsightRecord.from_dict(d)
        self.assertEqual(restored, rec)

    def test_insight_record_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            InsightRecord(
                insight_id="",
                camera_id="cam",
                insight_type=InsightType.LOITERING.value,
                severity=Severity.HIGH.value,
                track_id=1,
                zone_id="z",
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
                description="desc",
            )
        with self.assertRaises(ValueError):
            InsightRecord(
                insight_id="id",
                camera_id="cam",
                insight_type="invalid_type",
                severity=Severity.HIGH.value,
                track_id=1,
                zone_id="z",
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
                description="desc",
            )
        with self.assertRaises(ValueError):
            InsightRecord(
                insight_id="id",
                camera_id="cam",
                insight_type=InsightType.LOITERING.value,
                severity=Severity.HIGH.value,
                track_id=1,
                zone_id="z",
                start_time=10.0,
                end_time=5.0,  # end < start
                duration=1.0,
                description="desc",
            )

    def test_insight_config_validation_and_to_dict(self) -> None:
        cfg = InsightConfig(
            loitering_factor=2.0,
            min_loitering_seconds=10.0,
            speed_anomaly_threshold_px_s=150.0,
            restricted_zones=("secure_gate",),
        )
        d = cfg.to_dict()
        self.assertEqual(d["loitering_factor"], 2.0)
        self.assertIn("secure_gate", d["restricted_zones"])

        with self.assertRaises(ValueError):
            InsightConfig(loitering_factor=0.5)  # must be > 1.0


class TestInsightDetectors(unittest.TestCase):
    """Test individual insight detectors and full generation."""

    def test_detect_loitering_insights(self) -> None:
        cfg = InsightConfig(
            min_loitering_seconds=10.0,
            loitering_factor=2.0,
            restricted_zones=("restricted_area",),
            per_zone_loitering_seconds={"short_zone": 5.0},
        )
        events = [
            # Normal dwell
            DummyEvent("e1", "dwell_time_event", 1, "lobby", 0.0, 4.0, 4.0),
            DummyEvent("e2", "dwell_time_event", 2, "lobby", 1.0, 5.0, 4.0),
            # Prolonged dwell in lobby (median=4.0, factor=2.0 -> threshold 10.0)
            DummyEvent("e3", "dwell_time_event", 3, "lobby", 10.0, 25.0, 15.0),
            # Loitering in restricted area
            DummyEvent(
                "e4", "dwell_time_event", 4, "restricted_area", 30.0, 45.0, 15.0
            ),
            # Custom zone threshold
            DummyEvent("e5", "dwell_time_event", 5, "short_zone", 50.0, 57.0, 7.0),
        ]
        insights = detect_loitering_insights(events, cfg, camera_id="c1")
        self.assertEqual(len(insights), 3)

        # Track 3 loitering in lobby
        lobby_ins = next(i for i in insights if i.track_id == 3)
        self.assertEqual(lobby_ins.severity, "medium")
        self.assertEqual(lobby_ins.zone_id, "lobby")

        # Track 4 loitering in restricted area -> severity escalated to high/critical
        sec_ins = next(i for i in insights if i.track_id == 4)
        self.assertIn(sec_ins.severity, ("high", "critical"))
        self.assertTrue(sec_ins.details["is_restricted_zone"])

        # Track 5 short_zone loitering
        short_ins = next(i for i in insights if i.track_id == 5)
        self.assertEqual(short_ins.zone_id, "short_zone")

    def test_detect_wrong_way_insights(self) -> None:
        cfg = InsightConfig(
            allowed_line_directions={"entry_turnstile": "A->B"},
            disallowed_line_crossings=("security_line:B->A",),
            restricted_zones=("security_line",),
        )
        events = [
            # Allowed crossing
            DummyEvent("e1", "line_crossing", 1, "entry_turnstile:A->B", 2.0, 2.0),
            # Wrong-way crossing
            DummyEvent("e2", "line_crossing", 2, "entry_turnstile:B->A", 5.0, 5.0),
            # Disallowed crossing
            DummyEvent("e3", "line_crossing", 3, "security_line:B->A", 10.0, 10.0),
        ]
        insights = detect_wrong_way_insights(events, cfg, camera_id="c1")
        self.assertEqual(len(insights), 2)

        t2_ins = next(i for i in insights if i.track_id == 2)
        self.assertEqual(t2_ins.insight_type, "wrong_way")
        self.assertEqual(t2_ins.details["expected_direction"], "A->B")

        t3_ins = next(i for i in insights if i.track_id == 3)
        self.assertEqual(t3_ins.severity, "high")  # restricted zone

    def test_detect_speed_anomaly_insights(self) -> None:
        cfg = InsightConfig(
            speed_anomaly_threshold_px_s=100.0,
            min_speed_anomaly_frames=3,
        )
        # Track 1: normal walking speed (approx 20 px/s)
        t1_pts = [DummyPoint(1, i, i * 0.1, 10.0 + i * 2.0, 10.0) for i in range(10)]
        # Track 2: sudden sprint (200 px/s over 5 frames)
        t2_pts = [
            DummyPoint(2, i, i * 0.1, 10.0 + (i * 20.0 if i >= 2 else i * 2.0), 10.0)
            for i in range(8)
        ]
        trajectories = {1: t1_pts, 2: t2_pts}
        insights = detect_speed_anomaly_insights(trajectories, cfg, camera_id="c1")
        self.assertEqual(len(insights), 1)
        ins = insights[0]
        self.assertEqual(ins.track_id, 2)
        self.assertEqual(ins.insight_type, "high_speed")
        self.assertGreater(ins.details["max_speed_px_s"], 150.0)

    def test_detect_stationary_and_drop_insights(self) -> None:
        cfg = InsightConfig(
            stationary_speed_threshold_px_s=5.0,
            stationary_min_seconds=5.0,
            restricted_zones=("vault",),
        )
        events = [
            # Dwell event for 8s with stationary coordinates
            DummyEvent("e1", "dwell_time_event", 1, "vault", 0.0, 8.0, 8.0),
        ]
        # Stationary track coordinates
        pts = [DummyPoint(1, i, i * 1.0, 100.0 + (i * 0.1), 100.0) for i in range(9)]
        traces = [
            # Track 2 was lost in vault
            DummyTrace("stay_02", 2, "vault", 10.0, 15.0, track_lost_in_zone=True),
        ]

        insights = detect_stationary_and_drop_insights(
            events,
            {1: pts},
            cfg,
            traces=traces,
            camera_id="c1",
        )
        self.assertEqual(len(insights), 2)
        types = {i.insight_type for i in insights}
        self.assertIn("stationary_in_zone", types)
        self.assertIn("track_drop_in_zone", types)

    def test_detect_cross_camera_transitions(self) -> None:
        cfg = InsightConfig(
            cross_camera_min_seconds=1.0,
            cross_camera_max_seconds=10.0,
        )
        events_by_camera = {
            "cam_entrance": [
                DummyEvent("e1", "zone_exit", 1, "turnstile", 5.0, 5.0),
            ],
            "cam_hallway": [
                # Entity arrives 3s later
                DummyEvent("e2", "zone_entry", 10, "hall_entry", 8.0, 8.0),
            ],
        }
        transitions = detect_cross_camera_transitions(events_by_camera, cfg)
        self.assertEqual(len(transitions), 1)
        trans = transitions[0]
        self.assertEqual(trans.camera_id, "cam_entrance->cam_hallway")
        self.assertEqual(trans.insight_type, "cross_camera_transition")
        self.assertEqual(trans.duration, 3.0)

    def test_generate_all_insights_aggregates_and_sorts(self) -> None:
        cfg = InsightConfig(
            min_loitering_seconds=5.0,
            allowed_line_directions={"gate": "A->B"},
        )
        events = [
            DummyEvent("e1", "line_crossing", 2, "gate:B->A", 10.0, 10.0),
            DummyEvent("e2", "dwell_time_event", 1, "lounge", 2.0, 9.0, 7.0),
        ]
        insights = generate_all_insights(events, {}, cfg, camera_id="cam_main")
        self.assertEqual(len(insights), 2)
        # Should be sorted chronologically by start_time: e2 (t=2.0) before e1 (t=10.0)
        self.assertEqual(insights[0].start_time, 2.0)
        self.assertEqual(insights[1].start_time, 10.0)


if __name__ == "__main__":
    unittest.main()
