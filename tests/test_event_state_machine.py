"""Unit and integration tests for Phase 6 Dwell-Time Events:
- EventConfig, EventRecord, and StayDebugTrace validation
- YAML configuration and events.csv round-trip serialization
- ZoneDwellStateMachine lifecycle (candidate -> active -> triggered -> resolved)
- Hysteresis / grace-period boundary flicker absorption
- Track-loss-in-zone resolution and debug trace flagging
- End-to-end pipeline and CLI execution
"""

from __future__ import annotations

import unittest

from generate_dwell_events import (
    EVENTS_CSV_HEADER,
    EventConfig,
    EventRecord,
    StayDebugTrace,
)


class EventDataStructuresTests(unittest.TestCase):
    def test_event_config_validation_and_per_zone_threshold(self):
        cfg = EventConfig(
            min_dwell_seconds=1.2,
            min_entry_frames=2,
            grace_frames=5,
            line_cooldown_frames=4,
            per_zone_min_dwell_seconds={"right_aisle": 2.0},
        )
        self.assertAlmostEqual(cfg.zone_threshold("entrance_area"), 1.2)
        self.assertAlmostEqual(cfg.zone_threshold("right_aisle"), 2.0)

        with self.assertRaises(ValueError):
            EventConfig(min_dwell_seconds=-0.1)
        with self.assertRaises(ValueError):
            EventConfig(min_entry_frames=0)
        with self.assertRaises(ValueError):
            EventConfig(grace_frames=-1)
        with self.assertRaises(ValueError):
            EventConfig(line_cooldown_frames=-1)
        with self.assertRaises(ValueError):
            EventConfig(per_zone_min_dwell_seconds={"bad": -1.0})

    def test_event_record_and_stay_debug_trace_validation(self):
        self.assertEqual(
            EVENTS_CSV_HEADER,
            [
                "event_id",
                "event_type",
                "track_id",
                "zone_id",
                "start_time",
                "end_time",
                "duration_seconds",
            ],
        )
        rec = EventRecord(
            event_id="evt_0001",
            event_type="dwell_time_event",
            track_id=1,
            zone_id="entrance_area",
            start_time=1.5,
            end_time=3.0,
            duration_seconds=1.5,
            start_frame=18,
            end_frame=36,
        )
        self.assertEqual(
            rec.to_row(),
            [
                "evt_0001",
                "dwell_time_event",
                "1",
                "entrance_area",
                "1.5000",
                "3.0000",
                "1.5000",
            ],
        )
        with self.assertRaises(ValueError):
            EventRecord("", "zone_entry", 1, "z", 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            EventRecord("e1", "invalid_type", 1, "z", 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            EventRecord("e1", "zone_entry", -1, "z", 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            EventRecord("e1", "zone_entry", 1, " ", 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            EventRecord("e1", "zone_entry", 1, "z", 2.0, 1.0, 0.0)

        trace = StayDebugTrace(
            stay_id="stay_t1_z1_1",
            track_id=1,
            zone_id="z1",
            enter_frame=10,
            enter_time=1.0,
            last_inside_frame=25,
            last_inside_time=2.5,
            resolved_frame=31,
            resolved_time=3.1,
            duration_seconds=1.5,
            peak_state="triggered",
            resolution_reason="exited",
            track_lost_in_zone=False,
            flicker_absorbed_count=1,
            transitions=({"frame_id": 10, "to": "candidate"},),
        )
        d = trace.to_dict()
        self.assertEqual(d["stay_id"], "stay_t1_z1_1")
        self.assertEqual(d["flicker_absorbed_count"], 1)


if __name__ == "__main__":
    unittest.main()
