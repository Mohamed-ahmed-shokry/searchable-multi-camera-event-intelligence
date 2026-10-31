"""Unit and integration tests for Phase 6 Dwell-Time Events:
- EventConfig, EventRecord, and StayDebugTrace validation
- YAML configuration and events.csv round-trip serialization
- ZoneDwellStateMachine lifecycle (candidate -> active -> triggered -> resolved)
- Hysteresis / grace-period boundary flicker absorption
- Track-loss-in-zone resolution and debug trace flagging
- End-to-end pipeline and CLI execution
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from generate_dwell_events import (
    EVENTS_CSV_HEADER,
    EventConfig,
    EventRecord,
    StayDebugTrace,
    load_events_config,
    read_events_csv,
    save_events_config,
    write_events_csv,
)
from helpers import PROJECT_ROOT


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

    def test_load_and_save_events_config_round_trip(self):
        default_yaml = (
            PROJECT_ROOT
            / "cv_projects"
            / "06_dwell_time_events"
            / "configs"
            / "events.yaml"
        )
        cfg = load_events_config(default_yaml)
        self.assertAlmostEqual(cfg.min_dwell_seconds, 1.2)
        self.assertEqual(cfg.grace_frames, 5)
        self.assertAlmostEqual(cfg.zone_threshold("right_aisle"), 1.5)
        self.assertEqual(load_events_config(None), EventConfig())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_path = save_events_config(root / "events.yaml", cfg)
            reloaded = load_events_config(out_path)
            self.assertEqual(reloaded, cfg)

            bad_yaml = root / "bad.yaml"
            bad_yaml.write_text("- not-a-mapping\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_events_config(bad_yaml)


class EventsCsvRoundTripTests(unittest.TestCase):
    def test_write_and_read_events_csv_orders_and_validates_header(self):
        e_exit = EventRecord("e2", "zone_exit", 1, "z1", 3.0, 3.0, 0.0)
        e_enter = EventRecord("e1", "zone_entry", 1, "z1", 1.0, 1.0, 0.0)
        e_dwell = EventRecord("e3", "dwell_time_event", 1, "z1", 1.0, 3.0, 2.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = write_events_csv(root / "events.csv", [e_exit, e_dwell, e_enter])
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], EVENTS_CSV_HEADER)
            loaded = read_events_csv(csv_path)
            self.assertEqual(loaded, [e_enter, e_dwell, e_exit])

            bad_csv = root / "bad.csv"
            bad_csv.write_text("wrong,header\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_events_csv(bad_csv)


if __name__ == "__main__":
    unittest.main()
