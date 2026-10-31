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
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from generate_dwell_events import (
    EVENTS_CSV_HEADER,
    EventConfig,
    EventRecord,
    StayDebugTrace,
    ZoneDwellStateMachine,
    draw_dwell_events_overlay,
    evaluate_dwell_events,
    load_events_config,
    read_events_csv,
    save_dwell_examples,
    save_events_config,
    write_event_debug_logs,
    write_events_csv,
)
from helpers import PROJECT_ROOT, make_video
from reason_zones_lines import LineSpec, SceneConfig, TrajectoryPoint, ZoneSpec


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


class ZoneDwellStateMachineLifecycleTests(unittest.TestCase):
    def test_normal_stay_reaching_dwell_threshold_emits_entry_dwell_and_exit(self):
        cfg = EventConfig(
            min_dwell_seconds=1.0,
            min_entry_frames=2,
            grace_frames=1,
        )
        sm = ZoneDwellStateMachine(track_id=1, zone_id="lobby", config=cfg)

        # Frame 0: outside
        self.assertEqual(sm.step(0, 0.0, is_inside=False), [])
        self.assertEqual(sm.state, "outside")

        # Frame 1: first inside frame -> candidate (min_entry_frames=2 not yet met)
        self.assertEqual(sm.step(1, 0.5, is_inside=True), [])
        self.assertEqual(sm.state, "candidate")

        # Frame 2: second inside frame -> promoted to active, emits zone_entry at t=0.5
        ev_f2 = sm.step(2, 1.0, is_inside=True)
        self.assertEqual(sm.state, "active")
        self.assertEqual(len(ev_f2), 1)
        self.assertEqual(ev_f2[0].event_type, "zone_entry")
        self.assertAlmostEqual(ev_f2[0].start_time, 0.5)

        # Frame 3: t=1.5 -> duration = 1.5 - 0.5 = 1.0s >= 1.0s -> triggered
        self.assertEqual(sm.step(3, 1.5, is_inside=True), [])
        self.assertEqual(sm.state, "triggered")

        # Frame 4: t=2.0 -> still inside, accumulating dwell (1.5s)
        self.assertEqual(sm.step(4, 2.0, is_inside=True), [])
        self.assertAlmostEqual(sm.current_dwell_seconds(), 1.5)

        # Frame 5: first outside frame (within grace_frames=1 -> not resolved yet)
        self.assertEqual(sm.step(5, 2.5, is_inside=False), [])
        self.assertEqual(sm.state, "triggered")

        # Frame 6: second outside frame (outside_streak=2 > grace_frames=1) -> resolved!
        ev_f6 = sm.step(6, 3.0, is_inside=False)
        self.assertEqual(sm.state, "outside")
        self.assertEqual(
            [e.event_type for e in ev_f6], ["dwell_time_event", "zone_exit"]
        )
        dwell_ev, exit_ev = ev_f6[0], ev_f6[1]
        self.assertAlmostEqual(dwell_ev.start_time, 0.5)
        self.assertAlmostEqual(dwell_ev.end_time, 2.0)
        self.assertAlmostEqual(dwell_ev.duration_seconds, 1.5)
        self.assertAlmostEqual(exit_ev.start_time, 2.5)
        self.assertEqual(len(sm.completed_traces), 1)
        self.assertEqual(sm.completed_traces[0].peak_state, "triggered")
        self.assertEqual(sm.completed_traces[0].resolution_reason, "exited")

    def test_short_pass_through_emits_entry_and_exit_without_dwell_event(self):
        cfg = EventConfig(min_dwell_seconds=2.0, min_entry_frames=1, grace_frames=0)
        sm = ZoneDwellStateMachine(track_id=2, zone_id="corridor", config=cfg)

        ev1 = sm.step(10, 1.0, is_inside=True)
        self.assertEqual([e.event_type for e in ev1], ["zone_entry"])
        self.assertEqual(sm.state, "active")

        self.assertEqual(sm.step(11, 1.5, is_inside=True), [])
        ev2 = sm.step(12, 2.0, is_inside=False)
        self.assertEqual([e.event_type for e in ev2], ["zone_exit"])
        self.assertEqual(sm.completed_traces[0].peak_state, "active")


class HysteresisAndFlickerTests(unittest.TestCase):
    def test_grace_period_absorbs_boundary_flicker_without_resetting_dwell(self):
        cfg = EventConfig(
            min_dwell_seconds=1.0,
            min_entry_frames=1,
            grace_frames=3,
        )
        sm = ZoneDwellStateMachine(track_id=3, zone_id="entrance_area", config=cfg)

        emitted: list[EventRecord] = []
        # Enter at frame 0 (t=0.0)
        emitted.extend(sm.step(0, 0.0, is_inside=True))
        emitted.extend(sm.step(1, 0.4, is_inside=True))
        # Boundary flicker: 3 frames outside (<= grace_frames=3)
        emitted.extend(sm.step(2, 0.5, is_inside=False))
        emitted.extend(sm.step(3, 0.6, is_inside=False))
        emitted.extend(sm.step(4, 0.7, is_inside=False))
        # Re-enter at frame 5 (t=1.2) -> flicker absorbed, dwell reaches 1.2s >= 1.0s!
        emitted.extend(sm.step(5, 1.2, is_inside=True))
        self.assertEqual(sm.state, "triggered")
        self.assertEqual(sm.flicker_absorbed_count, 1)

        # Genuine exit: 4 frames outside (> grace_frames=3), first outside at frame 6 (t=1.5)
        for fid, ts in [(6, 1.5), (7, 1.6), (8, 1.7), (9, 1.8)]:
            emitted.extend(sm.step(fid, ts, is_inside=False))

        self.assertEqual(
            [e.event_type for e in emitted],
            ["zone_entry", "dwell_time_event", "zone_exit"],
        )
        dwell = emitted[1]
        self.assertAlmostEqual(dwell.start_time, 0.0)
        self.assertAlmostEqual(dwell.end_time, 1.2)
        self.assertAlmostEqual(dwell.duration_seconds, 1.2)
        self.assertEqual(sm.completed_traces[0].flicker_absorbed_count, 1)

    def test_sub_entry_spike_discarded_as_candidate_without_events(self):
        cfg = EventConfig(
            min_dwell_seconds=1.0,
            min_entry_frames=3,
            grace_frames=1,
        )
        sm = ZoneDwellStateMachine(track_id=4, zone_id="lobby", config=cfg)
        self.assertEqual(sm.step(0, 0.0, is_inside=True), [])
        self.assertEqual(sm.state, "candidate")
        self.assertEqual(sm.step(1, 0.1, is_inside=False), [])
        self.assertEqual(sm.step(2, 0.2, is_inside=False), [])
        self.assertEqual(sm.state, "outside")
        self.assertEqual(len(sm.completed_traces), 1)
        self.assertEqual(
            sm.completed_traces[0].resolution_reason,
            "discarded_candidate",
        )


class TrackLossInZoneTests(unittest.TestCase):
    def test_finish_track_resolves_open_stay_at_last_inside_time_and_flags_loss(self):
        cfg = EventConfig(
            min_dwell_seconds=1.5,
            min_entry_frames=1,
            grace_frames=5,
        )
        sm = ZoneDwellStateMachine(track_id=7, zone_id="right_aisle", config=cfg)

        emitted: list[EventRecord] = []
        emitted.extend(sm.step(10, 1.0, is_inside=True))
        emitted.extend(sm.step(20, 2.0, is_inside=True))
        emitted.extend(sm.step(30, 3.0, is_inside=True))  # duration 2.0s >= 1.5s
        self.assertEqual(sm.state, "triggered")

        # Track disappears after 1 outside frame (within grace_frames=5)
        emitted.extend(sm.step(31, 3.1, is_inside=False))
        emitted.extend(sm.finish_track())

        self.assertEqual(sm.state, "outside")
        self.assertEqual(
            [e.event_type for e in emitted],
            ["zone_entry", "dwell_time_event"],
        )
        dwell = emitted[1]
        self.assertAlmostEqual(dwell.start_time, 1.0)
        self.assertAlmostEqual(dwell.end_time, 3.0)
        self.assertAlmostEqual(dwell.duration_seconds, 2.0)

        trace = sm.completed_traces[0]
        self.assertTrue(trace.track_lost_in_zone)
        self.assertEqual(trace.resolution_reason, "track_lost_in_zone")
        self.assertEqual(trace.resolved_frame, 30)
        self.assertAlmostEqual(trace.resolved_time, 3.0)


class EvaluateDwellEventsTests(unittest.TestCase):
    def test_evaluate_dwell_events_combines_zones_and_deduplicates_line_jitter(self):
        scene = SceneConfig(
            zones=(
                ZoneSpec(
                    id="zone_a",
                    polygon=((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)),
                ),
            ),
            lines=(LineSpec(id="gate", points=((50.0, 0.0), (50.0, 60.0))),),
        )
        cfg = EventConfig(
            min_dwell_seconds=1.0,
            min_entry_frames=1,
            grace_frames=1,
            line_cooldown_frames=3,
            include_line_crossings=True,
        )
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 20.0, 30.0),  # inside zone_a
                TrajectoryPoint(1, 1, 0.5, 30.0, 30.0),  # inside zone_a
                TrajectoryPoint(
                    1, 2, 1.2, 40.0, 30.0
                ),  # inside zone_a (1.2s -> triggered)
                TrajectoryPoint(1, 3, 1.5, 60.0, 30.0),  # cross gate (B->A), outside 1
                TrajectoryPoint(
                    1, 4, 1.6, 45.0, 30.0
                ),  # re-enter (flicker absorbed! line cross ignored by cooldown!)
                TrajectoryPoint(
                    1, 5, 2.0, 65.0, 30.0
                ),  # outside 1 (cross within cooldown)
                TrajectoryPoint(
                    1, 6, 2.2, 70.0, 30.0
                ),  # outside 2 (> grace_frames=1 -> resolved!)
            ]
        }
        events, traces = evaluate_dwell_events(trajs, scene, cfg)
        self.assertEqual(
            [e.event_id for e in events],
            ["evt_0001", "evt_0002", "evt_0003", "evt_0004"],
        )
        self.assertEqual(
            [e.event_type for e in events],
            ["zone_entry", "dwell_time_event", "line_crossing", "zone_exit"],
        )
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0].flicker_absorbed_count, 1)


class EventDebugLogTests(unittest.TestCase):
    def test_write_event_debug_logs_creates_combined_and_individual_trace_files(self):
        trace = StayDebugTrace(
            stay_id="stay_t1_zone_a_001",
            track_id=1,
            zone_id="zone_a",
            enter_frame=5,
            enter_time=0.5,
            last_inside_frame=20,
            last_inside_time=2.0,
            resolved_frame=26,
            resolved_time=2.6,
            duration_seconds=1.5,
            peak_state="triggered",
            resolution_reason="exited",
            track_lost_in_zone=False,
            flicker_absorbed_count=0,
            transitions=(
                {"frame_id": 5, "from": "outside", "to": "candidate"},
                {"frame_id": 5, "from": "candidate", "to": "active"},
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            debug_dir = Path(tmp) / "event_debug_log"
            combined_path = write_event_debug_logs(debug_dir, [trace])
            self.assertTrue(combined_path.exists())
            single_path = debug_dir / "stay_t1_zone_a_001.json"
            self.assertTrue(single_path.exists())

            loaded_all = json.loads(combined_path.read_text(encoding="utf-8"))
            self.assertEqual(len(loaded_all), 1)
            self.assertEqual(loaded_all[0]["stay_id"], "stay_t1_zone_a_001")


class DwellVisualOverlayTests(unittest.TestCase):
    def test_draw_dwell_events_overlay_and_save_dwell_examples(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        scene = SceneConfig(
            zones=(
                ZoneSpec(
                    id="z1",
                    polygon=((10.0, 10.0), (50.0, 10.0), (50.0, 45.0), (10.0, 45.0)),
                ),
            ),
            lines=(LineSpec(id="l1", points=((50.0, 5.0), (50.0, 45.0))),),
        )
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 20.0, 25.0),
                TrajectoryPoint(1, 1, 0.5, 30.0, 25.0),
                TrajectoryPoint(1, 2, 1.2, 40.0, 25.0),
                TrajectoryPoint(1, 3, 1.5, 58.0, 25.0),
            ]
        }
        events, traces = evaluate_dwell_events(
            trajs,
            scene,
            EventConfig(min_dwell_seconds=1.0, min_entry_frames=1, grace_frames=0),
        )
        rendered = draw_dwell_events_overlay(
            frame,
            scene,
            trajs,
            current_frame_id=2,
            current_timestamp=1.2,
            traces=traces,
            events=events,
        )
        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(rendered.sum()), 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = make_video(root, count=4)
            ex_dir = root / "dwell_examples"
            saved = save_dwell_examples(
                video_path,
                ex_dir,
                scene,
                trajs,
                traces,
                events,
            )
            self.assertEqual(len(saved), 1)
            self.assertTrue(saved[0].exists())


if __name__ == "__main__":
    unittest.main()
