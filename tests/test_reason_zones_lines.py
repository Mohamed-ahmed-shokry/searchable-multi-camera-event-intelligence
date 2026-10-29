"""Tests for Phase 5 Zone and Line Reasoning: scene config serialization,
point-in-polygon evaluation, directional line crossings, boundary flicker
analysis, and spatial video rendering.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from extract_trajectories import TrajectoryPoint
from helpers import PROJECT_ROOT
from reason_zones_lines import (
    ZONE_LINE_EVENTS_CSV_HEADER,
    LineSpec,
    SceneConfig,
    ZoneLineSignal,
    ZoneSpec,
    check_line_crossing,
    detect_zone_signals,
    load_scene_config,
    point_in_zone,
    read_zone_line_events_csv,
    save_scene_config,
    write_zone_line_events_csv,
)


class SpatialGeometrySpecsTests(unittest.TestCase):
    def test_zone_spec_validates_id_vertices_and_area(self):
        z = ZoneSpec(
            id="entrance",
            polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)),
        )
        self.assertAlmostEqual(z.shapely_polygon.area, 8000.0)

        with self.assertRaises(ValueError):
            ZoneSpec(id="", polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))
        with self.assertRaises(ValueError):
            ZoneSpec(id="bad", polygon=((0.0, 0.0), (10.0, 0.0)))
        with self.assertRaises(ValueError):
            # Collinear points -> zero area
            ZoneSpec(id="collinear", polygon=((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)))

    def test_line_spec_validates_id_endpoints_and_length(self):
        line = LineSpec(id="doorway", points=((0.0, 0.0), (30.0, 40.0)))
        self.assertAlmostEqual(line.shapely_line.length, 50.0)

        with self.assertRaises(ValueError):
            LineSpec(id=" ", points=((0.0, 0.0), (10.0, 10.0)))
        with self.assertRaises(ValueError):
            LineSpec(id="zero", points=((5.0, 5.0), (5.0, 5.0)))

        cfg = SceneConfig(
            zones=(z := ZoneSpec("z", ((0, 0), (1, 0), (0, 1))),),
            lines=(line,),
        )
        self.assertEqual(len(cfg.zones), 1)
        self.assertEqual(cfg.zones[0], z)

    def test_load_and_save_scene_config_round_trip(self):
        default_cfg_path = (
            PROJECT_ROOT
            / "cv_projects"
            / "05_zone_line_reasoning"
            / "configs"
            / "scene_config.yaml"
        )
        cfg = load_scene_config(default_cfg_path)
        self.assertEqual(len(cfg.zones), 3)
        self.assertEqual(len(cfg.lines), 2)
        self.assertEqual(cfg.zones[0].id, "entrance_area")
        self.assertEqual(cfg.lines[0].id, "doorway_line")

        with tempfile.TemporaryDirectory() as tmp:
            out_yaml = Path(tmp) / "roundtrip.yaml"
            save_scene_config(out_yaml, cfg)
            reloaded = load_scene_config(out_yaml)
            self.assertEqual(reloaded, cfg)

            dup_zone = Path(tmp) / "dup_zone.yaml"
            dup_zone.write_text(
                "zones:\n"
                "  - id: z1\n    polygon: [[0,0],[10,0],[10,10]]\n"
                "  - id: z1\n    polygon: [[0,0],[20,0],[20,20]]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_scene_config(dup_zone)


class ZoneLineSignalCsvTests(unittest.TestCase):
    def test_signal_validation_and_csv_round_trip(self):
        self.assertEqual(
            ZONE_LINE_EVENTS_CSV_HEADER,
            [
                "frame_id",
                "track_id",
                "signal_type",
                "zone_id",
                "line_id",
                "direction",
                "timestamp",
            ],
        )
        s_enter = ZoneLineSignal(
            frame_id=10,
            track_id=1,
            signal_type="zone_enter",
            zone_id="entrance_area",
            timestamp=0.8333,
        )
        s_cross = ZoneLineSignal(
            frame_id=12,
            track_id=1,
            signal_type="line_cross",
            line_id="doorway_line",
            direction="A->B",
            timestamp=1.0,
        )
        with self.assertRaises(ValueError):
            ZoneLineSignal(frame_id=0, track_id=1, signal_type="zone_enter")
        with self.assertRaises(ValueError):
            ZoneLineSignal(
                frame_id=0,
                track_id=1,
                signal_type="line_cross",
                line_id="doorway",
            )

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "zone_line_events.csv"
            write_zone_line_events_csv(csv_path, [s_cross, s_enter])
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ZONE_LINE_EVENTS_CSV_HEADER)
            self.assertEqual(
                rows[1],
                ["10", "1", "zone_enter", "entrance_area", "", "", "0.8333"],
            )
            loaded = read_zone_line_events_csv(csv_path)
            self.assertEqual(loaded, [s_enter, s_cross])


class PointInZoneTests(unittest.TestCase):
    def test_point_in_zone_interior_boundary_and_exterior(self):
        zone = ZoneSpec(
            id="box",
            polygon=((10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)),
        )
        self.assertTrue(point_in_zone(30.0, 40.0, zone))
        self.assertTrue(point_in_zone(10.0, 20.0, zone))  # vertex on boundary
        self.assertTrue(point_in_zone(30.0, 60.0, zone))  # edge on boundary
        self.assertFalse(point_in_zone(9.9, 40.0, zone))
        self.assertFalse(point_in_zone(30.0, 60.1, zone))


class DetectZoneSignalsTests(unittest.TestCase):
    def test_detect_zone_signals_enter_and_exit_transitions(self):
        zone = ZoneSpec(
            id="lobby",
            polygon=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
        )
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 10.0, 50.0),  # outside
                TrajectoryPoint(1, 1, 0.1, 30.0, 50.0),  # enter
                TrajectoryPoint(1, 2, 0.2, 60.0, 50.0),  # stay inside
                TrajectoryPoint(1, 3, 0.3, 90.0, 50.0),  # exit
            ]
        }
        sigs = detect_zone_signals(trajs, [zone])
        self.assertEqual(len(sigs), 2)
        self.assertEqual(sigs[0].signal_type, "zone_enter")
        self.assertEqual(sigs[0].frame_id, 1)
        self.assertEqual(sigs[1].signal_type, "zone_exit")
        self.assertEqual(sigs[1].frame_id, 3)


class CheckLineCrossingTests(unittest.TestCase):
    def test_check_line_crossing_directions_and_finite_segment_bounds(self):
        # Horizontal line from (10, 50) to (100, 50)
        line = LineSpec(id="gate", points=((10.0, 50.0), (100.0, 50.0)))
        # Moving from y=40 (s < 0, Side A) to y=60 (s > 0, Side B) at x=50
        self.assertEqual(
            check_line_crossing((50.0, 40.0), (50.0, 60.0), line),
            "A->B",
        )
        # Moving from y=60 (Side B) to y=40 (Side A) at x=50
        self.assertEqual(
            check_line_crossing((50.0, 60.0), (50.0, 40.0), line),
            "B->A",
        )
        # Passing outside the finite segment at x=200 -> None
        self.assertIsNone(check_line_crossing((200.0, 40.0), (200.0, 60.0), line))
        # Staying on the same side -> None
        self.assertIsNone(check_line_crossing((50.0, 30.0), (50.0, 40.0), line))


if __name__ == "__main__":
    unittest.main()
