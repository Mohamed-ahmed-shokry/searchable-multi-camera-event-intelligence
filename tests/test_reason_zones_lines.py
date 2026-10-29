"""Tests for Phase 5 Zone and Line Reasoning: scene config serialization,
point-in-polygon evaluation, directional line crossings, boundary flicker
analysis, and spatial video rendering.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from helpers import PROJECT_ROOT
from reason_zones_lines import (
    ZONE_LINE_EVENTS_CSV_HEADER,
    LineSpec,
    SceneConfig,
    ZoneLineSignal,
    ZoneSpec,
    load_scene_config,
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


if __name__ == "__main__":
    unittest.main()
