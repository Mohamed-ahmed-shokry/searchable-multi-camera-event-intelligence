"""Tests for Phase 5 Zone and Line Reasoning: scene config serialization,
point-in-polygon evaluation, directional line crossings, boundary flicker
analysis, and spatial video rendering.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import PROJECT_ROOT
from reason_zones_lines import (
    LineSpec,
    SceneConfig,
    ZoneSpec,
    load_scene_config,
    save_scene_config,
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


if __name__ == "__main__":
    unittest.main()
