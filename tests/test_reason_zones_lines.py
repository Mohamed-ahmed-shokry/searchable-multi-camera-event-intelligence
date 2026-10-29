"""Tests for Phase 5 Zone and Line Reasoning: scene config serialization,
point-in-polygon evaluation, directional line crossings, boundary flicker
analysis, and spatial video rendering.
"""

from __future__ import annotations

import unittest

from reason_zones_lines import LineSpec, SceneConfig, ZoneSpec


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
            zones=(z := ZoneSpec("z", ((0, 0), (1, 0), (0, 1))),), lines=(line,)
        )
        self.assertEqual(len(cfg.zones), 1)
        self.assertEqual(cfg.zones[0], z)


if __name__ == "__main__":
    unittest.main()
