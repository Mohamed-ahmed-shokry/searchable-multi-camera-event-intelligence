"""Tests for Phase 4 trajectory extraction data structures, anchor computation,
CSV serialization, rolling-mean smoothing, path summaries, and rendering.
"""

from __future__ import annotations

import unittest

from extract_trajectories import (
    TRAJECTORIES_CSV_HEADER,
    TrajectoryPoint,
    compute_anchor_point,
)


class TrajectorySchemaTests(unittest.TestCase):
    def test_compute_anchor_point_bottom_center_and_centroid(self):
        bx, by = compute_anchor_point(10.0, 20.0, 50.0, 100.0, anchor="bottom_center")
        self.assertAlmostEqual(bx, 30.0)
        self.assertAlmostEqual(by, 100.0)

        cx, cy = compute_anchor_point(10.0, 20.0, 50.0, 100.0, anchor="centroid")
        self.assertAlmostEqual(cx, 30.0)
        self.assertAlmostEqual(cy, 60.0)

        with self.assertRaises(ValueError):
            compute_anchor_point(10.0, 20.0, 50.0, 100.0, anchor="top_left")

    def test_trajectory_point_validation_and_to_row(self):
        self.assertEqual(
            TRAJECTORIES_CSV_HEADER,
            ["track_id", "frame_id", "timestamp", "x", "y"],
        )
        pt = TrajectoryPoint(track_id=3, frame_id=12, timestamp=0.4, x=30.5, y=100.25)
        self.assertEqual(pt.to_row(), ["3", "12", "0.4000", "30.50", "100.25"])

        with self.assertRaises(ValueError):
            TrajectoryPoint(track_id=-1, frame_id=0, timestamp=0.0, x=0.0, y=0.0)
        with self.assertRaises(ValueError):
            TrajectoryPoint(track_id=1, frame_id=-1, timestamp=0.0, x=0.0, y=0.0)
        with self.assertRaises(ValueError):
            TrajectoryPoint(track_id=1, frame_id=0, timestamp=-0.1, x=0.0, y=0.0)


if __name__ == "__main__":
    unittest.main()
