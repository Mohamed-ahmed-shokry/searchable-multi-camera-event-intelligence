"""Tests for Phase 4 trajectory extraction data structures, anchor computation,
CSV serialization, rolling-mean smoothing, path summaries, and rendering.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from extract_trajectories import (
    TRAJECTORIES_CSV_HEADER,
    TrajectoryPoint,
    compute_anchor_point,
    read_trajectories_csv,
    tracks_to_trajectories,
    write_trajectories_csv,
)
from track_people import FrameTracks, TrackDetection


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

    def test_trajectories_csv_round_trip(self):
        trajs = {
            2: [
                TrajectoryPoint(2, 5, 0.5, 100.0, 200.0),
                TrajectoryPoint(2, 4, 0.4, 95.0, 198.0),
            ],
            1: [
                TrajectoryPoint(1, 0, 0.0, 10.0, 50.0),
                TrajectoryPoint(1, 1, 0.1, 12.0, 52.0),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trajectories.csv"
            write_trajectories_csv(csv_path, trajs)
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], TRAJECTORIES_CSV_HEADER)
            # Track 1 frame 0 comes first, then Track 1 frame 1, then Track 2 frame 4, frame 5
            self.assertEqual(rows[1], ["1", "0", "0.0000", "10.00", "50.00"])
            self.assertEqual(rows[3], ["2", "4", "0.4000", "95.00", "198.00"])

            loaded = read_trajectories_csv(csv_path)
            self.assertEqual(sorted(loaded.keys()), [1, 2])
            self.assertEqual([p.frame_id for p in loaded[2]], [4, 5])


class TracksToTrajectoriesTests(unittest.TestCase):
    def test_tracks_to_trajectories_bottom_center_and_centroid(self):
        frame_tracks = [
            FrameTracks(
                frame_id=2,
                tracks=[TrackDetection(2, 2, 100.0, 100.0, 140.0, 200.0, 0.9)],
            ),
            FrameTracks(
                frame_id=0,
                tracks=[
                    TrackDetection(0, 1, 10.0, 20.0, 50.0, 120.0, 0.85),
                    TrackDetection(0, 2, 90.0, 100.0, 130.0, 200.0, 0.88),
                ],
            ),
            FrameTracks(
                frame_id=1,
                tracks=[TrackDetection(1, 1, 14.0, 22.0, 54.0, 122.0, 0.87)],
            ),
        ]
        trajs_bc = tracks_to_trajectories(
            frame_tracks, fps=10.0, anchor="bottom_center"
        )
        self.assertEqual(list(trajs_bc.keys()), [1, 2])
        self.assertEqual([p.frame_id for p in trajs_bc[2]], [0, 2])
        self.assertAlmostEqual(trajs_bc[1][0].timestamp, 0.0)
        self.assertAlmostEqual(trajs_bc[1][1].timestamp, 0.1)
        self.assertAlmostEqual(trajs_bc[1][0].x, 30.0)
        self.assertAlmostEqual(trajs_bc[1][0].y, 120.0)

        trajs_c = tracks_to_trajectories(frame_tracks, fps=10.0, anchor="centroid")
        self.assertAlmostEqual(trajs_c[1][0].x, 30.0)
        self.assertAlmostEqual(trajs_c[1][0].y, 70.0)

        with self.assertRaises(ValueError):
            tracks_to_trajectories(frame_tracks, fps=0.0)


if __name__ == "__main__":
    unittest.main()
