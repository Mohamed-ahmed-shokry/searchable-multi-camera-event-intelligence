"""Tests for Phase 4 trajectory extraction data structures, anchor computation,
CSV serialization, rolling-mean smoothing, path summaries, and rendering.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from compare_smoothing import compare_smoothing
from compare_smoothing import main as compare_smoothing_main
from extract_trajectories import (
    TRAJECTORIES_CSV_HEADER,
    TRAJECTORY_SUMMARY_CSV_HEADER,
    TrajectoryPoint,
    analyze_trajectory_quality,
    compute_anchor_point,
    compute_trajectory_summaries,
    draw_trajectory_trails,
    load_trajectory_config,
    main,
    read_trajectories_csv,
    read_trajectory_summary_csv,
    render_trajectory_map,
    run_trajectory_extraction,
    smooth_trajectories,
    tracks_to_trajectories,
    write_trajectories_csv,
    write_trajectory_summary_csv,
)
from helpers import PROJECT_ROOT, make_sequence
from track_people import FrameTracks, TrackDetection, write_tracks_csv


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


class SmoothTrajectoriesTests(unittest.TestCase):
    def test_smooth_trajectories_rolling_mean(self):
        raw = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 10.0, 100.0),
                TrajectoryPoint(1, 1, 0.1, 20.0, 106.0),
                TrajectoryPoint(1, 2, 0.2, 30.0, 100.0),
                TrajectoryPoint(1, 3, 0.3, 40.0, 106.0),
            ]
        }
        w1 = smooth_trajectories(raw, window=1)
        self.assertEqual(w1[1], raw[1])

        w3 = smooth_trajectories(raw, window=3)
        self.assertAlmostEqual(w3[1][0].x, 10.0)
        self.assertAlmostEqual(w3[1][1].x, 15.0)
        self.assertAlmostEqual(w3[1][2].x, 20.0)
        self.assertAlmostEqual(w3[1][2].y, 102.0)
        self.assertAlmostEqual(w3[1][3].y, 104.0)

        with self.assertRaises(ValueError):
            smooth_trajectories(raw, window=0)


class TrajectorySummaryTests(unittest.TestCase):
    def test_compute_trajectory_summaries_and_csv_round_trip(self):
        self.assertEqual(
            TRAJECTORY_SUMMARY_CSV_HEADER,
            ["track_id", "path_length_px", "duration_seconds", "frame_count"],
        )
        # 3-4-5 right triangles: (0,0) -> (3,4) -> (6,8) => path length = 5 + 5 = 10 px
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 0.0, 0.0),
                TrajectoryPoint(1, 1, 0.1, 3.0, 4.0),
                TrajectoryPoint(1, 3, 0.3, 6.0, 8.0),
            ],
            2: [
                TrajectoryPoint(2, 5, 0.5, 50.0, 50.0),
            ],
        }
        summaries = compute_trajectory_summaries(trajs, fps=10.0)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0].track_id, 1)
        self.assertAlmostEqual(summaries[0].path_length_px, 10.0)
        self.assertAlmostEqual(summaries[0].duration_seconds, 0.4)  # (3 - 0 + 1) / 10
        self.assertEqual(summaries[0].frame_count, 3)

        self.assertEqual(summaries[1].track_id, 2)
        self.assertAlmostEqual(summaries[1].path_length_px, 0.0)
        self.assertAlmostEqual(summaries[1].duration_seconds, 0.1)
        self.assertEqual(summaries[1].frame_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            summary_csv = Path(tmp) / "track_summary.csv"
            write_trajectory_summary_csv(summary_csv, summaries)
            loaded = read_trajectory_summary_csv(summary_csv)
            self.assertEqual(len(loaded), 2)
            self.assertAlmostEqual(loaded[0].path_length_px, 10.0)


class TrajectoryQualityTests(unittest.TestCase):
    def test_analyze_trajectory_quality_detects_teleport_and_jitter_reduction(self):
        raw = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 10.0, 100.0),
                TrajectoryPoint(1, 1, 0.1, 20.0, 108.0),
                TrajectoryPoint(1, 2, 0.2, 30.0, 100.0),
                TrajectoryPoint(1, 3, 0.3, 40.0, 108.0),
                TrajectoryPoint(1, 4, 0.4, 240.0, 108.0),  # 200 px jump
            ]
        }
        report_raw = analyze_trajectory_quality(raw, jump_threshold_px=150.0)
        self.assertEqual(report_raw["teleport_jump_count"], 1)
        smoothed = smooth_trajectories(
            {1: raw[1][:4]}, window=3
        )  # check jitter on oscillating segment
        report_osc_raw = analyze_trajectory_quality({1: raw[1][:4]})
        report_osc_smooth = analyze_trajectory_quality(smoothed)
        self.assertLess(
            float(report_osc_smooth["mean_jitter_px"]),  # type: ignore[arg-type]
            float(report_osc_raw["mean_jitter_px"]),  # type: ignore[arg-type]
        )


class TrajectoryRenderingTests(unittest.TestCase):
    def test_draw_trajectory_trails_and_render_trajectory_map(self):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 20.0, 100.0),
                TrajectoryPoint(1, 1, 0.1, 60.0, 105.0),
                TrajectoryPoint(1, 2, 0.2, 100.0, 110.0),
            ]
        }
        boxes = [TrackDetection(2, 1, 80.0, 50.0, 120.0, 110.0, 0.9)]
        trailed = draw_trajectory_trails(
            frame,
            trajs,
            current_frame_id=2,
            max_trail_length=10,
            current_boxes=boxes,
        )
        self.assertEqual(trailed.shape, frame.shape)
        self.assertGreater(int(trailed.sum()), 0)

        tmap = render_trajectory_map(frame, trajs)
        self.assertEqual(tmap.shape, frame.shape)
        self.assertGreater(int(tmap.sum()), 0)


class RunTrajectoryExtractionTests(unittest.TestCase):
    def test_load_trajectory_config_and_run_pipeline_and_cli(self):
        cfg_path = (
            PROJECT_ROOT
            / "cv_projects"
            / "04_trajectory_extraction"
            / "configs"
            / "trajectory.yaml"
        )
        cfg = load_trajectory_config(cfg_path)
        self.assertEqual(cfg["anchor"], "bottom_center")
        self.assertEqual(cfg["smoothing_window"], 3)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq_dir = make_sequence(root, count=4)
            tracks_csv = root / "tracks.csv"
            frame_tracks = [
                FrameTracks(0, [TrackDetection(0, 1, 4.0, 10.0, 14.0, 40.0, 0.9)]),
                FrameTracks(1, [TrackDetection(1, 1, 8.0, 10.0, 18.0, 40.0, 0.9)]),
                FrameTracks(2, [TrackDetection(2, 1, 12.0, 10.0, 22.0, 40.0, 0.9)]),
                FrameTracks(3, [TrackDetection(3, 1, 16.0, 10.0, 26.0, 40.0, 0.9)]),
            ]
            write_tracks_csv(tracks_csv, frame_tracks)

            out_dir = root / "outputs"
            report = run_trajectory_extraction(
                seq_dir,
                out_dir,
                tracks_csv=tracks_csv,
                smoothing_window=1,
            )
            self.assertEqual(report["num_unique_tracks"], 1)
            self.assertAlmostEqual(report["total_path_length_px"], 12.0)
            self.assertTrue((out_dir / "trajectories.csv").exists())
            self.assertTrue((out_dir / "track_summary.csv").exists())
            self.assertTrue((out_dir / "trajectory_map.png").exists())
            self.assertTrue((out_dir / "trajectory_overlay_video.mp4").exists())

            cli_out = root / "cli_outputs"
            rc = main(
                [
                    str(seq_dir),
                    "--tracks-csv",
                    str(tracks_csv),
                    "-o",
                    str(cli_out),
                    "--config",
                    str(cfg_path),
                    "--smoothing-window",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((cli_out / "trajectory_extraction_summary.json").exists())


class CompareSmoothingTests(unittest.TestCase):
    def test_compare_smoothing_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tracks_csv = root / "tracks.csv"
            frame_tracks = [
                FrameTracks(0, [TrackDetection(0, 1, 10.0, 20.0, 30.0, 80.0, 0.9)]),
                FrameTracks(1, [TrackDetection(1, 1, 20.0, 20.0, 40.0, 86.0, 0.9)]),
                FrameTracks(2, [TrackDetection(2, 1, 30.0, 20.0, 50.0, 80.0, 0.9)]),
                FrameTracks(3, [TrackDetection(3, 1, 40.0, 20.0, 60.0, 86.0, 0.9)]),
                FrameTracks(4, [TrackDetection(4, 1, 50.0, 20.0, 70.0, 80.0, 0.9)]),
            ]
            write_tracks_csv(tracks_csv, frame_tracks)
            out_dir = root / "cmp"
            report = compare_smoothing(
                tracks_csv,
                out_dir,
                fps=10.0,
                windows=[1, 3, 5],
            )
            bc = report["anchors"]["bottom_center"]
            self.assertGreater(bc["window_3"]["jitter_reduction_pct"], 0.0)
            self.assertTrue((out_dir / "smoothing_comparison.json").exists())

            rc = compare_smoothing_main(
                [
                    str(tracks_csv),
                    "-o",
                    str(root / "cmp_cli"),
                    "--fps",
                    "10.0",
                    "--windows",
                    "1,3",
                ]
            )
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
