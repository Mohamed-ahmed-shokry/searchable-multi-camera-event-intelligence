"""Unit and integration tests for Phase 11 CV Pipeline Integration (src/event_intelligence/)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import make_sequence, make_video

from event_intelligence.detection import (
    DETECTIONS_CSV_HEADER,
    Detection,
    FrameDetections,
    read_detections_csv,
    write_detections_csv,
)
from event_intelligence.tracking import (
    FrameTracks,
    TrackDetection,
    compute_track_summaries,
    evaluate_tracking,
    read_tracks_csv,
    write_tracks_csv,
)
from event_intelligence.trajectories import (
    compute_anchor_point,
    compute_trajectory_summaries,
    read_trajectories_csv,
    smooth_trajectories,
    tracks_to_trajectories,
    write_trajectories_csv,
)
from event_intelligence.video import (
    VideoSource,
    inspect_source,
    sample_frame_indices,
    validate_video_info,
)


class TestPromotedVideoPackage(unittest.TestCase):
    """Test event_intelligence.video promotion on video files and MOT sequences."""

    def test_video_source_and_inspect_source_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_path = make_video(root, count=6)
            seq_path = make_sequence(root, name="MOT17-11-FRCNN", count=5)

            with VideoSource(clip_path) as vs:
                self.assertEqual(len(vs), 6)
                self.assertEqual(vs.metadata.source_type, "video_file")

            indices = sample_frame_indices(6, 3)
            self.assertEqual(indices, [0, 2, 5])

            out_dir = root / "inspected"
            entry = inspect_source(clip_path, out_dir, sample_count=3)
            self.assertEqual(entry["frame_count"], 6)
            self.assertEqual(validate_video_info([entry]), [])

            with VideoSource(seq_path) as seq_vs:
                self.assertEqual(seq_vs.metadata.source_type, "frame_sequence")


class TestPromotedDetectionPackage(unittest.TestCase):
    """Test event_intelligence.detection CSV schema and round-trip."""

    def test_detections_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "detections.csv"
            records = [
                FrameDetections(0, [Detection(10.0, 20.0, 40.0, 80.0, 0.91)]),
                FrameDetections(1, []),
            ]
            total = write_detections_csv(records, csv_path)
            self.assertEqual(total, 1)
            rows = read_detections_csv(csv_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0].keys()), DETECTIONS_CSV_HEADER)
            self.assertAlmostEqual(rows[0]["confidence"], 0.91)
            self.assertIsNone(rows[1]["x1"])


class TestPromotedTrackingPackage(unittest.TestCase):
    """Test event_intelligence.tracking CSV round-trip, summaries, and CLEAR-MOT evaluation."""

    def test_tracks_csv_summaries_and_evaluate_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "tracks.csv"
            ft0 = FrameTracks(0, [TrackDetection(0, 1, 10.0, 20.0, 40.0, 80.0, 0.95)])
            ft1 = FrameTracks(1, [TrackDetection(1, 1, 12.0, 20.0, 42.0, 80.0, 0.92)])
            write_tracks_csv(csv_path, [ft0, ft1])
            loaded = read_tracks_csv(csv_path)
            self.assertEqual(len(loaded), 2)

            summaries = compute_track_summaries(loaded, fps=10.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].duration_frames, 2)
            self.assertAlmostEqual(summaries[0].duration_seconds, 0.2)

            gt = {
                0: [TrackDetection(0, 10, 10.0, 20.0, 40.0, 80.0, 1.0)],
                1: [TrackDetection(1, 10, 12.0, 20.0, 42.0, 80.0, 1.0)],
            }
            metrics = evaluate_tracking(gt, loaded)
            self.assertEqual(metrics["mota"], 1.0)
            self.assertEqual(metrics["idf1"], 1.0)
            self.assertEqual(metrics["id_switches"], 0)


class TestPromotedTrajectoriesPackage(unittest.TestCase):
    """Test event_intelligence.trajectories anchor math, smoothing, and path length."""

    def test_trajectories_extraction_smoothing_and_summary(self) -> None:
        ax, ay = compute_anchor_point(10.0, 20.0, 30.0, 80.0, anchor="bottom_center")
        self.assertEqual((ax, ay), (20.0, 80.0))

        ft0 = FrameTracks(0, [TrackDetection(0, 1, 10.0, 20.0, 30.0, 80.0, 0.9)])
        ft1 = FrameTracks(1, [TrackDetection(1, 1, 13.0, 24.0, 33.0, 84.0, 0.9)])
        raw = tracks_to_trajectories([ft0, ft1], fps=10.0)
        smoothed = smooth_trajectories(raw, window=1)
        summaries = compute_trajectory_summaries(smoothed, fps=10.0)
        self.assertEqual(len(summaries), 1)
        self.assertAlmostEqual(summaries[0].path_length_px, 5.0, places=4)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trajectories.csv"
            write_trajectories_csv(csv_path, smoothed)
            reloaded = read_trajectories_csv(csv_path)
            self.assertEqual(len(reloaded[1]), 2)


if __name__ == "__main__":
    unittest.main()
