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


if __name__ == "__main__":
    unittest.main()
