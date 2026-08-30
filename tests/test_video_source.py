"""Regression tests against real, small OpenCV media files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import make_sequence, make_video
from video_source import VideoSource


class VideoSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_sequence_metadata_matches_fixture(self) -> None:
        with VideoSource(make_sequence(self.root)) as source:
            self.assertEqual(source.metadata.source_type, "frame_sequence")
            self.assertEqual((source.metadata.width, source.metadata.height), (64, 48))
            self.assertEqual(source.metadata.fps, 10)
            self.assertEqual(len(source), 5)
            self.assertAlmostEqual(source.metadata.duration_seconds, 0.5)
            self.assertIsNone(source.metadata.codec)

    def test_sequence_supports_random_access(self) -> None:
        with VideoSource(make_sequence(self.root)) as source:
            self.assertAlmostEqual(float(source.read_frame(4).mean()), 160, delta=2)
            self.assertAlmostEqual(float(source.read_frame(0).mean()), 0, delta=2)

    def test_frame_indices_are_zero_based_and_bounded(self) -> None:
        with VideoSource(make_sequence(self.root)) as source:
            for index in (-1, len(source)):
                with self.subTest(index=index), self.assertRaises(IndexError):
                    source.read_frame(index)

    def test_video_metadata_matches_fixture(self) -> None:
        with VideoSource(make_video(self.root)) as source:
            self.assertEqual(source.metadata.source_type, "video_file")
            self.assertEqual((source.metadata.width, source.metadata.height), (64, 48))
            self.assertEqual(source.metadata.fps, 10)
            self.assertEqual(len(source), 5)
            self.assertAlmostEqual(source.metadata.duration_seconds, 0.5)

    def test_video_supports_backward_and_forward_reads(self) -> None:
        with VideoSource(make_video(self.root)) as source:
            for index in (0, 1, 4, 0):
                with self.subTest(index=index):
                    frame = source.read_frame(index)
                    self.assertEqual(frame.shape, (48, 64, 3))
                    self.assertAlmostEqual(float(frame.mean()), index * 40, delta=3)

    def test_missing_video_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            VideoSource(self.root / "missing.avi")


if __name__ == "__main__":
    unittest.main()
