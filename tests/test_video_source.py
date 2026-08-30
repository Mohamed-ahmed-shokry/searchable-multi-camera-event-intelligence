"""Regression tests against real, small OpenCV media files."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

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

    @staticmethod
    def change_metadata(source: Path, key: str, value: str) -> None:
        path = source / "seqinfo.ini"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "\n".join(f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines),
            encoding="utf-8",
        )

    def test_invalid_sequence_metadata_is_rejected(self) -> None:
        cases = [
            ("frameRate", "nan"), ("frameRate", "inf"), ("frameRate", "0"),
            ("frameRate", "-2"), ("frameRate", "1e-320"), ("seqLength", "0"),
            ("seqLength", "-1"), ("seqLength", "4"), ("seqLength", "6"),
            ("imWidth", "0"), ("imHeight", "-1"), ("imDir", ".."),
            ("imExt", ".*"), ("frameRate", "unknown"),
        ]
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key, value=value):
                source = make_sequence(self.root, name=f"invalid_{index}")
                self.change_metadata(source, key, value)
                with self.assertRaises(ValueError):
                    VideoSource(source)

    def test_missing_frame_does_not_shift_later_frame_ids(self) -> None:
        source = make_sequence(self.root)
        (source / "img1" / "000002.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "declares 5 frames but found 4"):
            VideoSource(source)

    def test_gap_in_numbering_is_rejected_even_when_count_matches(self) -> None:
        source = make_sequence(self.root)
        (source / "img1" / "000002.jpg").rename(source / "img1" / "000006.jpg")
        with self.assertRaisesRegex(ValueError, "contiguous"):
            VideoSource(source)

    def test_non_numeric_frame_names_are_rejected(self) -> None:
        source = make_sequence(self.root)
        (source / "img1" / "000001.jpg").rename(source / "img1" / "cover.jpg")
        with self.assertRaisesRegex(ValueError, "numbered"):
            VideoSource(source)

    def test_unpadded_frame_names_are_sorted_numerically(self) -> None:
        source = make_sequence(self.root, count=12)
        for path in (source / "img1").glob("*.jpg"):
            path.rename(path.with_name(f"{int(path.stem)}.jpg"))
        with VideoSource(source) as reader:
            self.assertAlmostEqual(float(reader.read_frame(9).mean()), 104, delta=2)

    def test_frame_dimensions_must_match_metadata(self) -> None:
        source = make_sequence(self.root)
        self.change_metadata(source, "imWidth", "65")
        with VideoSource(source) as reader, self.assertRaisesRegex(ValueError, "dimensions"):
            reader.read_frame(0)

    def test_corrupt_image_fails_when_read(self) -> None:
        source = make_sequence(self.root)
        (source / "img1" / "000003.jpg").write_bytes(b"not an image")
        with VideoSource(source) as reader, self.assertRaisesRegex(OSError, "Failed to read"):
            reader.read_frame(2)

    def test_required_sequence_fields_are_not_inferred(self) -> None:
        source = make_sequence(self.root)
        (source / "seqinfo.ini").write_text("[Sequence]\nimDir=img1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid seqinfo.ini"):
            VideoSource(source)

    def test_directory_without_seqinfo_is_rejected_before_capture(self) -> None:
        with patch("video_source.cv2.VideoCapture") as capture:
            with self.assertRaisesRegex(ValueError, "seqinfo.ini"):
                VideoSource(self.root)
            capture.assert_not_called()

    def test_closed_sources_reject_reads_and_close_is_idempotent(self) -> None:
        for path in (make_sequence(self.root), make_video(self.root)):
            with self.subTest(path=path):
                source = VideoSource(path)
                source.close()
                source.close()
                with self.assertRaisesRegex(ValueError, "closed"):
                    source.read_frame(0)
                with self.assertRaisesRegex(ValueError, "closed"):
                    source.__enter__()

    def test_indices_require_integers(self) -> None:
        with VideoSource(make_sequence(self.root)) as source:
            for index in (True, 0.5, "1"):
                with self.subTest(index=index), self.assertRaises(TypeError):
                    source.read_frame(index)
            self.assertEqual(source.read_frame(np.int64(0)).shape, (48, 64, 3))

    def fake_capture(self, overrides: dict | None = None) -> tuple[Path, Mock]:
        path = self.root / "mock.avi"
        path.touch()
        properties = {
            cv2.CAP_PROP_FPS: 10.0, cv2.CAP_PROP_FRAME_COUNT: 5.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64.0, cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FOURCC: 0.0,
        }
        properties.update(overrides or {})
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = properties.__getitem__
        capture.set.return_value = True
        capture.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
        return path, capture

    def test_capture_is_released_when_opening_fails(self) -> None:
        path, capture = self.fake_capture()
        capture.isOpened.return_value = False
        with patch("video_source.cv2.VideoCapture", return_value=capture):
            with self.assertRaises(OSError):
                VideoSource(path)
        capture.release.assert_called_once()

    def test_invalid_video_metadata_releases_capture(self) -> None:
        for property_id, value in (
            (cv2.CAP_PROP_FPS, 0), (cv2.CAP_PROP_FPS, float("nan")),
            (cv2.CAP_PROP_FRAME_COUNT, float("inf")), (cv2.CAP_PROP_FRAME_COUNT, 2.5),
            (cv2.CAP_PROP_FRAME_WIDTH, -1), (cv2.CAP_PROP_FRAME_HEIGHT, 0),
        ):
            with self.subTest(property_id=property_id, value=value):
                path, capture = self.fake_capture({property_id: value})
                with patch("video_source.cv2.VideoCapture", return_value=capture):
                    with self.assertRaises(ValueError):
                        VideoSource(path)
                capture.release.assert_called_once()

    def test_failed_seek_does_not_read_or_mislabel_a_frame(self) -> None:
        path, capture = self.fake_capture()
        capture.set.return_value = False
        with patch("video_source.cv2.VideoCapture", return_value=capture):
            with VideoSource(path) as source, self.assertRaisesRegex(OSError, "seek"):
                source.read_frame(2)
        capture.read.assert_not_called()
        capture.release.assert_called_once()

    def test_invalid_decoded_video_frames_are_rejected(self) -> None:
        for frame in (None, np.zeros((0, 0, 3)), np.zeros((20, 20, 3))):
            with self.subTest(shape=None if frame is None else frame.shape):
                path, capture = self.fake_capture()
                capture.read.return_value = (True, frame)
                with patch("video_source.cv2.VideoCapture", return_value=capture):
                    with VideoSource(path) as source, self.assertRaises((OSError, ValueError)):
                        source.read_frame(0)

    def test_unknown_codec_is_null(self) -> None:
        path, capture = self.fake_capture()
        with patch("video_source.cv2.VideoCapture", return_value=capture):
            with VideoSource(path) as source:
                self.assertIsNone(source.metadata.codec)


if __name__ == "__main__":
    unittest.main()
