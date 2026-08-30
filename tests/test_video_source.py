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


if __name__ == "__main__":
    unittest.main()
