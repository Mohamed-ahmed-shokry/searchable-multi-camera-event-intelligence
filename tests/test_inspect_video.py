"""Check inspection artifacts without relying on external datasets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from helpers import make_sequence
import inspect_video


class InspectionTests(unittest.TestCase):
    def test_short_sources_sample_each_frame_once(self) -> None:
        self.assertEqual(inspect_video.sample_frame_indices(5, 12), [0, 1, 2, 3, 4])

    def test_samples_are_unique_and_in_range(self) -> None:
        indices = inspect_video.sample_frame_indices(100, 12)
        self.assertEqual(len(indices), 12)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 <= index < 100 for index in indices))

    def test_contact_sheet_contains_every_frame(self) -> None:
        frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (20, 60, 100, 140, 180)]
        sheet = inspect_video.build_contact_sheet(frames)
        self.assertEqual(sheet.shape, (270, 960, 3))
        self.assertEqual(int(sheet[0, 0, 0]), 20)
        self.assertEqual(int(sheet[135, 0, 0]), 180)
        self.assertEqual(int(sheet[135, 240, 0]), 0)

    def test_inspection_writes_readable_samples_and_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            output = root / "results"
            with patch.object(inspect_video, "OUTPUT_DIR", output):
                result = inspect_video.inspect_source(str(source))
            self.assertEqual(result["frame_count"], 5)
            self.assertEqual(result["sampled_frame_indices"], [0, 1, 2, 3, 4])
            samples = sorted((output / "sample_frames" / "sequence").glob("*.jpg"))
            self.assertEqual(len(samples), 5)
            self.assertTrue(all(cv2.imread(str(path)) is not None for path in samples))
            self.assertIsNotNone(cv2.imread(str(output / "contact_sheet_sequence.png")))


if __name__ == "__main__":
    unittest.main()
