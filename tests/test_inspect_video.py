"""Check inspection artifacts without relying on external datasets."""

from __future__ import annotations

import tempfile
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from helpers import INSPECTION_SCRIPTS, make_sequence
import inspect_video


class InspectionTests(unittest.TestCase):
    def test_short_sources_sample_each_frame_once(self) -> None:
        self.assertEqual(inspect_video.sample_frame_indices(5, 12), [0, 1, 2, 3, 4])

    def test_samples_are_unique_and_in_range(self) -> None:
        indices = inspect_video.sample_frame_indices(100, 12)
        self.assertEqual(len(indices), 12)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(0 <= index < 100 for index in indices))
        self.assertEqual((indices[0], indices[-1]), (0, 99))

    def test_contact_sheet_contains_every_frame(self) -> None:
        frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (20, 60, 100, 140, 180)]
        sheet = inspect_video.build_contact_sheet(frames)
        self.assertEqual(sheet.shape, (270, 960, 3))
        self.assertEqual(int(sheet[67, 120, 0]), 20)
        self.assertEqual(int(sheet[202, 120, 0]), 180)
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
            samples = sorted((output / "sample_frames" / result["name"]).glob("*.jpg"))
            self.assertEqual(len(samples), 5)
            self.assertTrue(all(cv2.imread(str(path)) is not None for path in samples))
            self.assertIsNotNone(cv2.imread(str(output / result["contact_sheet"])))

    def test_matching_stems_have_separate_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [make_sequence(root / "left"), make_sequence(root / "right")]
            run = inspect_video.run_inspection(sources, root / "run")
            records = json.loads((run / "video_info.json").read_text(encoding="utf-8"))
            self.assertNotEqual(records[0]["name"], records[1]["name"])
            self.assertEqual(len(list((run / "sample_frames").glob("*/*.jpg"))), 10)
            self.assertEqual(json.loads((run / "run.json").read_text())["status"], "complete")

    def test_repeated_run_never_replaces_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            output = root / "run"
            inspect_video.run_inspection([source], output)
            before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            with self.assertRaises(FileExistsError):
                inspect_video.run_inspection([source], output)
            after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_duplicate_and_invalid_sources_do_not_create_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            output = root / "run"
            for sources in ([], [source, source], [source, root / "absent.avi"]):
                with self.subTest(sources=sources), self.assertRaises((ValueError, FileNotFoundError)):
                    inspect_video.run_inspection(sources, output)
                self.assertFalse(output.exists())

    def test_encoding_failure_marks_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            output = root / "run"
            with patch("inspect_video.cv2.imencode", return_value=(False, None)):
                with self.assertRaisesRegex(OSError, "encode"):
                    inspect_video.run_inspection([source], output)
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "failed")
            self.assertFalse((output / "video_info.json").exists())

    def test_failed_later_source_retains_partial_run_without_claiming_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [make_sequence(root, name="first"), make_sequence(root, name="second")]
            (sources[1] / "img1" / "000002.jpg").write_bytes(b"corrupt")
            output = root / "run"
            with self.assertRaises(OSError):
                inspect_video.run_inspection(sources, output)
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "failed")
            self.assertFalse((output / "video_info.json").exists())
            self.assertTrue((output / f"contact_sheet_{inspect_video.source_name(sources[0])}.png").exists())

    def test_default_runs_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            with patch.object(inspect_video, "OUTPUT_DIR", root / "runs"):
                first = inspect_video.run_inspection([source])
                second = inspect_video.run_inspection([source])
            self.assertNotEqual(first, second)
            self.assertTrue((first / "video_info.json").exists())
            self.assertTrue((second / "video_info.json").exists())

    def test_cli_runs_from_another_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            output = root / "run"
            result = subprocess.run(
                [sys.executable, str(INSPECTION_SCRIPTS / "inspect_video.py"), str(source), "--output-dir", str(output)],
                cwd=root, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "video_info.json").exists())

    def test_image_writer_never_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.png"
            path.write_bytes(b"keep me")
            with self.assertRaises(FileExistsError):
                inspect_video.write_image(path, np.zeros((20, 20, 3), dtype=np.uint8))
            self.assertEqual(path.read_bytes(), b"keep me")

    def test_one_sample_uses_the_first_frame(self) -> None:
        self.assertEqual(inspect_video.sample_frame_indices(100, 1), [0])
        self.assertEqual(inspect_video.sample_frame_indices(1, 12), [0])

    def test_invalid_sample_counts_are_rejected(self) -> None:
        for frames, samples in ((0, 12), (-1, 12), (5, 0), (5, -1), (5, 121)):
            with self.subTest(frames=frames, samples=samples), self.assertRaises(ValueError):
                inspect_video.sample_frame_indices(frames, samples)
        with self.assertRaises(TypeError):
            inspect_video.sample_frame_indices(10, True)

    def test_square_thumbnail_is_centered_without_distortion(self) -> None:
        thumb = inspect_video.make_thumbnail(np.full((100, 100, 3), 255, dtype=np.uint8))
        self.assertEqual(thumb.shape, (135, 240, 3))
        occupied = np.where(thumb[67, :, 0] > 0)[0]
        self.assertEqual(len(occupied), 135)
        self.assertEqual(occupied[0], 52)

    def test_empty_contact_sheet_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inspect_video.build_contact_sheet([])

    def test_custom_sampling_records_timestamps_and_holds_only_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            with patch("inspect_video.build_contact_sheet", wraps=inspect_video.build_contact_sheet) as build:
                output = inspect_video.run_inspection([source], root / "run", sample_count=3)
            self.assertTrue(all(frame.shape == (135, 240, 3) for frame in build.call_args.args[0]))
            record = json.loads((output / "video_info.json").read_text())[0]
            self.assertEqual(record["sampled_frame_indices"], [0, 2, 4])
            self.assertEqual(record["sampled_timestamps_seconds"], [0.0, 0.2, 0.4])

    def test_invalid_sample_argument_fails_without_artifacts_or_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_sequence(root)
            output = root / "run"
            result = subprocess.run(
                [sys.executable, str(INSPECTION_SCRIPTS / "inspect_video.py"), str(source), "--output-dir", str(output), "--samples", "0"],
                cwd=root, text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("samples must be", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
