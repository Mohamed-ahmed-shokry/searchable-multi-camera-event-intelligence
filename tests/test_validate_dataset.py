"""Dataset spec validator: pure-logic tests plus a CLI integration check."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import validate_dataset


def mot17_entry(name: str, **overrides) -> dict:
    digits = name.replace("MOT17-", "")[:2]
    base = validate_dataset.MOT17_SPECS[
        digits if digits in validate_dataset.MOT17_SPECS else "04"
    ]
    entry = {
        "name": f"mot17/{name}",
        "source_type": "frame_sequence",
        "fps": base["fps"],
        "width": base["width"],
        "height": base["height"],
        "frame_count": base["frame_count"],
        "duration_seconds": base["frame_count"] / base["fps"],
        "codec": None,
    }
    entry.update(overrides)
    return entry


def clip_entry(**overrides) -> dict:
    entry = {
        "name": "fallback_clips/people-detection.mp4",
        "source_type": "video_file",
        "fps": 12,
        "width": 768,
        "height": 432,
        "frame_count": 596,
        "duration_seconds": 596 / 12,
        "codec": "avc1",
    }
    entry.update(overrides)
    return entry


class ValidatorTests(unittest.TestCase):
    def test_valid_mot17_entry_passes(self) -> None:
        self.assertEqual(
            validate_dataset.check_mot17_entry(mot17_entry("MOT17-04-FRCNN")), []
        )

    def test_mot17_variant_names_resolve_to_the_same_base(self) -> None:
        for variant in ("MOT17-04-FRCNN", "MOT17-04-SDP", "MOT17-04-DPM"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    validate_dataset.check_mot17_entry(mot17_entry(variant)), []
                )

    def test_spec_mismatch_reports_each_field(self) -> None:
        problems = validate_dataset.check_mot17_entry(
            mot17_entry("MOT17-04-FRCNN", fps=25, frame_count=999)
        )
        self.assertIn("fps", problems[0])
        self.assertIn("frame_count", problems[1])

    def test_unknown_mot17_base_sequence_is_reported(self) -> None:
        problems = validate_dataset.check_mot17_entry(mot17_entry("MOT17-99-FRCNN"))
        self.assertIn("no spec known", problems[0])

    def test_non_mot17_sequence_name_is_reported(self) -> None:
        entry = mot17_entry("MOT17-04-FRCNN")
        entry["name"] = "some_other_sequence"
        problems = validate_dataset.check_mot17_entry(entry)
        self.assertIn("no spec known", problems[0])

    def test_valid_clip_entry_passes(self) -> None:
        self.assertEqual(validate_dataset.check_clip_entry(clip_entry()), [])

    def test_clip_negative_fields_are_rejected(self) -> None:
        for field in ("fps", "width", "height", "frame_count"):
            with self.subTest(field=field):
                problems = validate_dataset.check_clip_entry(clip_entry(**{field: -1}))
                self.assertIn(field, problems[0])

    def test_clip_inconsistent_duration_is_rejected(self) -> None:
        problems = validate_dataset.check_clip_entry(clip_entry(duration_seconds=999.0))
        self.assertIn("duration", problems[0])

    def test_mixed_manifest_routes_each_source_type(self) -> None:
        problems = validate_dataset.validate_video_info(
            [mot17_entry("MOT17-04-FRCNN"), clip_entry()]
        )
        self.assertEqual(problems, [])

    def test_manifest_problem_carry_entry_names(self) -> None:
        problems = validate_dataset.validate_video_info(
            [mot17_entry("MOT17-04-FRCNN", width=128), clip_entry(duration_seconds=1.0)]
        )
        self.assertTrue(problems[0].startswith("entry mot17/MOT17-04-FRCNN"))
        self.assertTrue(problems[1].startswith("entry fallback_clips/people-detection"))


class ValidatorCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cli(self, video_info: list[dict]) -> subprocess.CompletedProcess:
        path = self.root / "video_info.json"
        path.write_text(json.dumps(video_info), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(Path(validate_dataset.__file__)), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_cli_passes_and_prints_confirmation(self) -> None:
        result = self.run_cli([mot17_entry("MOT17-04-FRCNN"), clip_entry()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validation passed for 2 source(s)", result.stdout)

    def test_cli_fails_with_mismatches(self) -> None:
        result = self.run_cli([mot17_entry("MOT17-04-FRCNN", fps=99)])
        self.assertEqual(result.returncode, 1)
        self.assertIn("MISMATCH", result.stdout)

    def test_cli_rejects_non_array_documents(self) -> None:
        path = self.root / "bad.json"
        path.write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(Path(validate_dataset.__file__)), str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)

    def test_cli_reports_missing_file(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(validate_dataset.__file__)),
                str(self.root / "nope.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
