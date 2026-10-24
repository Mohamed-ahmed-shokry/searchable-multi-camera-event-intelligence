"""Validate an inspection run's ``video_info.json`` against known dataset specs.

Usage:
    uv run python cv_projects/01_video_inspection/scripts/validate_dataset.py <video_info.json>

A run passes when
    - every MOT17 entry (frame_sequence source) matches the published spec table
      (fps, resolution, frame count — sourced from motchallenge.net's archived
      sequence table), and
    - every video-file entry satisfies basic sanity (positive fps, dimensions and
      frame count, and a duration consistent with frame_count / fps).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

# Published training-set specs per base MOT17 sequence (motchallenge.net archive).
# Each FRCNN/SDP/DPM variant of a base sequence shares these values.
MOT17_SPECS: dict[str, dict[str, float]] = {
    "02": {"fps": 30, "width": 1920, "height": 1080, "frame_count": 600},
    "04": {"fps": 30, "width": 1920, "height": 1080, "frame_count": 1050},
    "05": {"fps": 30, "width": 640, "height": 480, "frame_count": 837},
    "09": {"fps": 30, "width": 1920, "height": 1080, "frame_count": 537},
    "10": {"fps": 30, "width": 1920, "height": 1080, "frame_count": 654},
    "11": {"fps": 30, "width": 1920, "height": 1080, "frame_count": 900},
    "13": {"fps": 30, "width": 640, "height": 480, "frame_count": 750},
}

MISMATCH_TOLERANCE_MULTIPLIER = 1.5  # video-file duration vs frame_count/fps slack


def mot17_base_sequence(name: str) -> str | None:
    """Extract the base sequence digits from a name like ``MOT17-04-FRCNN``."""
    parts = name.replace("\\", "/").rstrip("/").split("/")
    label = parts[-1].upper()
    if "MOT17-" not in label:
        return None
    base = label.split("MOT17-")[1]
    return base[:2] if base[:2].isdigit() else None


def check_mot17_entry(entry: dict) -> list[str]:
    base = mot17_base_sequence(str(entry["name"]))
    spec = MOT17_SPECS.get(base or "")
    if spec is None:
        return [
            f"entry {entry['name']}: no spec known for MOT17 base sequence {base!r}"
        ]
    problems = []
    if entry["fps"] != spec["fps"]:
        problems.append(f"fps {entry['fps']} != published {spec['fps']}")
    if (entry["width"], entry["height"]) != (spec["width"], spec["height"]):
        problems.append(
            f"resolution {(entry['width'], entry['height'])} != published "
            f"{(spec['width'], spec['height'])}"
        )
    if entry["frame_count"] != spec["frame_count"]:
        problems.append(
            f"frame_count {entry['frame_count']} != published {spec['frame_count']}"
        )
    return [f"entry {entry['name']}: {problem}" for problem in problems]


def check_clip_entry(entry: dict) -> list[str]:
    problems = []
    for field in ("fps", "width", "height", "frame_count"):
        value = entry[field]
        if not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"{field} must be positive, got {value!r}")
    if problems:
        return [f"entry {entry['name']}: {problem}" for problem in problems]
    expected_duration = entry["frame_count"] / entry["fps"]
    if abs(entry["duration_seconds"] - expected_duration) > (
        MISMATCH_TOLERANCE_MULTIPLIER / entry["fps"]
    ):
        problems.append(
            f"duration {entry['duration_seconds']:.2f}s is inconsistent "
            f"with frame_count/{entry['fps']} = {expected_duration:.2f}s"
        )
    return [f"entry {entry['name']}: {problem}" for problem in problems]


def validate_video_info(video_info: list[dict]) -> list[str]:
    problems = []
    for entry in video_info:
        if entry.get("source_type") == "frame_sequence":
            problems.extend(check_mot17_entry(entry))
        else:
            problems.extend(check_clip_entry(entry))
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a video_info.json against known dataset specs."
    )
    parser.add_argument(
        "video_info", type=Path, help="Path to a produced video_info.json"
    )
    args = parser.parse_args(argv)
    try:
        video_info = json.loads(args.video_info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: could not read {args.video_info}: {exc}\n")
    if not isinstance(video_info, list):
        parser.exit(
            2, f"error: {args.video_info} must contain a JSON array of entries\n"
        )
    problems = validate_video_info(video_info)
    for problem in problems:
        print(f"MISMATCH: {problem}")
    if problems:
        parser.exit(1, f"validation failed ({len(problems)} problem(s))\n")
    print(f"validation passed for {len(video_info)} source(s)")
    return 0


if __name__ == "__main__":
    raising_exit: int = main()
    raise SystemExit(raising_exit)
