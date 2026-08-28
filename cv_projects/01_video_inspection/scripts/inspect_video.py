"""Phase 1 entry point: inspect video sources and produce metadata + contact sheets.

Usage:
    uv run python cv_projects/01_video_inspection/scripts/inspect_video.py <source> [<source> ...]

Each <source> is a video file or a MOT17-style frame-sequence directory. Writes,
under cv_projects/01_video_inspection/outputs/:
    video_info.json           one metadata entry per source
    sample_frames/<name>/     evenly spaced sampled frames
    contact_sheet_<name>.png  grid of the sampled frames
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from video_source import VideoSource

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
SAMPLE_COUNT = 12
CONTACT_SHEET_COLS = 4
THUMB_SIZE = (240, 135)  # width, height


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= sample_count:
        return list(range(frame_count))
    step = frame_count / sample_count
    return [int(i * step) for i in range(sample_count)]


def build_contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    thumbs = [cv2.resize(f, THUMB_SIZE) for f in frames]
    cols = CONTACT_SHEET_COLS
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * THUMB_SIZE[1], cols * THUMB_SIZE[0], 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * THUMB_SIZE[1], c * THUMB_SIZE[0]
        sheet[y : y + THUMB_SIZE[1], x : x + THUMB_SIZE[0]] = thumb
    return sheet


def inspect_source(source_path: str) -> dict:
    name = Path(source_path).stem or Path(source_path).name

    with VideoSource(source_path) as source:
        meta = source.metadata
        indices = sample_frame_indices(meta.frame_count, SAMPLE_COUNT)

        frames_dir = OUTPUT_DIR / "sample_frames" / name
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        for idx in indices:
            frame = source.read_frame(idx)
            frames.append(frame)
            cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.jpg"), frame)

        if frames:
            sheet = build_contact_sheet(frames)
            cv2.imwrite(str(OUTPUT_DIR / f"contact_sheet_{name}.png"), sheet)

    return {"name": name, **asdict(meta), "sampled_frame_indices": indices}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect video sources for Phase 1.")
    parser.add_argument(
        "sources", nargs="+", help="Video file(s) or MOT17-style frame-sequence directory(ies)"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [inspect_source(s) for s in args.sources]

    info_path = OUTPUT_DIR / "video_info.json"
    info_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {info_path} ({len(results)} source(s))")


if __name__ == "__main__":
    main()
