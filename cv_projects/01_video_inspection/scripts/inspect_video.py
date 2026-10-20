"""Phase 1 entry point: inspect video sources and produce metadata + contact sheets.

Usage:
    uv run python cv_projects/01_video_inspection/scripts/inspect_video.py <source> [<source> ...]

Each <source> is a video file or a MOT17-style frame-sequence directory. Writes
to a new run directory under cv_projects/01_video_inspection/outputs/:
    run.json                  completion/failure status
    video_info.json           one metadata entry per source
    sample_frames/<name>/     evenly spaced sampled frames
    contact_sheet_<name>.png  grid of the sampled frames
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from video_source import VideoSource

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
SAMPLE_COUNT = 12
MAX_SAMPLE_COUNT = 120
CONTACT_SHEET_COLS = 4
THUMB_SIZE = (240, 135)  # width, height


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if isinstance(frame_count, bool) or isinstance(sample_count, bool):
        raise TypeError("Frame and sample counts must be integers")
    frame_count, sample_count = (
        operator.index(frame_count),
        operator.index(sample_count),
    )
    if frame_count <= 0 or not 1 <= sample_count <= MAX_SAMPLE_COUNT:
        raise ValueError(
            f"Frame count must be positive and samples must be between 1 and {MAX_SAMPLE_COUNT}"
        )
    count = min(frame_count, sample_count)
    if count == 1:
        return [0]
    return [index * (frame_count - 1) // (count - 1) for index in range(count)]


def make_thumbnail(frame: np.ndarray) -> np.ndarray:
    """Fit a BGR frame into the thumbnail without stretching its aspect ratio."""
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or frame.size == 0
        or frame.dtype != np.uint8
    ):
        raise ValueError("Contact sheet frames must be nonempty uint8 BGR images")
    height, width = frame.shape[:2]
    scale = min(THUMB_SIZE[0] / width, THUMB_SIZE[1] / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, size, interpolation=interpolation)
    thumb = np.zeros((THUMB_SIZE[1], THUMB_SIZE[0], 3), dtype=np.uint8)
    x, y = (THUMB_SIZE[0] - size[0]) // 2, (THUMB_SIZE[1] - size[1]) // 2
    thumb[y : y + size[1], x : x + size[0]] = resized
    return thumb


def build_contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    if not 1 <= len(frames) <= MAX_SAMPLE_COUNT:
        raise ValueError(
            f"Contact sheets require between 1 and {MAX_SAMPLE_COUNT} frames"
        )
    thumbs = [make_thumbnail(frame) for frame in frames]
    cols = CONTACT_SHEET_COLS
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * THUMB_SIZE[1], cols * THUMB_SIZE[0], 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        y, x = r * THUMB_SIZE[1], c * THUMB_SIZE[0]
        sheet[y : y + THUMB_SIZE[1], x : x + THUMB_SIZE[0]] = thumb
    return sheet


def source_name(source_path: str | Path) -> str:
    """Keep readable source names while separating identical filename stems."""
    path = Path(source_path).expanduser().resolve()
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", path.stem).strip("._")[:64] or "source"
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def write_image(path: Path, frame: np.ndarray) -> None:
    """Encode explicitly, check success, and never replace an existing artifact."""
    ok, encoded = cv2.imencode(path.suffix, frame)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    with path.open("xb") as handle:
        handle.write(encoded.tobytes())


def write_json(path: Path, value: object) -> None:
    """Publish valid UTF-8 JSON only after the temporary write completes."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def inspect_source(
    source_path: str | Path,
    output_dir: Path | None = None,
    *,
    sample_count: int = SAMPLE_COUNT,
) -> dict:
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    name = source_name(source_path)

    with VideoSource(source_path) as source:
        meta = source.metadata
        indices = sample_frame_indices(meta.frame_count, sample_count)

        frames_dir = output_dir / "sample_frames" / name
        sheet_path = output_dir / f"contact_sheet_{name}.png"
        if sheet_path.exists():
            raise FileExistsError(f"Refusing to replace contact sheet: {sheet_path}")
        frames_dir.mkdir(parents=True, exist_ok=False)

        frames = []
        for idx in indices:
            frame = source.read_frame(idx)
            frames.append(make_thumbnail(frame))
            write_image(frames_dir / f"frame_{idx:06d}.jpg", frame)

        if frames:
            sheet = build_contact_sheet(frames)
            write_image(sheet_path, sheet)

    return {
        "name": name,
        **asdict(meta),
        "sampled_frame_indices": indices,
        "sampled_timestamps_seconds": [index / meta.fps for index in indices],
        "sample_frames": [
            f"sample_frames/{name}/frame_{idx:06d}.jpg" for idx in indices
        ],
        "contact_sheet": sheet_path.name,
    }


def run_inspection(
    sources: Sequence[str | Path],
    output_dir: Path | None = None,
    *,
    sample_count: int = SAMPLE_COUNT,
) -> Path:
    """Inspect a batch in an exclusive directory and retain failed-run evidence."""
    paths = [Path(source).expanduser().resolve() for source in sources]
    if not paths:
        raise ValueError("At least one source is required")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate input sources are not allowed")
    sample_frame_indices(
        1, sample_count
    )  # Validate before creating any output directories.
    sample_count = operator.index(sample_count)
    for path in paths:
        with VideoSource(path):
            pass
    if output_dir is None:
        run_id = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]
        output_dir = OUTPUT_DIR / run_id
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "sources": [str(path) for path in paths],
        "requested_sample_count": sample_count,
    }
    write_json(output_dir / "run.json", manifest)
    try:
        results = [
            inspect_source(path, output_dir, sample_count=sample_count)
            for path in paths
        ]
        write_json(output_dir / "video_info.json", results)
        manifest.update(status="complete", source_count=len(results))
        write_json(output_dir / "run.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            write_json(output_dir / "run.json", manifest)
        except OSError:
            pass  # Preserve the original failure if the output device is unavailable.
        raise
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect video sources for Phase 1.")
    parser.add_argument(
        "sources",
        nargs="+",
        help="Video file(s) or MOT17-style frame-sequence directory(ies)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="New run directory; must not already exist"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLE_COUNT,
        help="Samples per source, 1..120 (default: 12)",
    )
    args = parser.parse_args(argv)
    try:
        output_dir = run_inspection(
            args.sources, args.output_dir, sample_count=args.samples
        )
    except (OSError, ValueError, cv2.error) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Wrote {output_dir / 'video_info.json'} ({len(args.sources)} source(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
