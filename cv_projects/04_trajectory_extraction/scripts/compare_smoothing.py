"""Phase 4: Compare anchor strategies (bottom_center vs centroid) and
rolling-mean smoothing windows (e.g. w=1, 3, 5) on real or synthetic tracks.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from extract_trajectories import (
        VALID_ANCHORS,
        analyze_trajectory_quality,
        compute_trajectory_summaries,
        read_tracks_csv,
        smooth_trajectories,
        tracks_to_trajectories,
    )
except ImportError:
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in sys.path:
        sys.path.insert(0, str(_this_dir))
    from extract_trajectories import (
        VALID_ANCHORS,
        analyze_trajectory_quality,
        compute_trajectory_summaries,
        read_tracks_csv,
        smooth_trajectories,
        tracks_to_trajectories,
    )


def compare_smoothing(
    tracks_csv: Path | str,
    output_dir: Path | str,
    *,
    fps: float = 30.0,
    windows: Sequence[int] = (1, 3, 5),
    jump_threshold_px: float = 150.0,
) -> dict[str, Any]:
    """Compare raw vs rolling-mean smoothed trajectories across anchor strategies."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not windows:
        raise ValueError("windows must not be empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_tracks = read_tracks_csv(Path(tracks_csv))

    anchor_results: dict[str, dict[str, Any]] = {}
    for anchor in VALID_ANCHORS:
        raw_trajs = tracks_to_trajectories(frame_tracks, fps=fps, anchor=anchor)
        raw_quality = analyze_trajectory_quality(
            raw_trajs, jump_threshold_px=jump_threshold_px
        )
        base_jitter = float(raw_quality["mean_jitter_px"])  # type: ignore[arg-type]

        window_reports: dict[str, Any] = {}
        for w in windows:
            smoothed = smooth_trajectories(raw_trajs, window=w)
            summaries = compute_trajectory_summaries(smoothed, fps=fps)
            quality = analyze_trajectory_quality(
                smoothed, jump_threshold_px=jump_threshold_px
            )
            jitter = float(quality["mean_jitter_px"])  # type: ignore[arg-type]
            reduction_pct = (
                round(100.0 * (base_jitter - jitter) / base_jitter, 2)
                if base_jitter > 0.0
                else 0.0
            )
            window_reports[f"window_{w}"] = {
                "window": w,
                "total_path_length_px": round(
                    sum(item.path_length_px for item in summaries), 2
                ),
                "mean_step_px": quality["mean_step_px"],
                "max_step_px": quality["max_step_px"],
                "mean_jitter_px": jitter,
                "jitter_reduction_pct": reduction_pct,
                "teleport_jump_count": quality["teleport_jump_count"],
            }
        anchor_results[anchor] = window_reports

    report: dict[str, Any] = {
        "tracks_csv": str(tracks_csv),
        "fps": fps,
        "windows": list(windows),
        "anchors": anchor_results,
    }
    out_path = out_dir / "smoothing_comparison.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4: Compare anchor strategies and rolling-mean smoothing windows."
        )
    )
    parser.add_argument(
        "tracks_csv",
        help="Path to Phase 3 tracks.csv file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="cv_projects/04_trajectory_extraction/outputs/smoothing_comparison",
        help="Output directory for smoothing_comparison.json.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate (fps) used to compute timestamps and durations.",
    )
    parser.add_argument(
        "--windows",
        default="1,3,5",
        help="Comma-separated rolling-mean window sizes to evaluate.",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=150.0,
        help="Pixel step threshold for flagging teleport jumps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    windows = [int(part.strip()) for part in args.windows.split(",") if part.strip()]
    report = compare_smoothing(
        args.tracks_csv,
        args.output_dir,
        fps=args.fps,
        windows=windows,
        jump_threshold_px=args.jump_threshold,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
