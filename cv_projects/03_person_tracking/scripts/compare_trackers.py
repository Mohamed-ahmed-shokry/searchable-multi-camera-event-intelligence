"""Phase 3: Side-by-side ByteTrack vs BoT-SORT comparison and early quantitative
MOT17 checkpoint generator (`checkpoint_metrics.json`).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evaluate_mot import evaluate_tracking, load_mot17_gt
from track_people import PersonTracker, read_tracks_csv, run_tracking

DEFAULT_TRACKERS: tuple[str, ...] = ("bytetrack.yaml", "botsort.yaml")


def _discover_gt_path(source_path: Path, explicit_gt: Path | str | None) -> Path | None:
    if explicit_gt is not None:
        gt_candidate = Path(explicit_gt).expanduser().resolve()
        if not gt_candidate.is_file():
            raise FileNotFoundError(f"Ground-truth file not found: {gt_candidate}")
        return gt_candidate
    auto_gt = source_path / "gt" / "gt.txt"
    return auto_gt if auto_gt.is_file() else None


def compare_trackers(
    source_path: str | Path,
    *,
    output_dir: str
    | Path = "cv_projects/03_person_tracking/outputs/tracker_comparison",
    trackers: Sequence[str] = DEFAULT_TRACKERS,
    model_name: str = "yolo11m.pt",
    confidence_threshold: float = 0.4,
    device: str | None = None,
    gt_path: str | Path | None = None,
    max_frames: int | None = None,
    save_videos: bool = False,
    tracker_factory: Callable[[str], PersonTracker] | None = None,
) -> dict[str, Any]:
    """Run multiple trackers on the same source and write checkpoint_metrics.json."""
    src_path = Path(source_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source does not exist: {src_path}")

    resolved_gt = _discover_gt_path(src_path, gt_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker_results: dict[str, dict[str, Any]] = {}
    processed_frames = 0

    for tracker_cfg in trackers:
        tracker_key = Path(tracker_cfg).stem
        sub_out = out_dir / tracker_key
        tracker_instance = (
            tracker_factory(tracker_cfg) if tracker_factory is not None else None
        )
        run_info = run_tracking(
            src_path,
            output_dir=sub_out,
            tracker=tracker_instance,
            model_name=model_name,
            tracker_config=tracker_cfg,
            confidence_threshold=confidence_threshold,
            device=device,
            save_video=save_videos,
            max_frames=max_frames,
        )
        processed_frames = int(run_info["frame_count"])
        anomalies = run_info["anomalies"]

        entry: dict[str, Any] = {
            "tracker_config": tracker_cfg,
            "fps": run_info["fps"],
            "duration_seconds": run_info["duration_seconds"],
            "num_unique_tracks": run_info["num_unique_tracks"],
            "total_track_observations": run_info["total_track_observations"],
            "fragmentation_count": len(anomalies["fragmentations"]),
            "potential_handoff_count": len(anomalies["potential_handoffs"]),
            "tracks_csv": run_info["tracks_csv"],
            "track_summary_csv": run_info["track_summary_csv"],
            "anomalies": anomalies,
        }

        if resolved_gt is not None:
            gt_by_frame = load_mot17_gt(resolved_gt, max_frames=processed_frames)
            pred_frames = read_tracks_csv(run_info["tracks_csv"])
            mot_metrics = evaluate_tracking(gt_by_frame, pred_frames)
            entry.update(mot_metrics)

        tracker_results[tracker_key] = entry

    checkpoint: dict[str, Any] = {
        "source_path": str(src_path),
        "gt_path": str(resolved_gt) if resolved_gt is not None else None,
        "model": model_name,
        "confidence_threshold": confidence_threshold,
        "frame_count": processed_frames,
        "trackers": tracker_results,
    }
    metrics_path = out_dir / "checkpoint_metrics.json"
    metrics_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ByteTrack and BoT-SORT and generate checkpoint_metrics.json."
    )
    parser.add_argument("source", help="Path to video file or MOT17 sequence directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="cv_projects/03_person_tracking/outputs/tracker_comparison",
        help="Output directory for comparison runs and checkpoint_metrics.json",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="yolo11m.pt",
        help="YOLO model weights name or path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        help="Inference device (e.g. 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--gt",
        default=None,
        help="Optional path to MOT17 gt.txt (auto-detected if source is MOT17 dir)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Render annotated_tracking_video.mp4 for each tracker",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        checkpoint = compare_trackers(
            args.source,
            output_dir=args.output_dir,
            model_name=args.model,
            confidence_threshold=args.conf,
            device=args.device,
            gt_path=args.gt,
            max_frames=args.max_frames,
            save_videos=args.save_videos,
        )
        for name, metrics in checkpoint["trackers"].items():
            line = (
                f"[{name}] FPS={metrics['fps']} "
                f"tracks={metrics['num_unique_tracks']} "
                f"frag={metrics['fragmentation_count']}"
            )
            if "mota" in metrics:
                line += (
                    f" MOTA={metrics['mota']:.4f} "
                    f"IDF1={metrics['idf1']:.4f} "
                    f"IDSW={metrics['id_switches']}"
                )
            print(line)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
