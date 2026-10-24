"""Benchmark and compare YOLO11 model variants (yolo11n, yolo11s, yolo11m).

Evaluates inference throughput (FPS) and person detection statistics on
real footage using CUDA acceleration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from detect_people import PersonDetector
from video_source import VideoSource

DEFAULT_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")


def benchmark_model(
    model_name: str,
    source_path: str | Path,
    *,
    confidence_threshold: float = 0.4,
    device: str | None = None,
    max_frames: int | None = 200,
) -> dict[str, Any]:
    """Run detection with a given model over a sequence and record performance."""
    detector = PersonDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    total_detections = 0
    frame_count = 0
    frames_with_detections = 0

    with VideoSource(source_path) as source:
        limit = min(len(source), max_frames) if max_frames is not None else len(source)
        # Warmup with first frame
        if limit > 0:
            warmup_frame = source.read_frame(0)
            detector.detect(warmup_frame)

        start_time = time.perf_counter()
        for _, frame in source.iter_frames(stop=limit):
            dets = detector.detect(frame)
            frame_count += 1
            if dets:
                frames_with_detections += 1
                total_detections += len(dets)

        duration = max(1e-6, time.perf_counter() - start_time)

    fps = round(frame_count / duration, 2)
    return {
        "model": model_name,
        "device": detector.device,
        "confidence_threshold": confidence_threshold,
        "frame_count": frame_count,
        "duration_seconds": round(duration, 3),
        "fps": fps,
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "avg_detections_per_frame": round(total_detections / max(1, frame_count), 2),
    }


def compare_models(
    source_path: str | Path,
    models: tuple[str, ...] = DEFAULT_MODELS,
    *,
    confidence_threshold: float = 0.4,
    device: str | None = None,
    max_frames: int | None = 200,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Compare multiple YOLO11 model variants on the same source footage."""
    results: list[dict[str, Any]] = []
    print(f"Benchmarking models on {source_path} (limit: {max_frames} frames)...")
    for model_name in models:
        print(f"  Testing {model_name}...")
        metrics = benchmark_model(
            model_name,
            source_path,
            confidence_threshold=confidence_threshold,
            device=device,
            max_frames=max_frames,
        )
        results.append(metrics)
        print(
            f"    -> {metrics['fps']} FPS | {metrics['total_detections']} detections "
            f"across {metrics['frames_with_detections']} frames"
        )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print markdown-style comparison table to console."""
    headers = ["Model", "Device", "FPS", "Duration(s)", "Total Dets", "Avg Dets/Frame"]
    print("\n| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results:
        row = [
            r["model"],
            r["device"],
            str(r["fps"]),
            str(r["duration_seconds"]),
            str(r["total_detections"]),
            str(r["avg_detections_per_frame"]),
        ]
        print("| " + " | ".join(row) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare YOLO11 model variants on test footage."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="data/fallback_clips/people-detection.mp4",
        help="Path to source video or frame sequence directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="List of model weights to compare",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (default: auto cuda/cpu)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Max frames to test per model (default: 200)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="cv_projects/02_person_detection/outputs/model_comparison.json",
        help="Path to save benchmark JSON results",
    )
    args = parser.parse_args(argv)

    try:
        results = compare_models(
            args.source,
            models=tuple(args.models),
            confidence_threshold=args.conf,
            device=args.device,
            max_frames=args.max_frames,
            output_path=args.output,
        )
        print_comparison_table(results)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
