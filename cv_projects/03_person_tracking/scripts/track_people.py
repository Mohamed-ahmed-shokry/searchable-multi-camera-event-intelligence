"""Phase 3: Person Tracking data structures, ByteTrack/BoT-SORT wrapper,
CSV exporters, track anomaly analyzer, and CLI entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from video_source import VideoSource
except ImportError:
    _subproject_1_scripts = (
        Path(__file__).resolve().parents[2] / "01_video_inspection" / "scripts"
    )
    if str(_subproject_1_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_1_scripts))
    from video_source import VideoSource

TRACKS_CSV_HEADER: list[str] = [
    "frame_id",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
]


@dataclass(frozen=True)
class TrackDetection:
    """A single tracked person bounding box on a specific video frame."""

    frame_id: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid bounding box coordinates: ({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def to_row(self) -> list[str]:
        return [
            str(self.frame_id),
            str(self.track_id),
            f"{self.x1:.2f}",
            f"{self.y1:.2f}",
            f"{self.x2:.2f}",
            f"{self.y2:.2f}",
            f"{self.confidence:.4f}",
        ]


@dataclass
class FrameTracks:
    """All tracked person bounding boxes for a single video frame."""

    frame_id: int
    tracks: list[TrackDetection] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")

    def to_rows(self) -> list[list[str]]:
        if not self.tracks:
            return [[str(self.frame_id), "", "", "", "", "", ""]]
        return [track.to_row() for track in self.tracks]


def write_tracks_csv(path: Path | str, frame_tracks: Sequence[FrameTracks]) -> Path:
    """Write per-frame tracks to CSV matching TRACKS_CSV_HEADER."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACKS_CSV_HEADER)
        for ft in frame_tracks:
            writer.writerows(ft.to_rows())
    return out_path


def read_tracks_csv(path: Path | str) -> list[FrameTracks]:
    """Read a tracks.csv file back into ordered FrameTracks objects."""
    in_path = Path(path)
    by_frame: dict[int, FrameTracks] = {}
    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != TRACKS_CSV_HEADER:
            raise ValueError(f"Expected CSV header {TRACKS_CSV_HEADER}, got {header!r}")
        for row in reader:
            if not row:
                continue
            frame_id = int(row[0])
            ft = by_frame.setdefault(frame_id, FrameTracks(frame_id=frame_id))
            if row[1] == "":
                continue
            ft.tracks.append(
                TrackDetection(
                    frame_id=frame_id,
                    track_id=int(row[1]),
                    x1=float(row[2]),
                    y1=float(row[3]),
                    x2=float(row[4]),
                    y2=float(row[5]),
                    confidence=float(row[6]),
                )
            )
    return [by_frame[fid] for fid in sorted(by_frame)]


TRACK_SUMMARY_CSV_HEADER: list[str] = [
    "track_id",
    "start_frame",
    "end_frame",
    "duration_frames",
    "duration_seconds",
    "observed_frames",
]


@dataclass(frozen=True)
class TrackSummary:
    """Lifecycle summary for a single tracked person identity."""

    track_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    observed_frames: int

    def to_row(self) -> list[str]:
        return [
            str(self.track_id),
            str(self.start_frame),
            str(self.end_frame),
            str(self.duration_frames),
            f"{self.duration_seconds:.4f}",
            str(self.observed_frames),
        ]


def compute_track_summaries(
    frame_tracks: Sequence[FrameTracks], fps: float
) -> list[TrackSummary]:
    """Aggregate per-frame tracks into per-identity TrackSummary records."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    frames_by_id: dict[int, set[int]] = {}
    for ft in frame_tracks:
        for trk in ft.tracks:
            frames_by_id.setdefault(trk.track_id, set()).add(trk.frame_id)

    summaries: list[TrackSummary] = []
    for track_id in sorted(frames_by_id):
        seen = frames_by_id[track_id]
        start_f = min(seen)
        end_f = max(seen)
        dur_frames = end_f - start_f + 1
        dur_sec = dur_frames / fps
        summaries.append(
            TrackSummary(
                track_id=track_id,
                start_frame=start_f,
                end_frame=end_f,
                duration_frames=dur_frames,
                duration_seconds=dur_sec,
                observed_frames=len(seen),
            )
        )
    return summaries


def write_track_summary_csv(
    path: Path | str, summaries: Sequence[TrackSummary]
) -> Path:
    """Write per-identity track summaries to CSV matching TRACK_SUMMARY_CSV_HEADER."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRACK_SUMMARY_CSV_HEADER)
        for summary in summaries:
            writer.writerow(summary.to_row())
    return out_path


def read_track_summary_csv(path: Path | str) -> list[TrackSummary]:
    """Read a track_summary.csv file into TrackSummary objects."""
    in_path = Path(path)
    summaries: list[TrackSummary] = []
    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != TRACK_SUMMARY_CSV_HEADER:
            raise ValueError(
                f"Expected CSV header {TRACK_SUMMARY_CSV_HEADER}, got {header!r}"
            )
        for row in reader:
            if not row:
                continue
            summaries.append(
                TrackSummary(
                    track_id=int(row[0]),
                    start_frame=int(row[1]),
                    end_frame=int(row[2]),
                    duration_frames=int(row[3]),
                    duration_seconds=float(row[4]),
                    observed_frames=int(row[5]),
                )
            )
    return summaries


def _box_center_and_scale(trk: TrackDetection) -> tuple[float, float, float]:
    cx = 0.5 * (trk.x1 + trk.x2)
    cy = 0.5 * (trk.y1 + trk.y2)
    scale = max(trk.x2 - trk.x1, trk.y2 - trk.y1, 1.0)
    return cx, cy, scale


def analyze_track_anomalies(
    frame_tracks: Sequence[FrameTracks], max_handoff_gap: int = 5
) -> dict[str, list[dict[str, int | float]]]:
    """Identify track fragmentation gaps and potential spatial ID handoffs."""
    history: dict[int, list[TrackDetection]] = {}
    for ft in frame_tracks:
        for trk in ft.tracks:
            history.setdefault(trk.track_id, []).append(trk)

    fragmentations: list[dict[str, int | float]] = []
    first_obs: dict[int, TrackDetection] = {}
    last_obs: dict[int, TrackDetection] = {}

    for track_id in sorted(history):
        obs = sorted(history[track_id], key=lambda item: item.frame_id)
        first_obs[track_id] = obs[0]
        last_obs[track_id] = obs[-1]
        for prev, curr in itertools.pairwise(obs):
            gap = curr.frame_id - prev.frame_id - 1
            if gap >= 1:
                fragmentations.append(
                    {
                        "track_id": track_id,
                        "lost_after_frame": prev.frame_id,
                        "reappeared_at_frame": curr.frame_id,
                        "gap_frames": gap,
                    }
                )

    potential_handoffs: list[dict[str, int | float]] = []
    track_ids = sorted(history)
    for old_id in track_ids:
        end_trk = last_obs[old_id]
        cx1, cy1, scale1 = _box_center_and_scale(end_trk)
        for new_id in track_ids:
            if new_id == old_id:
                continue
            start_trk = first_obs[new_id]
            frame_delta = start_trk.frame_id - end_trk.frame_id
            if not (0 <= frame_delta <= max_handoff_gap):
                continue
            cx2, cy2, scale2 = _box_center_and_scale(start_trk)
            dist = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
            if dist <= max(scale1, scale2):
                potential_handoffs.append(
                    {
                        "old_track_id": old_id,
                        "new_track_id": new_id,
                        "old_end_frame": end_trk.frame_id,
                        "new_start_frame": start_trk.frame_id,
                        "center_distance_px": round(dist, 2),
                    }
                )

    return {
        "fragmentations": fragmentations,
        "potential_handoffs": potential_handoffs,
    }


def select_default_device() -> str:
    """Select cuda:0 if PyTorch with CUDA is available, else cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


class PersonTracker:
    """Runs multi-object person tracking using YOLO11 + ByteTrack or BoT-SORT."""

    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        *,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str | None = None,
        target_classes: tuple[int, ...] = (0,),
        model: object = None,
    ) -> None:
        self.model_name = model_name
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or select_default_device()
        self.target_classes = set(target_classes)

        if model is not None:
            self._model = model
        else:
            from ultralytics import YOLO

            self._model = YOLO(model_name)

    def track(self, frame: object, frame_id: int) -> list[TrackDetection]:
        """Track people in a single frame with state persistence across frames."""
        results = self._model.track(  # type: ignore[attr-defined]
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=list(self.target_classes),
            verbose=False,
        )

        tracks: list[TrackDetection] = []
        if not results:
            return tracks

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return tracks

        for box in boxes:
            id_attr = getattr(box, "id", None)
            if id_attr is None:
                continue
            id_val = id_attr[0] if hasattr(id_attr, "__getitem__") else id_attr
            if id_val is None:
                continue
            track_id = int(id_val.item() if hasattr(id_val, "item") else id_val)

            cls_val = box.cls[0]
            cls_id = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
            if cls_id not in self.target_classes:
                continue

            conf_val = box.conf[0]
            conf = float(conf_val.item() if hasattr(conf_val, "item") else conf_val)
            if conf < self.confidence_threshold:
                continue

            xyxy_val = box.xyxy[0]
            xyxy = xyxy_val.tolist() if hasattr(xyxy_val, "tolist") else xyxy_val
            x1, y1, x2, y2 = [float(v) for v in xyxy]

            tracks.append(
                TrackDetection(
                    frame_id=frame_id,
                    track_id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                )
            )

        return tracks


TRACK_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (255, 100, 50),
    (50, 220, 100),
    (50, 150, 255),
    (220, 80, 220),
    (60, 230, 230),
    (255, 180, 40),
    (180, 255, 80),
    (255, 80, 140),
    (80, 200, 180),
    (160, 110, 255),
]


def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a given track_id."""
    return TRACK_COLOR_PALETTE[track_id % len(TRACK_COLOR_PALETTE)]


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackDetection],
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw per-track colored bounding boxes and ID labels onto a frame copy."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    font_thickness = 1

    for trk in tracks:
        color = track_color(trk.track_id)
        x1 = max(0, min(round(trk.x1), width - 1))
        y1 = max(0, min(round(trk.y1), height - 1))
        x2 = max(0, min(round(trk.x2), width - 1))
        y2 = max(0, min(round(trk.y2), height - 1))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"ID {trk.track_id} ({trk.confidence:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        pill_y1 = max(0, y1 - label_h - 4)
        pill_y2 = y1
        pill_x2 = min(width - 1, x1 + label_w + 4)

        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, max(label_h, pill_y2 - 2)),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return annotated


def create_video_writer(
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
    fourcc: str = "mp4v",
) -> cv2.VideoWriter:
    """Create a cv2.VideoWriter for rendering annotated tracking videos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(path), fourcc_code, fps, (width, height))
    if not writer.isOpened():
        fallback_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fallback_fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load tracking configuration from YAML with fallback parser."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except ImportError:
        cfg: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in stripped:
                    continue
                key, val = stripped.split(":", 1)
                key = key.strip()
                val = val.split("#")[0].strip().strip('"').strip("'")
                if not val:
                    continue
                if val.lower() == "true":
                    cfg[key] = True
                elif val.lower() == "false":
                    cfg[key] = False
                elif val.lower() == "null":
                    cfg[key] = None
                else:
                    try:
                        cfg[key] = float(val) if "." in val else int(val)
                    except ValueError:
                        cfg[key] = val
        return cfg


def run_tracking(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    tracker: PersonTracker | None = None,
    config_path: str | Path | None = None,
    model_name: str = "yolo11m.pt",
    tracker_config: str = "bytetrack.yaml",
    confidence_threshold: float = 0.4,
    device: str | None = None,
    save_video: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Execute person tracking against a video file or MOT17 frame sequence."""
    config = load_config(config_path) if config_path else {}
    final_model = str(config.get("model", model_name))
    final_tracker_cfg = str(config.get("tracker", tracker_config))
    final_conf = float(config.get("confidence_threshold", confidence_threshold))
    final_device = config.get("device", device)

    src_path = Path(source_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source does not exist: {src_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tracker is None:
        tracker = PersonTracker(
            model_name=final_model,
            tracker_config=final_tracker_cfg,
            confidence_threshold=final_conf,
            device=final_device,
        )

    frame_records: list[FrameTracks] = []
    start_time = time.perf_counter()

    with VideoSource(src_path) as source:
        meta = source.metadata
        fps = meta.fps if meta.fps > 0 else 25.0
        total_to_process = (
            min(meta.frame_count, max_frames)
            if max_frames is not None
            else meta.frame_count
        )

        writer: cv2.VideoWriter | None = None
        video_path = out_dir / "annotated_tracking_video.mp4"
        if save_video and total_to_process > 0:
            writer = create_video_writer(
                video_path,
                fps=fps,
                width=meta.width,
                height=meta.height,
            )

        try:
            for frame_id, frame in source.iter_frames(stop=total_to_process):
                tracks = tracker.track(frame, frame_id=frame_id)
                frame_records.append(FrameTracks(frame_id=frame_id, tracks=tracks))
                if writer is not None:
                    writer.write(draw_tracks(frame, tracks))
        finally:
            if writer is not None:
                writer.release()

    duration = max(1e-6, time.perf_counter() - start_time)
    tracks_csv = write_tracks_csv(out_dir / "tracks.csv", frame_records)
    summaries = compute_track_summaries(frame_records, fps=fps)
    summary_csv = write_track_summary_csv(out_dir / "track_summary.csv", summaries)
    anomalies = analyze_track_anomalies(frame_records)
    total_track_boxes = sum(len(ft.tracks) for ft in frame_records)

    result: dict[str, Any] = {
        "source_path": str(src_path),
        "frame_count": len(frame_records),
        "num_unique_tracks": len(summaries),
        "total_track_observations": total_track_boxes,
        "model": tracker.model_name,
        "tracker": tracker.tracker_config,
        "confidence_threshold": tracker.confidence_threshold,
        "device": tracker.device,
        "duration_seconds": round(duration, 3),
        "fps": round(len(frame_records) / duration, 2),
        "tracks_csv": str(tracks_csv.resolve()),
        "track_summary_csv": str(summary_csv.resolve()),
        "annotated_tracking_video": str(video_path.resolve()) if save_video else None,
        "anomalies": anomalies,
    }
    (out_dir / "tracking_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-object person tracking on video files or MOT17 sequences."
    )
    parser.add_argument("source", help="Path to video file or MOT17 sequence directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="cv_projects/03_person_tracking/outputs",
        help="Target output directory",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="yolo11m.pt",
        help="YOLO model weights name or path",
    )
    parser.add_argument(
        "--tracker",
        "-t",
        default="bytetrack.yaml",
        help="Tracker configuration file (bytetrack.yaml or botsort.yaml)",
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
        "--no-video",
        action="store_true",
        help="Skip annotated tracking video rendering",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_tracking(
            args.source,
            output_dir=args.output_dir,
            config_path=args.config,
            model_name=args.model,
            tracker_config=args.tracker,
            confidence_threshold=args.conf,
            device=args.device,
            save_video=not args.no_video,
            max_frames=args.max_frames,
        )
        print(
            f"Tracked {summary['frame_count']} frames in {summary['duration_seconds']}s "
            f"({summary['fps']} FPS) with {summary['tracker']} on {summary['device']}."
        )
        print(
            f"Unique tracks: {summary['num_unique_tracks']} "
            f"({summary['total_track_observations']} total boxes)."
        )
        print(f"Tracks saved to: {summary['tracks_csv']}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
