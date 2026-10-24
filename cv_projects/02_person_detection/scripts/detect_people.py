"""Person detection runner and utilities for Phase 2.

Provides schema-compliant detection structures, YOLO11 inference filtering,
bounding box annotation, and CSV serialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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

DETECTIONS_CSV_HEADER = [
    "frame_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "class",
]


@dataclass(slots=True, frozen=True)
class Detection:
    """Single bounding-box detection within a video frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "person"

    def to_dict(self, frame_id: int) -> dict[str, Any]:
        """Serialize detection to a CSV-row compatible dictionary."""
        return {
            "frame_id": frame_id,
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "confidence": round(float(self.confidence), 4),
            "class": self.class_name,
        }


@dataclass(slots=True)
class FrameDetections:
    """Collection of person detections for a single frame index."""

    frame_id: int
    detections: list[Detection] = field(default_factory=list)

    def to_rows(self) -> list[dict[str, Any]]:
        """Return CSV row dictionaries.

        If no detections exist for this frame, returns a single row with empty
        coordinates so the frame is explicitly accounted for in the output.
        """
        if not self.detections:
            return [
                {
                    "frame_id": self.frame_id,
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "confidence": "",
                    "class": "",
                }
            ]
        return [d.to_dict(self.frame_id) for d in self.detections]


def write_detections_csv(
    frame_records: list[FrameDetections],
    output_path: str | Path,
) -> int:
    """Write frame detection records to a CSV file.

    Parameters
    ----------
    frame_records : list[FrameDetections]
        Records for each frame, typically ordered by frame_id.
    output_path : str | Path
        Target CSV file path.

    Returns
    -------
    int
        Total number of non-empty person detections written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTIONS_CSV_HEADER)
        writer.writeheader()
        for record in frame_records:
            total_detections += len(record.detections)
            for row in record.to_rows():
                writer.writerow(row)

    return total_detections


def read_detections_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a detections CSV file.

    Returns
    -------
    list[dict[str, Any]]
        List of row dictionaries with parsed values.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detections CSV not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != DETECTIONS_CSV_HEADER:
            raise ValueError(
                f"Invalid CSV header: expected {DETECTIONS_CSV_HEADER}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            parsed: dict[str, Any] = {"frame_id": int(row["frame_id"])}
            has_coords = row["x1"] != "" and row["x1"] is not None
            if has_coords:
                parsed["x1"] = float(row["x1"])
                parsed["y1"] = float(row["y1"])
                parsed["x2"] = float(row["x2"])
                parsed["y2"] = float(row["y2"])
                parsed["confidence"] = float(row["confidence"])
                parsed["class"] = row["class"]
            else:
                parsed["x1"] = None
                parsed["y1"] = None
                parsed["x2"] = None
                parsed["y2"] = None
                parsed["confidence"] = None
                parsed["class"] = None
            rows.append(parsed)

    return rows


def select_default_device() -> str:
    """Select cuda:0 if PyTorch with CUDA is available, else cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


class PersonDetector:
    """Runs person-filtered object detection using YOLO11."""

    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        *,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str | None = None,
        target_classes: tuple[int, ...] = (0,),
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or select_default_device()
        self.target_classes = set(target_classes)

        if model is not None:
            self._model = model
            self.class_names = getattr(model, "names", {0: "person"})
        else:
            from ultralytics import YOLO

            self._model = YOLO(model_name)
            self.class_names = getattr(self._model, "names", {0: "person"})

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in a single frame.

        Filters out any bounding boxes that do not belong to target_classes
        or fail the confidence threshold.
        """
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=list(self.target_classes),
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
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
            class_name = self.class_names.get(cls_id, "person")

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    class_name=class_name,
                )
            )

        return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    box_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and confidence labels on a frame.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR image frame.
    detections : list[Detection]
        List of detections for this frame.
    box_color : tuple[int, int, int]
        BGR color for bounding boxes and label pill.
    thickness : int
        Border line thickness.

    Returns
    -------
    np.ndarray
        Annotated BGR image frame copy.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for det in detections:
        x1 = max(0, min(round(det.x1), width - 1))
        y1 = max(0, min(round(det.y1), height - 1))
        x2 = max(0, min(round(det.x2), width - 1))
        y2 = max(0, min(round(det.y2), height - 1))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1

        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        pill_y1 = max(0, y1 - label_h - 4)
        pill_y2 = y1
        pill_x2 = min(width - 1, x1 + label_w + 4)

        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), box_color, -1)
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
    """Create a cv2.VideoWriter for rendering annotated videos."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(path), fourcc_code, fps, (width, height))
    if not writer.isOpened():
        # Fallback to MJPG if mp4v is not supported by backend
        fallback_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fallback_fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


def load_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load configuration from a YAML file with graceful fallback."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg: dict[str, Any] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" in stripped:
                    key, val = stripped.split(":", 1)
                    key = key.strip()
                    val = val.split("#")[0].strip().strip('"').strip("'")
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


def run_detection(
    source_path: str | Path,
    *,
    output_dir: str | Path,
    detector: PersonDetector | None = None,
    config_path: str | Path | None = None,
    model_name: str = "yolo11m.pt",
    confidence_threshold: float = 0.4,
    device: str | None = None,
    save_video: bool = True,
    save_sample_frames: bool = False,
    sample_frame_interval: int = 50,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Execute person detection against a video file or frame sequence.

    Outputs schema-compliant detections.csv and optional annotated video.
    """
    config = load_config(config_path) if config_path else {}
    final_model = config.get("model", model_name)
    final_conf = float(config.get("confidence_threshold", confidence_threshold))
    final_device = config.get("device", device)

    src_path = Path(source_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source does not exist: {src_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = PersonDetector(
            model_name=final_model,
            confidence_threshold=final_conf,
            device=final_device,
        )

    frame_records: list[FrameDetections] = []
    frames_with_detections = 0
    start_time = time.perf_counter()

    with VideoSource(source_path) as source:
        meta = source.metadata
        total_to_process = (
            min(meta.frame_count, max_frames)
            if max_frames is not None
            else meta.frame_count
        )

        writer: cv2.VideoWriter | None = None
        if save_video and total_to_process > 0:
            writer = create_video_writer(
                out_dir / "annotated_video.mp4",
                fps=meta.fps if meta.fps > 0 else 25.0,
                width=meta.width,
                height=meta.height,
            )

        frames_dir = out_dir / "annotated_frames"
        if save_sample_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        try:
            for frame_id, frame in source.iter_frames(stop=total_to_process):
                detections = detector.detect(frame)
                frame_records.append(FrameDetections(frame_id, detections))
                if detections:
                    frames_with_detections += 1

                if writer is not None or save_sample_frames:
                    annotated = draw_detections(frame, detections)
                    if writer is not None:
                        writer.write(annotated)
                    if save_sample_frames and (
                        frame_id % sample_frame_interval == 0
                        or frame_id == total_to_process - 1
                    ):
                        cv2.imwrite(
                            str(frames_dir / f"frame_{frame_id:06d}.jpg"),
                            annotated,
                        )
        finally:
            if writer is not None:
                writer.release()

    duration = max(1e-6, time.perf_counter() - start_time)
    csv_path = out_dir / "detections.csv"
    total_detections = write_detections_csv(frame_records, csv_path)

    summary: dict[str, Any] = {
        "source_path": str(Path(source_path).resolve()),
        "frame_count": len(frame_records),
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "model": detector.model_name,
        "confidence_threshold": detector.confidence_threshold,
        "device": detector.device,
        "duration_seconds": round(duration, 3),
        "fps": round(len(frame_records) / duration, 2),
        "detections_csv": str(csv_path.resolve()),
        "annotated_video": str((out_dir / "annotated_video.mp4").resolve())
        if save_video
        else None,
    }

    with (out_dir / "detection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_detection(
            args.source,
            output_dir=args.output_dir,
            config_path=args.config,
            model_name=args.model,
            confidence_threshold=args.conf,
            device=args.device,
            save_video=not args.no_video,
            save_sample_frames=args.sample_frames,
            max_frames=args.max_frames,
        )
        print(
            f"Processed {summary['frame_count']} frames in {summary['duration_seconds']}s "
            f"({summary['fps']} FPS) on {summary['device']}."
        )
        print(
            f"Detected {summary['total_detections']} person instances across "
            f"{summary['frames_with_detections']} frames."
        )
        print(f"Detections saved to: {summary['detections_csv']}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        parser.exit(2, f"error: {exc}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run person detection on video files or MOT17 frame sequences."
    )
    parser.add_argument("source", help="Path to video file or frame sequence directory")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="cv_projects/02_person_detection/outputs",
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
        help="Skip annotated video rendering",
    )
    parser.add_argument(
        "--sample-frames",
        action="store_true",
        help="Save annotated sample frames to annotated_frames/",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (useful for quick smoke tests)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
