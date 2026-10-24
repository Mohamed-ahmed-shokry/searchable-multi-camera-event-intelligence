"""Person detection runner and utilities for Phase 2.

Provides schema-compliant detection structures, YOLO11 inference filtering,
bounding box annotation, and CSV serialization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
