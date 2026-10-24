"""Self-contained MOT17 ground-truth parser and CLEAR-MOT (MOTA, MOTP, IDSW)
+ IDF1 evaluator for Phase 3 quantitative tracking checkpoints.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from track_people import TrackDetection


def load_mot17_gt(
    gt_path: Path | str,
    *,
    min_visibility: float = 0.0,
    max_frames: int | None = None,
) -> dict[int, list[TrackDetection]]:
    """Load MOT17 ground-truth annotations (1-based frames, xywh) into 0-based Frame dict.

    Filters to active pedestrian annotations (`mark == 1`, `class_id == 1`,
    and `visibility >= min_visibility`).
    """
    path = Path(gt_path)
    if not path.is_file():
        raise FileNotFoundError(f"MOT17 ground-truth file not found: {path}")

    gt_by_frame: dict[int, list[TrackDetection]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 9:
                continue
            frame_1based = int(row[0].strip())
            frame_id = frame_1based - 1
            if frame_id < 0:
                continue
            if max_frames is not None and frame_id >= max_frames:
                continue

            mark = int(float(row[6].strip()))
            cls_id = int(float(row[7].strip()))
            visibility = float(row[8].strip())
            if mark != 1 or cls_id != 1 or visibility < min_visibility:
                continue

            track_id = int(row[1].strip())
            x1 = float(row[2].strip())
            y1 = float(row[3].strip())
            w = float(row[4].strip())
            h = float(row[5].strip())
            if w <= 0 or h <= 0:
                continue

            gt_by_frame.setdefault(frame_id, []).append(
                TrackDetection(
                    frame_id=frame_id,
                    track_id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x1 + w,
                    y2=y1 + h,
                    confidence=1.0,
                )
            )

    return gt_by_frame


def box_iou(a: TrackDetection, b: TrackDetection) -> float:
    """Compute Intersection-over-Union (IoU) between two TrackDetection boxes."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def compute_iou_matrix(
    gt_boxes: Sequence[TrackDetection],
    pred_boxes: Sequence[TrackDetection],
) -> np.ndarray:
    """Compute an (M, N) pairwise IoU matrix between ground-truth and predicted boxes."""
    m = len(gt_boxes)
    n = len(pred_boxes)
    matrix = np.zeros((m, n), dtype=np.float64)
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            matrix[i, j] = box_iou(gt, pred)
    return matrix
