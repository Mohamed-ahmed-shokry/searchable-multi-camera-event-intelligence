"""Self-contained MOT17 ground-truth parser and CLEAR-MOT (MOTA, MOTP, IDSW)
+ IDF1 evaluator for Phase 3 quantitative tracking checkpoints.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from track_people import FrameTracks, TrackDetection


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


def _normalize_pred_by_frame(
    predictions: Mapping[int, Sequence[TrackDetection]] | Sequence[FrameTracks],
) -> dict[int, list[TrackDetection]]:
    if isinstance(predictions, Mapping):
        return {int(k): list(v) for k, v in predictions.items()}
    return {ft.frame_id: list(ft.tracks) for ft in predictions}


def evaluate_tracking(
    gt_by_frame: Mapping[int, Sequence[TrackDetection]],
    predictions: Mapping[int, Sequence[TrackDetection]] | Sequence[FrameTracks],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute CLEAR-MOT (MOTA, MOTP, IDSW, FP, FN) and IDF1 metrics."""
    pred_by_frame = _normalize_pred_by_frame(predictions)
    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    total_gt = 0
    total_pred = 0
    total_matches = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    iou_sum = 0.0

    last_pred_for_gt: dict[int, int] = {}
    id_switch_events: list[dict[str, int | float]] = []
    pair_overlap_counts: dict[tuple[int, int], int] = {}

    for frame_id in all_frames:
        gt_list = list(gt_by_frame.get(frame_id, ()))
        pred_list = list(pred_by_frame.get(frame_id, ()))
        total_gt += len(gt_list)
        total_pred += len(pred_list)

        if not gt_list:
            total_fp += len(pred_list)
            continue
        if not pred_list:
            total_fn += len(gt_list)
            continue

        iou_mat = compute_iou_matrix(gt_list, pred_list)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        frame_pairs: list[tuple[int, int, float]] = []

        # Step 1: Carry over previous frame's correspondence if IoU >= threshold
        pred_idx_by_id = {p.track_id: j for j, p in enumerate(pred_list)}
        for i, gt in enumerate(gt_list):
            prev_pred_id = last_pred_for_gt.get(gt.track_id)
            if prev_pred_id is not None and prev_pred_id in pred_idx_by_id:
                j = pred_idx_by_id[prev_pred_id]
                if j not in matched_pred and iou_mat[i, j] >= iou_threshold:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    frame_pairs.append((i, j, float(iou_mat[i, j])))

        # Step 2: Greedy bipartite matching by descending IoU for remaining boxes
        candidates: list[tuple[float, int, int]] = []
        for i in range(len(gt_list)):
            if i in matched_gt:
                continue
            for j in range(len(pred_list)):
                if j in matched_pred:
                    continue
                score = float(iou_mat[i, j])
                if score >= iou_threshold:
                    candidates.append((score, i, j))
        candidates.sort(key=lambda item: item[0], reverse=True)

        for score, i, j in candidates:
            if i in matched_gt or j in matched_pred:
                continue
            matched_gt.add(i)
            matched_pred.add(j)
            frame_pairs.append((i, j, score))

        for i, j, score in frame_pairs:
            gt = gt_list[i]
            pred = pred_list[j]
            prev_id = last_pred_for_gt.get(gt.track_id)
            if prev_id is not None and prev_id != pred.track_id:
                id_switches += 1
                id_switch_events.append(
                    {
                        "frame_id": frame_id,
                        "gt_track_id": gt.track_id,
                        "old_pred_id": prev_id,
                        "new_pred_id": pred.track_id,
                        "iou": round(score, 4),
                    }
                )
            last_pred_for_gt[gt.track_id] = pred.track_id
            total_matches += 1
            iou_sum += score
            key = (gt.track_id, pred.track_id)
            pair_overlap_counts[key] = pair_overlap_counts.get(key, 0) + 1

        total_fn += len(gt_list) - len(matched_gt)
        total_fp += len(pred_list) - len(matched_pred)

    mota = 1.0 - (total_fn + total_fp + id_switches) / total_gt if total_gt > 0 else 0.0
    motp = iou_sum / total_matches if total_matches > 0 else 0.0
    precision = total_matches / total_pred if total_pred > 0 else 0.0
    recall = total_matches / total_gt if total_gt > 0 else 0.0

    # Global identity assignment for IDF1
    id_candidates = sorted(
        ((count, gid, pid) for (gid, pid), count in pair_overlap_counts.items()),
        reverse=True,
    )
    assigned_gt_ids: set[int] = set()
    assigned_pred_ids: set[int] = set()
    idtp = 0
    for count, gid, pid in id_candidates:
        if gid in assigned_gt_ids or pid in assigned_pred_ids:
            continue
        assigned_gt_ids.add(gid)
        assigned_pred_ids.add(pid)
        idtp += count

    denom = total_gt + total_pred
    idf1 = (2.0 * idtp / denom) if denom > 0 else 1.0

    return {
        "mota": round(mota, 4),
        "motp": round(motp, 4),
        "idf1": round(idf1, 4),
        "id_switches": id_switches,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "num_gt": total_gt,
        "num_predictions": total_pred,
        "matches": total_matches,
        "fp": total_fp,
        "fn": total_fn,
        "id_switch_events": id_switch_events,
    }
