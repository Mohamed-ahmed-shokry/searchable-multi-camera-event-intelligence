"""Phase 4: Trajectory Extraction data structures, anchor calculation,
rolling-mean smoothing, CSV exporters, trail/map renderers, and CLI.
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from track_people import FrameTracks, TrackDetection, read_tracks_csv, track_color
except ImportError:
    _subproject_3_scripts = (
        Path(__file__).resolve().parents[2] / "03_person_tracking" / "scripts"
    )
    if str(_subproject_3_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_3_scripts))
    from track_people import (  # noqa: F401
        FrameTracks,
        TrackDetection,
        read_tracks_csv,
        track_color,
    )

TRAJECTORIES_CSV_HEADER: list[str] = [
    "track_id",
    "frame_id",
    "timestamp",
    "x",
    "y",
]

TRAJECTORY_SUMMARY_CSV_HEADER: list[str] = [
    "track_id",
    "path_length_px",
    "duration_seconds",
    "frame_count",
]

VALID_ANCHORS: tuple[str, ...] = ("bottom_center", "centroid")


def compute_anchor_point(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    anchor: str = "bottom_center",
) -> tuple[float, float]:
    """Compute the ground-plane anchor point (x, y) from a bounding box.

    - 'bottom_center': (0.5 * (x1 + x2), y2) — foot proxy on ground plane
    - 'centroid': (0.5 * (x1 + x2), 0.5 * (y1 + y2)) — bounding-box center
    """
    if anchor == "bottom_center":
        return 0.5 * (x1 + x2), float(y2)
    if anchor == "centroid":
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    raise ValueError(f"Unsupported anchor {anchor!r}; expected one of {VALID_ANCHORS}")


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single timestamped 2D trajectory coordinate for a tracked person."""

    track_id: int
    frame_id: int
    timestamp: float
    x: float
    y: float

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0, got {self.timestamp}")

    def to_row(self) -> list[str]:
        return [
            str(self.track_id),
            str(self.frame_id),
            f"{self.timestamp:.4f}",
            f"{self.x:.2f}",
            f"{self.y:.2f}",
        ]


def _flatten_trajectories(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]] | Sequence[TrajectoryPoint],
) -> list[TrajectoryPoint]:
    if isinstance(trajectories, Mapping):
        points: list[TrajectoryPoint] = []
        for track_id in sorted(trajectories):
            pts = sorted(trajectories[track_id], key=lambda item: item.frame_id)
            points.extend(pts)
        return points
    return sorted(trajectories, key=lambda item: (item.track_id, item.frame_id))


def write_trajectories_csv(
    path: Path | str,
    trajectories: Mapping[int, Sequence[TrajectoryPoint]] | Sequence[TrajectoryPoint],
) -> Path:
    """Write timestamped trajectory points to CSV matching TRAJECTORIES_CSV_HEADER."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = _flatten_trajectories(trajectories)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAJECTORIES_CSV_HEADER)
        for pt in ordered:
            writer.writerow(pt.to_row())
    return out_path


def read_trajectories_csv(path: Path | str) -> dict[int, list[TrajectoryPoint]]:
    """Read a trajectories.csv file into a per-track dictionary of sorted TrajectoryPoints."""
    in_path = Path(path)
    by_track: dict[int, list[TrajectoryPoint]] = {}
    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != TRAJECTORIES_CSV_HEADER:
            raise ValueError(
                f"Expected CSV header {TRAJECTORIES_CSV_HEADER}, got {header!r}"
            )
        for row in reader:
            if not row:
                continue
            pt = TrajectoryPoint(
                track_id=int(row[0]),
                frame_id=int(row[1]),
                timestamp=float(row[2]),
                x=float(row[3]),
                y=float(row[4]),
            )
            by_track.setdefault(pt.track_id, []).append(pt)
    return {
        tid: sorted(by_track[tid], key=lambda item: item.frame_id)
        for tid in sorted(by_track)
    }


def tracks_to_trajectories(
    frame_tracks: Sequence[FrameTracks],
    fps: float,
    *,
    anchor: str = "bottom_center",
) -> dict[int, list[TrajectoryPoint]]:
    """Convert per-frame TrackDetections into per-track ordered TrajectoryPoint lists."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if anchor not in VALID_ANCHORS:
        raise ValueError(
            f"Unsupported anchor {anchor!r}; expected one of {VALID_ANCHORS}"
        )

    by_track: dict[int, list[TrajectoryPoint]] = {}
    for ft in frame_tracks:
        for trk in ft.tracks:
            ax, ay = compute_anchor_point(trk.x1, trk.y1, trk.x2, trk.y2, anchor=anchor)
            pt = TrajectoryPoint(
                track_id=trk.track_id,
                frame_id=trk.frame_id,
                timestamp=trk.frame_id / fps,
                x=ax,
                y=ay,
            )
            by_track.setdefault(trk.track_id, []).append(pt)

    return {
        tid: sorted(by_track[tid], key=lambda item: item.frame_id)
        for tid in sorted(by_track)
    }


def smooth_trajectories(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    window: int = 3,
) -> dict[int, list[TrajectoryPoint]]:
    """Apply a causal rolling-mean filter of size `window` independently per track."""
    if window < 1:
        raise ValueError(f"smoothing window must be >= 1, got {window}")

    smoothed: dict[int, list[TrajectoryPoint]] = {}
    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        if window == 1 or len(pts) <= 1:
            smoothed[tid] = list(pts)
            continue

        track_out: list[TrajectoryPoint] = []
        for idx, pt in enumerate(pts):
            start = max(0, idx - window + 1)
            win_pts = pts[start : idx + 1]
            mean_x = sum(p.x for p in win_pts) / len(win_pts)
            mean_y = sum(p.y for p in win_pts) / len(win_pts)
            track_out.append(
                TrajectoryPoint(
                    track_id=pt.track_id,
                    frame_id=pt.frame_id,
                    timestamp=pt.timestamp,
                    x=mean_x,
                    y=mean_y,
                )
            )
        smoothed[tid] = track_out
    return smoothed


@dataclass(frozen=True)
class TrajectorySummary:
    """Per-track trajectory metrics matching Phase 4 track_summary.csv schema."""

    track_id: int
    path_length_px: float
    duration_seconds: float
    frame_count: int

    def to_row(self) -> list[str]:
        return [
            str(self.track_id),
            f"{self.path_length_px:.2f}",
            f"{self.duration_seconds:.3f}",
            str(self.frame_count),
        ]


def compute_trajectory_summaries(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    fps: float,
) -> list[TrajectorySummary]:
    """Compute per-track Euclidean path length, span duration, and frame count."""
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    summaries: list[TrajectorySummary] = []
    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        if not pts:
            continue
        path_length = sum(
            math.hypot(nxt.x - cur.x, nxt.y - cur.y)
            for cur, nxt in itertools.pairwise(pts)
        )
        duration = (pts[-1].frame_id - pts[0].frame_id + 1) / fps
        summaries.append(
            TrajectorySummary(
                track_id=tid,
                path_length_px=path_length,
                duration_seconds=duration,
                frame_count=len(pts),
            )
        )
    return summaries


def write_trajectory_summary_csv(
    path: Path | str,
    summaries: Sequence[TrajectorySummary],
) -> Path:
    """Write per-track trajectory summaries to CSV matching TRAJECTORY_SUMMARY_CSV_HEADER."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(summaries, key=lambda item: item.track_id)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRAJECTORY_SUMMARY_CSV_HEADER)
        for item in ordered:
            writer.writerow(item.to_row())
    return out_path


def read_trajectory_summary_csv(path: Path | str) -> list[TrajectorySummary]:
    """Read a Phase 4 track_summary.csv file into TrajectorySummary records."""
    in_path = Path(path)
    summaries: list[TrajectorySummary] = []
    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != TRAJECTORY_SUMMARY_CSV_HEADER:
            raise ValueError(
                f"Expected CSV header {TRAJECTORY_SUMMARY_CSV_HEADER}, got {header!r}"
            )
        for row in reader:
            if not row:
                continue
            summaries.append(
                TrajectorySummary(
                    track_id=int(row[0]),
                    path_length_px=float(row[1]),
                    duration_seconds=float(row[2]),
                    frame_count=int(row[3]),
                )
            )
    return sorted(summaries, key=lambda item: item.track_id)


def analyze_trajectory_quality(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    *,
    jump_threshold_px: float = 150.0,
) -> dict[str, object]:
    """Quantify step distances, second-order coordinate jitter, and teleport jumps."""
    steps: list[float] = []
    accelerations: list[float] = []
    teleport_jumps: list[dict[str, float | int]] = []
    total_points = 0

    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        total_points += len(pts)
        for cur, nxt in itertools.pairwise(pts):
            dist = math.hypot(nxt.x - cur.x, nxt.y - cur.y)
            steps.append(dist)
            if dist > jump_threshold_px:
                teleport_jumps.append(
                    {
                        "track_id": tid,
                        "from_frame": cur.frame_id,
                        "to_frame": nxt.frame_id,
                        "distance_px": round(dist, 2),
                    }
                )
        for idx in range(len(pts) - 2):
            p0, p1, p2 = pts[idx], pts[idx + 1], pts[idx + 2]
            ax = (p2.x - p1.x) - (p1.x - p0.x)
            ay = (p2.y - p1.y) - (p1.y - p0.y)
            accelerations.append(math.hypot(ax, ay))

    return {
        "num_tracks": len(trajectories),
        "total_points": total_points,
        "mean_step_px": round(sum(steps) / len(steps), 3) if steps else 0.0,
        "max_step_px": round(max(steps), 3) if steps else 0.0,
        "mean_jitter_px": (
            round(sum(accelerations) / len(accelerations), 3) if accelerations else 0.0
        ),
        "teleport_jump_count": len(teleport_jumps),
        "teleport_jumps": teleport_jumps,
    }


def draw_trajectory_trails(
    frame: np.ndarray,
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    current_frame_id: int,
    *,
    max_trail_length: int = 60,
    thickness: int = 2,
    current_boxes: Sequence[TrackDetection] | None = None,
) -> np.ndarray:
    """Draw motion trails, anchor dots, and optional bounding boxes for active tracks."""
    canvas = frame.copy()
    if current_boxes:
        for box in current_boxes:
            color = track_color(box.track_id)
            p1 = (int(round(box.x1)), int(round(box.y1)))
            p2 = (int(round(box.x2)), int(round(box.y2)))
            cv2.rectangle(canvas, p1, p2, color, 1)

    for tid in sorted(trajectories):
        pts_up_to = [pt for pt in trajectories[tid] if pt.frame_id <= current_frame_id]
        if not pts_up_to or pts_up_to[-1].frame_id != current_frame_id:
            continue
        trail = pts_up_to[-max_trail_length:] if max_trail_length > 0 else pts_up_to
        color = track_color(tid)
        for cur, nxt in itertools.pairwise(trail):
            c_xy = (int(round(cur.x)), int(round(cur.y)))
            n_xy = (int(round(nxt.x)), int(round(nxt.y)))
            cv2.line(canvas, c_xy, n_xy, color, thickness, cv2.LINE_AA)

        head = trail[-1]
        head_xy = (int(round(head.x)), int(round(head.y)))
        cv2.circle(canvas, head_xy, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"ID {tid}",
            (head_xy[0] + 6, max(14, head_xy[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def render_trajectory_map(
    reference_frame: np.ndarray,
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    *,
    thickness: int = 2,
    dim_factor: float = 0.55,
) -> np.ndarray:
    """Render full-clip trajectory polylines for all tracks onto a reference frame."""
    canvas = np.clip(reference_frame.astype(np.float32) * dim_factor, 0, 255).astype(
        np.uint8
    )
    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        if not pts:
            continue
        color = track_color(tid)
        for cur, nxt in itertools.pairwise(pts):
            c_xy = (int(round(cur.x)), int(round(cur.y)))
            n_xy = (int(round(nxt.x)), int(round(nxt.y)))
            cv2.line(canvas, c_xy, n_xy, color, thickness, cv2.LINE_AA)

        start_xy = (int(round(pts[0].x)), int(round(pts[0].y)))
        end_xy = (int(round(pts[-1].x)), int(round(pts[-1].y)))
        cv2.circle(canvas, start_xy, 5, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, end_xy, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"ID {tid}",
            (end_xy[0] + 6, max(14, end_xy[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
