"""Phase 4: Trajectory Extraction data structures, anchor calculation,
rolling-mean smoothing, CSV exporters, trail/map renderers, and CLI.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from track_people import FrameTracks, read_tracks_csv, track_color
except ImportError:
    _subproject_3_scripts = (
        Path(__file__).resolve().parents[2] / "03_person_tracking" / "scripts"
    )
    if str(_subproject_3_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_3_scripts))
    from track_people import FrameTracks, read_tracks_csv, track_color  # noqa: F401

TRAJECTORIES_CSV_HEADER: list[str] = [
    "track_id",
    "frame_id",
    "timestamp",
    "x",
    "y",
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
