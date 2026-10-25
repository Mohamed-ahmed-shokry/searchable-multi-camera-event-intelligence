"""Phase 4: Trajectory Extraction data structures, anchor calculation,
rolling-mean smoothing, CSV exporters, trail/map renderers, and CLI.
"""

from __future__ import annotations

from dataclasses import dataclass

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
