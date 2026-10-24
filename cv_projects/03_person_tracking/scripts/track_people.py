"""Phase 3: Person Tracking data structures, ByteTrack/BoT-SORT wrapper,
CSV exporters, track anomaly analyzer, and CLI entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
