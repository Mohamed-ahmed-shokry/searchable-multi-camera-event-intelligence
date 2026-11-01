"""Phase 7: Visual Evidence best-frame selector, padded person cropper,
annotated snapshot renderer, enriched events.csv exporter, path verifier,
and single-pass sequential video CLI entrypoint.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from generate_dwell_events import VALID_EVENT_TYPES, EventRecord
except ImportError:
    _subproject_6_scripts = (
        Path(__file__).resolve().parents[2] / "06_dwell_time_events" / "scripts"
    )
    if str(_subproject_6_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_6_scripts))
    from generate_dwell_events import VALID_EVENT_TYPES, EventRecord

EVIDENCE_EVENTS_CSV_HEADER: list[str] = [
    "event_id",
    "event_type",
    "track_id",
    "zone_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "snapshot_path",
    "crop_path",
]

VALID_BEST_FRAME_STRATEGIES: tuple[str, ...] = (
    "highest_confidence",
    "midpoint",
    "start_frame",
    "largest_area",
)


@dataclass(frozen=True)
class EvidenceConfig:
    """Configuration for Phase 7 best-frame selection and snapshot/crop extraction."""

    best_frame_strategy: str = "highest_confidence"
    crop_padding_ratio: float = 0.10
    min_padding_px: int = 8
    jpeg_quality: int = 92
    overlay_scene_geometry: bool = True
    contact_sheet_columns: int = 4
    contact_sheet_max_items: int = 12

    def __post_init__(self) -> None:
        if self.best_frame_strategy not in VALID_BEST_FRAME_STRATEGIES:
            raise ValueError(
                f"best_frame_strategy must be one of {VALID_BEST_FRAME_STRATEGIES}, "
                f"got {self.best_frame_strategy!r}"
            )
        if self.crop_padding_ratio < 0.0:
            raise ValueError(
                f"crop_padding_ratio must be >= 0.0, got {self.crop_padding_ratio}"
            )
        if self.min_padding_px < 0:
            raise ValueError(f"min_padding_px must be >= 0, got {self.min_padding_px}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(
                f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}"
            )
        if self.contact_sheet_columns < 1:
            raise ValueError(
                f"contact_sheet_columns must be >= 1, got {self.contact_sheet_columns}"
            )
        if self.contact_sheet_max_items < 1:
            raise ValueError(
                f"contact_sheet_max_items must be >= 1, got {self.contact_sheet_max_items}"
            )


@dataclass(frozen=True)
class BestFrameSelection:
    """Selected best frame and bounding box for a single structured event."""

    event_id: str
    track_id: int
    frame_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    strategy_used: str
    used_nearest_fallback: bool = False

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class EvidenceEventRecord:
    """An EventRecord enriched with relative/portable snapshot_path and crop_path."""

    event_id: str
    event_type: str
    track_id: int
    zone_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    snapshot_path: str
    crop_path: str
    best_frame_id: int = 0
    best_confidence: float = 0.0
    clipped_at_frame_edge: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {VALID_EVENT_TYPES}, got {self.event_type!r}"
            )
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if not self.zone_id or not self.zone_id.strip():
            raise ValueError("zone_id must be a non-empty string")
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be >= 0.0, got {self.start_time}")
        if self.end_time < self.start_time - 1e-9:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})"
            )
        if self.duration_seconds < 0.0:
            raise ValueError(
                f"duration_seconds must be >= 0.0, got {self.duration_seconds}"
            )
        if not self.snapshot_path or not self.snapshot_path.strip():
            raise ValueError("snapshot_path must be a non-empty string")
        if not self.crop_path or not self.crop_path.strip():
            raise ValueError("crop_path must be a non-empty string")

    @classmethod
    def from_event(
        cls,
        event: EventRecord,
        *,
        snapshot_path: str,
        crop_path: str,
        best_frame_id: int = 0,
        best_confidence: float = 0.0,
        clipped_at_frame_edge: bool = False,
    ) -> EvidenceEventRecord:
        return cls(
            event_id=event.event_id,
            event_type=event.event_type,
            track_id=event.track_id,
            zone_id=event.zone_id,
            start_time=event.start_time,
            end_time=event.end_time,
            duration_seconds=event.duration_seconds,
            snapshot_path=snapshot_path,
            crop_path=crop_path,
            best_frame_id=best_frame_id,
            best_confidence=best_confidence,
            clipped_at_frame_edge=clipped_at_frame_edge,
        )

    def to_row(self) -> list[str]:
        return [
            self.event_id,
            self.event_type,
            str(self.track_id),
            self.zone_id,
            f"{self.start_time:.4f}",
            f"{self.end_time:.4f}",
            f"{self.duration_seconds:.4f}",
            self.snapshot_path,
            self.crop_path,
        ]
