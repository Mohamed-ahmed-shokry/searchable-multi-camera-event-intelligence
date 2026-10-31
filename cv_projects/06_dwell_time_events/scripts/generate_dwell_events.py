"""Phase 6: Dwell-Time Events stateful event lifecycle state machine
(`candidate -> active -> triggered -> resolved`), hysteresis / grace-period
flicker absorption, track-loss-in-zone handling, CSV/debug exporters,
visual proof renderers, and CLI entrypoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

EVENTS_CSV_HEADER: list[str] = [
    "event_id",
    "event_type",
    "track_id",
    "zone_id",
    "start_time",
    "end_time",
    "duration_seconds",
]

VALID_EVENT_TYPES: tuple[str, ...] = (
    "zone_entry",
    "zone_exit",
    "dwell_time_event",
    "line_crossing",
)

VALID_STAY_STATES: tuple[str, ...] = (
    "outside",
    "candidate",
    "active",
    "triggered",
    "resolved",
)


@dataclass(frozen=True)
class EventConfig:
    """Configuration for Phase 6 dwell-time state machine and hysteresis rules."""

    min_dwell_seconds: float = 1.2
    min_entry_frames: int = 1
    grace_frames: int = 5
    line_cooldown_frames: int = 5
    include_line_crossings: bool = True
    per_zone_min_dwell_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_dwell_seconds < 0.0:
            raise ValueError(
                f"min_dwell_seconds must be >= 0.0, got {self.min_dwell_seconds}"
            )
        if self.min_entry_frames < 1:
            raise ValueError(
                f"min_entry_frames must be >= 1, got {self.min_entry_frames}"
            )
        if self.grace_frames < 0:
            raise ValueError(f"grace_frames must be >= 0, got {self.grace_frames}")
        if self.line_cooldown_frames < 0:
            raise ValueError(
                f"line_cooldown_frames must be >= 0, got {self.line_cooldown_frames}"
            )
        for zid, val in self.per_zone_min_dwell_seconds.items():
            if val < 0.0:
                raise ValueError(
                    f"per_zone_min_dwell_seconds[{zid!r}] must be >= 0.0, got {val}"
                )

    def zone_threshold(self, zone_id: str) -> float:
        """Return the minimum dwell threshold (in seconds) for `zone_id`."""
        return float(
            self.per_zone_min_dwell_seconds.get(zone_id, self.min_dwell_seconds)
        )


@dataclass(frozen=True)
class EventRecord:
    """A deduplicated structured event row for events.csv."""

    event_id: str
    event_type: str
    track_id: int
    zone_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    start_frame: int = 0
    end_frame: int = 0

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

    def to_row(self) -> list[str]:
        return [
            self.event_id,
            self.event_type,
            str(self.track_id),
            self.zone_id,
            f"{self.start_time:.4f}",
            f"{self.end_time:.4f}",
            f"{self.duration_seconds:.4f}",
        ]


@dataclass(frozen=True)
class StayDebugTrace:
    """Audit log of a single (track_id, zone_id) stay lifecycle."""

    stay_id: str
    track_id: int
    zone_id: str
    enter_frame: int
    enter_time: float
    last_inside_frame: int
    last_inside_time: float
    resolved_frame: int
    resolved_time: float
    duration_seconds: float
    peak_state: str
    resolution_reason: str
    track_lost_in_zone: bool
    flicker_absorbed_count: int
    transitions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stay_id": self.stay_id,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "enter_frame": self.enter_frame,
            "enter_time": round(self.enter_time, 4),
            "last_inside_frame": self.last_inside_frame,
            "last_inside_time": round(self.last_inside_time, 4),
            "resolved_frame": self.resolved_frame,
            "resolved_time": round(self.resolved_time, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "peak_state": self.peak_state,
            "resolution_reason": self.resolution_reason,
            "track_lost_in_zone": self.track_lost_in_zone,
            "flicker_absorbed_count": self.flicker_absorbed_count,
            "transitions": list(self.transitions),
        }


def load_events_config(path: Path | str | None = None) -> EventConfig:
    """Load and validate a Phase 6 events.yaml configuration file."""
    if path is None:
        return EventConfig()
    cfg_path = Path(path)
    if not cfg_path.exists():
        return EventConfig()

    raw: Any = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in {cfg_path}, got {type(raw).__name__}")

    per_zone_raw = raw.get("per_zone_min_dwell_seconds") or {}
    if not isinstance(per_zone_raw, dict):
        raise TypeError("per_zone_min_dwell_seconds must be a mapping")

    return EventConfig(
        min_dwell_seconds=float(raw.get("min_dwell_seconds", 1.2)),
        min_entry_frames=int(raw.get("min_entry_frames", 1)),
        grace_frames=int(raw.get("grace_frames", 5)),
        line_cooldown_frames=int(raw.get("line_cooldown_frames", 5)),
        include_line_crossings=bool(raw.get("include_line_crossings", True)),
        per_zone_min_dwell_seconds={str(k): float(v) for k, v in per_zone_raw.items()},
    )


def save_events_config(path: Path | str, config: EventConfig) -> Path:
    """Serialize an EventConfig to YAML matching configs/events.yaml."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "min_dwell_seconds": round(float(config.min_dwell_seconds), 4),
        "min_entry_frames": int(config.min_entry_frames),
        "grace_frames": int(config.grace_frames),
        "line_cooldown_frames": int(config.line_cooldown_frames),
        "include_line_crossings": bool(config.include_line_crossings),
        "per_zone_min_dwell_seconds": {
            str(k): round(float(v), 4)
            for k, v in config.per_zone_min_dwell_seconds.items()
        },
    }
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return out_path
