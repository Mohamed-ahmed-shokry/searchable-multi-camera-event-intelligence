"""Phase 6: Dwell-Time Events stateful event lifecycle state machine
(`candidate -> active -> triggered -> resolved`), hysteresis / grace-period
flicker absorption, track-loss-in-zone handling, CSV/debug exporters,
visual proof renderers, and CLI entrypoint.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
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


def write_events_csv(
    path: Path | str,
    events: Sequence[EventRecord],
) -> Path:
    """Write structured events to CSV matching EVENTS_CSV_HEADER."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        events,
        key=lambda item: (
            item.start_time,
            item.end_time,
            item.track_id,
            item.event_type,
            item.zone_id,
            item.event_id,
        ),
    )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVENTS_CSV_HEADER)
        for ev in ordered:
            writer.writerow(ev.to_row())
    return out_path


def read_events_csv(path: Path | str) -> list[EventRecord]:
    """Read an events.csv file into ordered EventRecord instances."""
    in_path = Path(path)
    events: list[EventRecord] = []
    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EVENTS_CSV_HEADER:
            raise ValueError(f"Expected CSV header {EVENTS_CSV_HEADER}, got {header!r}")
        for row in reader:
            if not row:
                continue
            events.append(
                EventRecord(
                    event_id=row[0],
                    event_type=row[1],
                    track_id=int(row[2]),
                    zone_id=row[3],
                    start_time=float(row[4]),
                    end_time=float(row[5]),
                    duration_seconds=float(row[6]),
                )
            )
    return sorted(
        events,
        key=lambda item: (
            item.start_time,
            item.end_time,
            item.track_id,
            item.event_type,
            item.zone_id,
            item.event_id,
        ),
    )


class ZoneDwellStateMachine:
    """Per-(track_id, zone_id) lifecycle state machine:
    outside -> candidate -> active -> triggered -> resolved.
    """

    def __init__(
        self,
        track_id: int,
        zone_id: str,
        config: EventConfig | None = None,
    ) -> None:
        self.track_id = track_id
        self.zone_id = zone_id
        self.config = config or EventConfig()
        self.dwell_threshold = self.config.zone_threshold(zone_id)

        self.state: str = "outside"
        self.peak_state: str = "outside"
        self.enter_frame: int = 0
        self.enter_time: float = 0.0
        self.last_inside_frame: int = 0
        self.last_inside_time: float = 0.0
        self.inside_frames: int = 0
        self.outside_streak: int = 0
        self.first_outside_frame: int = 0
        self.first_outside_time: float = 0.0
        self.flicker_absorbed_count: int = 0
        self.transitions: list[dict[str, Any]] = []

        self._stay_index: int = 0
        self._event_seq: int = 0
        self.completed_traces: list[StayDebugTrace] = []

    def _next_event_id(self, prefix: str) -> str:
        self._event_seq += 1
        return f"{prefix}_t{self.track_id}_{self.zone_id}_{self._event_seq:03d}"

    def current_dwell_seconds(self) -> float:
        """Return accumulated dwell seconds for the current active/triggered stay."""
        if self.state == "outside":
            return 0.0
        return max(0.0, self.last_inside_time - self.enter_time)

    def step(
        self,
        frame_id: int,
        timestamp: float,
        is_inside: bool,
    ) -> list[EventRecord]:
        """Advance state machine by one frame observation and return any emitted events."""
        emitted: list[EventRecord] = []

        if self.state == "outside":
            if not is_inside:
                return emitted
            self._stay_index += 1
            self.state = "candidate"
            self.peak_state = "candidate"
            self.enter_frame = frame_id
            self.enter_time = timestamp
            self.last_inside_frame = frame_id
            self.last_inside_time = timestamp
            self.inside_frames = 1
            self.outside_streak = 0
            self.first_outside_frame = frame_id
            self.first_outside_time = timestamp
            self.flicker_absorbed_count = 0
            self.transitions = [
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 4),
                    "from": "outside",
                    "to": "candidate",
                    "reason": "entered_zone",
                }
            ]
            emitted.extend(self._promote_if_eligible(frame_id, timestamp))
            return emitted

        # Currently in candidate, active, or triggered
        if is_inside:
            if self.outside_streak > 0:
                self.flicker_absorbed_count += 1
                self.transitions.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": round(timestamp, 4),
                        "from": self.state,
                        "to": self.state,
                        "reason": f"flicker_absorbed_after_{self.outside_streak}_frames",
                    }
                )
                self.outside_streak = 0
            self.last_inside_frame = frame_id
            self.last_inside_time = timestamp
            self.inside_frames += 1
            emitted.extend(self._promote_if_eligible(frame_id, timestamp))
            return emitted

        # Observation is outside while stay is open
        if self.outside_streak == 0:
            self.first_outside_frame = frame_id
            self.first_outside_time = timestamp
        self.outside_streak += 1

        if self.outside_streak > self.config.grace_frames:
            emitted.extend(
                self._resolve_current_stay(
                    resolved_frame=self.first_outside_frame,
                    resolved_time=self.first_outside_time,
                    reason="exited",
                    track_lost=False,
                )
            )
        return emitted

    def finish_track(self) -> list[EventRecord]:
        """Resolve any open stay when the track terminates (track loss inside zone)."""
        if self.state == "outside":
            return []
        return self._resolve_current_stay(
            resolved_frame=self.last_inside_frame,
            resolved_time=self.last_inside_time,
            reason="track_lost_in_zone",
            track_lost=True,
        )

    def _promote_if_eligible(
        self,
        frame_id: int,
        timestamp: float,
    ) -> list[EventRecord]:
        emitted: list[EventRecord] = []
        if (
            self.state == "candidate"
            and self.inside_frames >= self.config.min_entry_frames
        ):
            self.state = "active"
            self.peak_state = "active"
            self.transitions.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 4),
                    "from": "candidate",
                    "to": "active",
                    "reason": "min_entry_frames_met",
                }
            )
            emitted.append(
                EventRecord(
                    event_id=self._next_event_id("entry"),
                    event_type="zone_entry",
                    track_id=self.track_id,
                    zone_id=self.zone_id,
                    start_time=self.enter_time,
                    end_time=self.enter_time,
                    duration_seconds=0.0,
                    start_frame=self.enter_frame,
                    end_frame=self.enter_frame,
                )
            )

        if (
            self.state == "active"
            and self.current_dwell_seconds() >= self.dwell_threshold - 1e-9
        ):
            self.state = "triggered"
            self.peak_state = "triggered"
            self.transitions.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 4),
                    "from": "active",
                    "to": "triggered",
                    "reason": "dwell_threshold_reached",
                }
            )
        return emitted

    def _resolve_current_stay(
        self,
        *,
        resolved_frame: int,
        resolved_time: float,
        reason: str,
        track_lost: bool,
    ) -> list[EventRecord]:
        emitted: list[EventRecord] = []
        duration = self.current_dwell_seconds()
        prior_state = self.state
        resolution_label = (
            "discarded_candidate" if prior_state == "candidate" else reason
        )

        self.transitions.append(
            {
                "frame_id": resolved_frame,
                "timestamp": round(resolved_time, 4),
                "from": prior_state,
                "to": "resolved",
                "reason": resolution_label,
            }
        )

        if prior_state == "triggered":
            emitted.append(
                EventRecord(
                    event_id=self._next_event_id("dwell"),
                    event_type="dwell_time_event",
                    track_id=self.track_id,
                    zone_id=self.zone_id,
                    start_time=self.enter_time,
                    end_time=self.last_inside_time,
                    duration_seconds=duration,
                    start_frame=self.enter_frame,
                    end_frame=self.last_inside_frame,
                )
            )

        if prior_state in {"active", "triggered"} and not track_lost:
            emitted.append(
                EventRecord(
                    event_id=self._next_event_id("exit"),
                    event_type="zone_exit",
                    track_id=self.track_id,
                    zone_id=self.zone_id,
                    start_time=resolved_time,
                    end_time=resolved_time,
                    duration_seconds=0.0,
                    start_frame=resolved_frame,
                    end_frame=resolved_frame,
                )
            )

        self.completed_traces.append(
            StayDebugTrace(
                stay_id=f"stay_t{self.track_id}_{self.zone_id}_{self._stay_index:03d}",
                track_id=self.track_id,
                zone_id=self.zone_id,
                enter_frame=self.enter_frame,
                enter_time=self.enter_time,
                last_inside_frame=self.last_inside_frame,
                last_inside_time=self.last_inside_time,
                resolved_frame=resolved_frame,
                resolved_time=resolved_time,
                duration_seconds=duration,
                peak_state=self.peak_state,
                resolution_reason=resolution_label,
                track_lost_in_zone=track_lost,
                flicker_absorbed_count=self.flicker_absorbed_count,
                transitions=tuple(self.transitions),
            )
        )
        self.state = "outside"
        self.outside_streak = 0
        self.inside_frames = 0
        return emitted
