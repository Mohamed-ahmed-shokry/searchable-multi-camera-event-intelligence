"""Phase 6: Dwell-Time Events stateful event lifecycle state machine
(`candidate -> active -> triggered -> resolved`), hysteresis / grace-period
flicker absorption, track-loss-in-zone handling, CSV/debug exporters,
visual proof renderers, and CLI entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from reason_zones_lines import (
        SceneConfig,
        TrajectoryPoint,
        VideoSource,
        create_video_writer,
        detect_line_signals,
        draw_trajectory_trails,
        draw_zones_and_lines,
        load_scene_config,
        point_in_zone,
        read_trajectories_csv,
    )
except ImportError:
    _subproject_5_scripts = (
        Path(__file__).resolve().parents[2] / "05_zone_line_reasoning" / "scripts"
    )
    if str(_subproject_5_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_5_scripts))
    from reason_zones_lines import (
        SceneConfig,
        TrajectoryPoint,
        VideoSource,
        create_video_writer,
        detect_line_signals,
        draw_trajectory_trails,
        draw_zones_and_lines,
        load_scene_config,
        point_in_zone,
        read_trajectories_csv,
    )

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


def evaluate_dwell_events(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    scene_config: SceneConfig,
    config: EventConfig | None = None,
) -> tuple[list[EventRecord], list[StayDebugTrace]]:
    """Evaluate stateful dwell/zone events and deduplicated line crossings across all tracks."""
    cfg = config or EventConfig()
    raw_events: list[EventRecord] = []
    traces: list[StayDebugTrace] = []
    zone_polys = [(z.id, z.shapely_polygon) for z in scene_config.zones]

    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        if not pts:
            continue
        for zid, poly in zone_polys:
            sm = ZoneDwellStateMachine(track_id=tid, zone_id=zid, config=cfg)
            for pt in pts:
                inside = point_in_zone(pt.x, pt.y, poly)
                raw_events.extend(sm.step(pt.frame_id, pt.timestamp, is_inside=inside))
            raw_events.extend(sm.finish_track())
            traces.extend(sm.completed_traces)

    if cfg.include_line_crossings and scene_config.lines:
        raw_line_sigs = detect_line_signals(trajectories, scene_config.lines)
        last_cross_frame: dict[tuple[int, str], int] = {}
        line_seq = 0
        for sig in raw_line_sigs:
            key = (sig.track_id, sig.line_id)
            if (
                key in last_cross_frame
                and (sig.frame_id - last_cross_frame[key]) <= cfg.line_cooldown_frames
            ):
                continue
            last_cross_frame[key] = sig.frame_id
            line_seq += 1
            raw_events.append(
                EventRecord(
                    event_id=f"line_{line_seq:04d}",
                    event_type="line_crossing",
                    track_id=sig.track_id,
                    zone_id=f"{sig.line_id}:{sig.direction}",
                    start_time=sig.timestamp,
                    end_time=sig.timestamp,
                    duration_seconds=0.0,
                    start_frame=sig.frame_id,
                    end_frame=sig.frame_id,
                )
            )

    ordered_raw = sorted(
        raw_events,
        key=lambda item: (
            item.start_time,
            item.end_time,
            item.track_id,
            item.event_type,
            item.zone_id,
        ),
    )
    numbered_events = [
        EventRecord(
            event_id=f"evt_{idx:04d}",
            event_type=ev.event_type,
            track_id=ev.track_id,
            zone_id=ev.zone_id,
            start_time=ev.start_time,
            end_time=ev.end_time,
            duration_seconds=ev.duration_seconds,
            start_frame=ev.start_frame,
            end_frame=ev.end_frame,
        )
        for idx, ev in enumerate(ordered_raw, start=1)
    ]
    ordered_traces = sorted(
        traces,
        key=lambda item: (item.enter_time, item.track_id, item.zone_id),
    )
    return numbered_events, ordered_traces


def write_event_debug_logs(
    debug_dir: Path | str,
    traces: Sequence[StayDebugTrace],
) -> Path:
    """Write per-stay debug trace files into event_debug_log/ and return stay_traces.json path."""
    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        traces,
        key=lambda item: (item.enter_time, item.track_id, item.zone_id, item.stay_id),
    )
    all_payload = [tr.to_dict() for tr in ordered]
    combined_path = out_dir / "stay_traces.json"
    combined_path.write_text(json.dumps(all_payload, indent=2), encoding="utf-8")

    for tr in ordered:
        single_path = out_dir / f"{tr.stay_id}.json"
        single_path.write_text(json.dumps(tr.to_dict(), indent=2), encoding="utf-8")

    return combined_path


def draw_dwell_events_overlay(
    frame: np.ndarray,
    scene_config: SceneConfig,
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    current_frame_id: int,
    current_timestamp: float,
    traces: Sequence[StayDebugTrace],
    events: Sequence[EventRecord],
    *,
    max_trail_length: int = 60,
) -> np.ndarray:
    """Render zones, lines, trajectories, live dwell timers, and recent event banners."""
    with_geom = draw_zones_and_lines(frame, scene_config, alpha=0.22)
    out = draw_trajectory_trails(
        with_geom,
        trajectories,
        current_frame_id=current_frame_id,
        max_trail_length=max_trail_length,
    )

    active_stays = [
        tr
        for tr in traces
        if tr.enter_frame <= current_frame_id <= tr.last_inside_frame
        and tr.peak_state in {"active", "triggered"}
    ]
    row_y = 26
    for tr in active_stays:
        elapsed = max(0.0, current_timestamp - tr.enter_time)
        tag = "DWELL" if tr.peak_state == "triggered" else "ACTIVE"
        color = (0, 255, 128) if tr.peak_state == "triggered" else (255, 220, 80)
        label = f"[{tag}] T{tr.track_id} @ {tr.zone_id}: {elapsed:.2f}s"
        cv2.putText(
            out,
            label,
            (12, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
        row_y += 22

    recent_events = [
        ev
        for ev in events
        if (
            ev.event_type == "dwell_time_event"
            and 0 <= current_frame_id - ev.end_frame <= 10
        )
        or (
            ev.event_type != "dwell_time_event"
            and 0 <= current_frame_id - ev.start_frame <= 6
        )
    ]
    for ev in recent_events[:4]:
        summary = (
            f"{ev.event_id} {ev.event_type} T{ev.track_id} {ev.zone_id} "
            f"({ev.duration_seconds:.2f}s)"
        )
        cv2.putText(
            out,
            summary,
            (12, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        row_y += 18

    return out


def save_dwell_examples(
    video_path: Path | str,
    examples_dir: Path | str,
    scene_config: SceneConfig,
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    traces: Sequence[StayDebugTrace],
    events: Sequence[EventRecord],
    *,
    max_examples: int = 6,
) -> list[Path]:
    """Save annotated visual proof PNGs for triggered dwell stays into dwell_examples/."""
    out_dir = Path(examples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [tr for tr in traces if tr.peak_state == "triggered"]
    if not candidates:
        candidates = [tr for tr in traces if tr.peak_state == "active"]
    selected = candidates[:max_examples]
    if not selected:
        return []

    target_by_frame: dict[int, list[StayDebugTrace]] = {}
    for tr in selected:
        target_by_frame.setdefault(tr.last_inside_frame, []).append(tr)

    max_target_frame = max(target_by_frame)
    saved_paths: list[Path] = []

    with VideoSource(video_path) as src:
        fps = src.metadata.fps if src.metadata.fps > 0 else 30.0
        for frame_id, frame in src.iter_frames(stop=max_target_frame + 1):
            if frame_id not in target_by_frame:
                continue
            ts = frame_id / fps
            rendered = draw_dwell_events_overlay(
                frame,
                scene_config,
                trajectories,
                current_frame_id=frame_id,
                current_timestamp=ts,
                traces=traces,
                events=events,
            )
            for tr in target_by_frame[frame_id]:
                img_path = out_dir / f"{tr.stay_id}_frame_{frame_id:04d}.png"
                cv2.imwrite(str(img_path), rendered)
                saved_paths.append(img_path)

    return saved_paths


def run_dwell_time_events(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    trajectories_csv: Path | str | None = None,
    trajectories: Mapping[int, Sequence[TrajectoryPoint]] | None = None,
    scene_config_path: Path | str | None = None,
    scene_config: SceneConfig | None = None,
    events_config_path: Path | str | None = None,
    events_config: EventConfig | None = None,
    max_trail_length: int = 60,
    max_frames: int | None = None,
    save_video: bool = True,
) -> dict[str, Any]:
    """Run the Phase 6 dwell-time state machine pipeline and export events, debug logs, and visuals."""
    if trajectories is None and trajectories_csv is None:
        raise ValueError("Either trajectories_csv or trajectories must be provided")
    if scene_config is None and scene_config_path is None:
        raise ValueError("Either scene_config_path or scene_config must be provided")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cfg_scene = (
        scene_config
        if scene_config is not None
        else load_scene_config(Path(scene_config_path))  # type: ignore[arg-type]
    )
    cfg_events = (
        events_config
        if events_config is not None
        else load_events_config(events_config_path)
    )
    loaded_trajs: dict[int, list[TrajectoryPoint]] = (
        {tid: list(pts) for tid, pts in trajectories.items()}
        if trajectories is not None
        else read_trajectories_csv(Path(trajectories_csv))  # type: ignore[arg-type]
    )
    if max_frames is not None:
        loaded_trajs = {
            tid: [pt for pt in pts if pt.frame_id < max_frames]
            for tid, pts in loaded_trajs.items()
        }
        loaded_trajs = {tid: pts for tid, pts in loaded_trajs.items() if pts}

    events, traces = evaluate_dwell_events(loaded_trajs, cfg_scene, cfg_events)
    events_csv_path = write_events_csv(out_dir / "events.csv", events)
    saved_events_cfg_path = save_events_config(out_dir / "events.yaml", cfg_events)
    debug_log_path = write_event_debug_logs(out_dir / "event_debug_log", traces)
    example_paths = save_dwell_examples(
        video_path,
        out_dir / "dwell_examples",
        cfg_scene,
        loaded_trajs,
        traces,
        events,
    )

    with VideoSource(video_path) as src:
        meta = src.metadata
        fps = meta.fps if meta.fps > 0 else 30.0
        width, height = meta.width, meta.height
        total_to_process = (
            min(meta.frame_count, max_frames)
            if max_frames is not None
            else meta.frame_count
        )

        video_out_path = out_dir / "annotated_events_video.mp4"
        writer: cv2.VideoWriter | None = None
        if save_video and total_to_process > 0:
            writer = create_video_writer(
                video_out_path,
                fps=fps,
                width=width,
                height=height,
            )

        processed_frames = 0
        try:
            for frame_id, frame in src.iter_frames(stop=total_to_process):
                if writer is not None:
                    ts = frame_id / fps
                    rendered = draw_dwell_events_overlay(
                        frame,
                        cfg_scene,
                        loaded_trajs,
                        current_frame_id=frame_id,
                        current_timestamp=ts,
                        traces=traces,
                        events=events,
                        max_trail_length=max_trail_length,
                    )
                    writer.write(rendered)
                processed_frames += 1
        finally:
            if writer is not None:
                writer.release()

    counts_by_type: dict[str, int] = {
        etype: sum(1 for ev in events if ev.event_type == etype)
        for etype in VALID_EVENT_TYPES
    }
    stays_by_peak_state: dict[str, int] = {
        "candidate": sum(1 for tr in traces if tr.peak_state == "candidate"),
        "active": sum(1 for tr in traces if tr.peak_state == "active"),
        "triggered": sum(1 for tr in traces if tr.peak_state == "triggered"),
    }
    track_lost_stays = [tr.stay_id for tr in traces if tr.track_lost_in_zone]
    total_flickers_absorbed = sum(tr.flicker_absorbed_count for tr in traces)
    elapsed_sec = time.perf_counter() - t0

    report: dict[str, Any] = {
        "video_path": str(video_path),
        "trajectories_csv": (
            str(trajectories_csv) if trajectories_csv is not None else None
        ),
        "scene_config_path": (
            str(scene_config_path) if scene_config_path is not None else None
        ),
        "events_config_path": (
            str(events_config_path)
            if events_config_path is not None
            else str(saved_events_cfg_path)
        ),
        "fps": round(fps, 3),
        "resolution": [width, height],
        "processed_frames": processed_frames,
        "num_tracks": len(loaded_trajs),
        "total_events": len(events),
        "events_per_frame_ratio": (
            round(len(events) / processed_frames, 4) if processed_frames > 0 else 0.0
        ),
        "counts_by_type": counts_by_type,
        "total_stays": len(traces),
        "stays_by_peak_state": stays_by_peak_state,
        "total_flickers_absorbed": total_flickers_absorbed,
        "track_lost_in_zone_count": len(track_lost_stays),
        "track_lost_stay_ids": track_lost_stays,
        "elapsed_seconds": round(elapsed_sec, 3),
        "outputs": {
            "events_csv": str(events_csv_path),
            "events_yaml": str(saved_events_cfg_path),
            "stay_traces_json": str(debug_log_path),
            "dwell_examples": [str(p) for p in example_paths],
            "annotated_events_video": str(video_out_path) if save_video else None,
        },
    }
    summary_path = out_dir / "dwell_events_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for generate_dwell_events.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6: Stateful dwell-time event lifecycle state machine "
            "(candidate -> active -> triggered -> resolved) with hysteresis."
        )
    )
    parser.add_argument(
        "video",
        help="Path to the input video file or MOT image-sequence directory.",
    )
    parser.add_argument(
        "--trajectories-csv",
        required=True,
        help="Path to Phase 4 trajectories.csv file.",
    )
    parser.add_argument(
        "--scene-config",
        default="cv_projects/05_zone_line_reasoning/configs/scene_config.yaml",
        help="Path to Phase 5 scene_config.yaml file.",
    )
    parser.add_argument(
        "--config",
        default="cv_projects/06_dwell_time_events/configs/events.yaml",
        help="Path to Phase 6 events.yaml configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="cv_projects/06_dwell_time_events/outputs",
        help="Directory for generated events.csv, event_debug_log/, and dwell_examples/.",
    )
    parser.add_argument(
        "--max-trail-length",
        type=int,
        default=60,
        help="Maximum trail length rendered on annotated_events_video.mp4.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing annotated_events_video.mp4.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_dwell_time_events(
        args.video,
        args.output_dir,
        trajectories_csv=args.trajectories_csv,
        scene_config_path=args.scene_config,
        events_config_path=args.config,
        max_trail_length=args.max_trail_length,
        max_frames=args.max_frames,
        save_video=not args.no_video,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
