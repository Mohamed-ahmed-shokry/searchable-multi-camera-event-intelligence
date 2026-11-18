"""Domain types for multi-run comparison, baseline diffing, and cross-run trends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Aggregated KPI snapshot of a single processed event_intelligence run."""

    run_id: str
    run_dir: str
    db_path: str
    cameras_count: int
    detections_count: int
    tracks_count: int
    trajectories_count: int
    events_count: int
    insights_count: int
    high_critical_insights_count: int
    total_dwell_seconds: float
    mean_track_duration_seconds: float
    mean_path_length_px: float
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        for name, val in (
            ("cameras_count", self.cameras_count),
            ("detections_count", self.detections_count),
            ("tracks_count", self.tracks_count),
            ("trajectories_count", self.trajectories_count),
            ("events_count", self.events_count),
            ("insights_count", self.insights_count),
            ("high_critical_insights_count", self.high_critical_insights_count),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if self.total_dwell_seconds < 0.0:
            raise ValueError("total_dwell_seconds must be >= 0.0")
        if self.mean_track_duration_seconds < 0.0:
            raise ValueError("mean_track_duration_seconds must be >= 0.0")
        if self.mean_path_length_px < 0.0:
            raise ValueError("mean_path_length_px must be >= 0.0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "db_path": self.db_path,
            "cameras_count": self.cameras_count,
            "detections_count": self.detections_count,
            "tracks_count": self.tracks_count,
            "trajectories_count": self.trajectories_count,
            "events_count": self.events_count,
            "insights_count": self.insights_count,
            "high_critical_insights_count": self.high_critical_insights_count,
            "total_dwell_seconds": round(self.total_dwell_seconds, 4),
            "mean_track_duration_seconds": round(self.mean_track_duration_seconds, 4),
            "mean_path_length_px": round(self.mean_path_length_px, 2),
            "config_snapshot": dict(self.config_snapshot),
        }


@dataclass(frozen=True, slots=True)
class ConfigDiffEntry:
    """Single flattened parameter change between baseline and candidate run_config.yaml."""

    key: str
    baseline_value: Any
    candidate_value: Any

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("key must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
        }
