"""Phase 10 — Searchable Outputs: SQLite ingestion (intelligence.db), CLI and
programmatic queries for events, tracks, and visual evidence, and worked examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Configuration for Phase 10 SQLite ingestion and query execution."""

    source_name: str = "people-detection.mp4"
    detections_csv: str = (
        "cv_projects/02_person_detection/outputs/people_detection_run/detections.csv"
    )
    tracks_csv: str = (
        "cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv"
    )
    tracking_summary_csv: str = (
        "cv_projects/03_person_tracking/outputs/"
        "people_detection_bytetrack/track_summary.csv"
    )
    trajectories_csv: str = (
        "cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv"
    )
    trajectory_summary_csv: str = (
        "cv_projects/04_trajectory_extraction/outputs/"
        "people_detection/track_summary.csv"
    )
    events_csv: str = (
        "cv_projects/07_visual_evidence/outputs/people_detection/events.csv"
    )
    evidence_root: str = "cv_projects/07_visual_evidence/outputs/people_detection"
    zone_counts_csv: str = (
        "cv_projects/08_heatmaps_analytics/outputs/people_detection/zone_counts.csv"
    )
    line_counts_csv: str = (
        "cv_projects/08_heatmaps_analytics/outputs/people_detection/line_counts.csv"
    )
    default_query_limit: int = 50

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name must be non-empty")
        if self.default_query_limit <= 0:
            raise ValueError("default_query_limit must be positive")


@dataclass(frozen=True, slots=True)
class EventSearchResult:
    """Structured event row returned by query_events."""

    event_id: str
    event_type: str
    track_id: int
    zone_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    snapshot_path: str
    crop_path: str

    def __post_init__(self) -> None:
        if not self.event_id.strip() or not self.event_type.strip():
            raise ValueError("event_id and event_type must be non-empty")
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "snapshot_path": self.snapshot_path,
            "crop_path": self.crop_path,
        }


@dataclass(frozen=True, slots=True)
class TrackSearchResult:
    """Structured track summary row returned by query_tracks."""

    track_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    observed_frames: int
    path_length_px: float
    event_count: int = 0

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        if self.duration_frames <= 0 or self.duration_seconds < 0.0:
            raise ValueError(
                "duration_frames must be positive and duration_seconds >= 0"
            )
        if self.path_length_px < 0.0 or self.event_count < 0:
            raise ValueError("path_length_px and event_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_seconds": round(self.duration_seconds, 4),
            "observed_frames": self.observed_frames,
            "path_length_px": round(self.path_length_px, 2),
            "event_count": self.event_count,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLookupResult:
    """Result of looking up visual evidence for a specific event_id."""

    event: EventSearchResult
    resolved_snapshot_path: str
    resolved_crop_path: str
    snapshot_exists: bool
    crop_exists: bool
    snapshot_size_bytes: int
    crop_size_bytes: int

    @property
    def is_valid(self) -> bool:
        return (
            self.snapshot_exists
            and self.crop_exists
            and self.snapshot_size_bytes > 0
            and self.crop_size_bytes > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "resolved_snapshot_path": self.resolved_snapshot_path,
            "resolved_crop_path": self.resolved_crop_path,
            "snapshot_exists": self.snapshot_exists,
            "crop_exists": self.crop_exists,
            "snapshot_size_bytes": self.snapshot_size_bytes,
            "crop_size_bytes": self.crop_size_bytes,
            "is_valid": self.is_valid,
        }


@dataclass(frozen=True, slots=True)
class DatabaseSummary:
    """Row counts across all tables in intelligence.db."""

    db_path: str
    source_name: str
    detections_count: int
    track_observations_count: int
    tracks_count: int
    trajectories_count: int
    events_count: int
    zone_counts_rows: int
    line_counts_rows: int

    def __post_init__(self) -> None:
        for name, val in (
            ("detections_count", self.detections_count),
            ("track_observations_count", self.track_observations_count),
            ("tracks_count", self.tracks_count),
            ("trajectories_count", self.trajectories_count),
            ("events_count", self.events_count),
            ("zone_counts_rows", self.zone_counts_rows),
            ("line_counts_rows", self.line_counts_rows),
        ):
            if val < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "source_name": self.source_name,
            "detections_count": self.detections_count,
            "track_observations_count": self.track_observations_count,
            "tracks_count": self.tracks_count,
            "trajectories_count": self.trajectories_count,
            "events_count": self.events_count,
            "zone_counts_rows": self.zone_counts_rows,
            "line_counts_rows": self.line_counts_rows,
        }


def load_search_config(path: Path | str | None = None) -> SearchConfig:
    """Load a SearchConfig from a YAML file, or return defaults if None."""
    if path is None:
        return SearchConfig()
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Search config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return SearchConfig()
    if not isinstance(raw, dict):
        raise TypeError("Search config YAML root must be a mapping")

    defaults = SearchConfig()
    return SearchConfig(
        source_name=str(raw.get("source_name", defaults.source_name)),
        detections_csv=str(raw.get("detections_csv", defaults.detections_csv)),
        tracks_csv=str(raw.get("tracks_csv", defaults.tracks_csv)),
        tracking_summary_csv=str(
            raw.get("tracking_summary_csv", defaults.tracking_summary_csv)
        ),
        trajectories_csv=str(raw.get("trajectories_csv", defaults.trajectories_csv)),
        trajectory_summary_csv=str(
            raw.get("trajectory_summary_csv", defaults.trajectory_summary_csv)
        ),
        events_csv=str(raw.get("events_csv", defaults.events_csv)),
        evidence_root=str(raw.get("evidence_root", defaults.evidence_root)),
        zone_counts_csv=str(raw.get("zone_counts_csv", defaults.zone_counts_csv)),
        line_counts_csv=str(raw.get("line_counts_csv", defaults.line_counts_csv)),
        default_query_limit=int(
            raw.get("default_query_limit", defaults.default_query_limit)
        ),
    )


def save_search_config(config: SearchConfig, path: Path | str) -> Path:
    """Serialize a SearchConfig to YAML."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_name": config.source_name,
        "detections_csv": config.detections_csv,
        "tracks_csv": config.tracks_csv,
        "tracking_summary_csv": config.tracking_summary_csv,
        "trajectories_csv": config.trajectories_csv,
        "trajectory_summary_csv": config.trajectory_summary_csv,
        "events_csv": config.events_csv,
        "evidence_root": config.evidence_root,
        "zone_counts_csv": config.zone_counts_csv,
        "line_counts_csv": config.line_counts_csv,
        "default_query_limit": config.default_query_limit,
    }
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return out_path
