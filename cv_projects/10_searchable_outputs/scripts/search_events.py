"""Phase 10 — Searchable Outputs: SQLite ingestion (intelligence.db), CLI and
programmatic queries for events, tracks, and visual evidence, and worked examples.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    class_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON detections (frame_id);

CREATE TABLE IF NOT EXISTS track_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_track_obs_track_frame ON track_observations (track_id, frame_id);

CREATE TABLE IF NOT EXISTS tracks (
    track_id INTEGER PRIMARY KEY,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    duration_frames INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    observed_frames INTEGER NOT NULL,
    path_length_px REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracks_duration ON tracks (duration_seconds DESC);

CREATE TABLE IF NOT EXISTS trajectories (
    track_id INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    PRIMARY KEY (track_id, frame_id)
);
CREATE INDEX IF NOT EXISTS idx_trajectories_timestamp ON trajectories (timestamp);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    zone_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    snapshot_path TEXT NOT NULL,
    crop_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type_zone ON events (event_type, zone_id);
CREATE INDEX IF NOT EXISTS idx_events_track_id ON events (track_id);
CREATE INDEX IF NOT EXISTS idx_events_time_range ON events (start_time, end_time);

CREATE TABLE IF NOT EXISTS zone_counts (
    zone_id TEXT PRIMARY KEY,
    entry_count INTEGER NOT NULL,
    exit_count INTEGER NOT NULL,
    total_dwell_seconds REAL NOT NULL,
    peak_occupancy INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS line_counts (
    line_id TEXT PRIMARY KEY,
    crossings_ab INTEGER NOT NULL,
    crossings_ba INTEGER NOT NULL,
    total INTEGER NOT NULL
);
"""


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


def init_intelligence_db(db_path: Path | str) -> Path:
    """Create intelligence.db and execute SCHEMA_DDL idempotently."""
    out_db = Path(db_path)
    out_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(out_db)) as conn:
        conn.executescript(SCHEMA_DDL)
        conn.commit()
    return out_db


def _resolve_optional_path(rel_or_abs: str, project_root: Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (project_root / p)


def ingest_intelligence_db(
    db_path: Path | str,
    config: SearchConfig,
    *,
    project_root: Path | str | None = None,
) -> DatabaseSummary:
    """Load detections, tracks, trajectories, events, and zone/line counts into intelligence.db."""
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT
    out_db = init_intelligence_db(db_path)

    det_path = _resolve_optional_path(config.detections_csv, root)
    trk_path = _resolve_optional_path(config.tracks_csv, root)
    trk_sum_path = _resolve_optional_path(config.tracking_summary_csv, root)
    traj_path = _resolve_optional_path(config.trajectories_csv, root)
    traj_sum_path = _resolve_optional_path(config.trajectory_summary_csv, root)
    ev_path = _resolve_optional_path(config.events_csv, root)
    zc_path = _resolve_optional_path(config.zone_counts_csv, root)
    lc_path = _resolve_optional_path(config.line_counts_csv, root)

    det_rows: list[tuple[int, float, float, float, float, float, str]] = []
    if det_path.is_file():
        with det_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("x1") or not str(row["x1"]).strip():
                    continue
                det_rows.append(
                    (
                        int(row["frame_id"]),
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                        float(row["confidence"]),
                        str(row.get("class") or "person").strip(),
                    )
                )

    obs_rows: list[tuple[int, int, float, float, float, float, float]] = []
    if trk_path.is_file():
        with trk_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("track_id") or not str(row["track_id"]).strip():
                    continue
                obs_rows.append(
                    (
                        int(row["frame_id"]),
                        int(row["track_id"]),
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                        float(row["confidence"]),
                    )
                )

    traj_rows: list[tuple[int, int, float, float, float]] = []
    traj_points_by_track: dict[int, list[tuple[int, float, float, float]]] = {}
    if traj_path.is_file():
        with traj_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("track_id") or not str(row["track_id"]).strip():
                    continue
                tid = int(row["track_id"])
                fid = int(row["frame_id"])
                ts = float(row["timestamp"])
                x = float(row["x"])
                y = float(row["y"])
                traj_rows.append((tid, fid, ts, x, y))
                traj_points_by_track.setdefault(tid, []).append((fid, ts, x, y))

    path_len_by_track: dict[int, float] = {}
    if traj_sum_path.is_file():
        with traj_sum_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("track_id") or not str(row["track_id"]).strip():
                    continue
                path_len_by_track[int(row["track_id"])] = float(row["path_length_px"])
    else:
        for tid, pts in traj_points_by_track.items():
            ordered = sorted(pts, key=lambda item: item[0])
            dist = sum(
                math.hypot(b[2] - a[2], b[3] - a[3]) for a, b in pairwise(ordered)
            )
            path_len_by_track[tid] = round(dist, 2)

    track_rows: list[tuple[int, int, int, int, float, int, float]] = []
    if trk_sum_path.is_file():
        with trk_sum_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("track_id") or not str(row["track_id"]).strip():
                    continue
                tid = int(row["track_id"])
                track_rows.append(
                    (
                        tid,
                        int(row["start_frame"]),
                        int(row["end_frame"]),
                        int(row["duration_frames"]),
                        float(row["duration_seconds"]),
                        int(row["observed_frames"]),
                        float(path_len_by_track.get(tid, 0.0)),
                    )
                )
    else:
        for tid in sorted(traj_points_by_track):
            ordered = sorted(traj_points_by_track[tid], key=lambda item: item[0])
            sf = ordered[0][0]
            ef = ordered[-1][0]
            dur_f = max(1, ef - sf + 1)
            dur_s = max(0.0, ordered[-1][1] - ordered[0][1])
            track_rows.append(
                (
                    tid,
                    sf,
                    ef,
                    dur_f,
                    round(dur_s, 4),
                    len(ordered),
                    float(path_len_by_track.get(tid, 0.0)),
                )
            )

    event_rows: list[tuple[str, str, int, str, float, float, float, str, str]] = []
    if ev_path.is_file():
        with ev_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("event_id") or not str(row["event_id"]).strip():
                    continue
                zid = str(
                    row.get("zone_id") or row.get("zone_or_line_id") or ""
                ).strip()
                snap = str(row.get("snapshot_path") or "").strip()
                crop = str(row.get("crop_path") or "").strip()
                event_rows.append(
                    (
                        str(row["event_id"]).strip(),
                        str(row["event_type"]).strip(),
                        int(row["track_id"]),
                        zid,
                        float(row["start_time"]),
                        float(row["end_time"]),
                        float(row["duration_seconds"]),
                        snap,
                        crop,
                    )
                )

    zc_rows: list[tuple[str, int, int, float, int]] = []
    if zc_path.is_file():
        with zc_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("zone_id") or not str(row["zone_id"]).strip():
                    continue
                zc_rows.append(
                    (
                        str(row["zone_id"]).strip(),
                        int(row["entry_count"]),
                        int(row["exit_count"]),
                        float(row["total_dwell_seconds"]),
                        int(row["peak_occupancy"]),
                    )
                )

    lc_rows: list[tuple[str, int, int, int]] = []
    if lc_path.is_file():
        with lc_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("line_id") or not str(row["line_id"]).strip():
                    continue
                lc_rows.append(
                    (
                        str(row["line_id"]).strip(),
                        int(row["crossings_ab"]),
                        int(row["crossings_ba"]),
                        int(row["total"]),
                    )
                )

    with closing(sqlite3.connect(out_db)) as conn:
        for tbl in (
            "detections",
            "track_observations",
            "tracks",
            "trajectories",
            "events",
            "zone_counts",
            "line_counts",
        ):
            conn.execute(f"DELETE FROM {tbl}")  # noqa: S608

        conn.executemany(
            "INSERT INTO detections (frame_id, x1, y1, x2, y2, confidence, class_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            det_rows,
        )
        conn.executemany(
            "INSERT INTO track_observations (frame_id, track_id, x1, y1, x2, y2, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            obs_rows,
        )
        conn.executemany(
            "INSERT INTO tracks (track_id, start_frame, end_frame, duration_frames, "
            "duration_seconds, observed_frames, path_length_px) VALUES (?, ?, ?, ?, ?, ?, ?)",
            track_rows,
        )
        conn.executemany(
            "INSERT INTO trajectories (track_id, frame_id, timestamp, x, y) "
            "VALUES (?, ?, ?, ?, ?)",
            traj_rows,
        )
        conn.executemany(
            "INSERT INTO events (event_id, event_type, track_id, zone_id, start_time, "
            "end_time, duration_seconds, snapshot_path, crop_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            event_rows,
        )
        conn.executemany(
            "INSERT INTO zone_counts (zone_id, entry_count, exit_count, "
            "total_dwell_seconds, peak_occupancy) VALUES (?, ?, ?, ?, ?)",
            zc_rows,
        )
        conn.executemany(
            "INSERT INTO line_counts (line_id, crossings_ab, crossings_ba, total) "
            "VALUES (?, ?, ?, ?)",
            lc_rows,
        )
        conn.commit()

    return DatabaseSummary(
        db_path=str(out_db),
        source_name=config.source_name,
        detections_count=len(det_rows),
        track_observations_count=len(obs_rows),
        tracks_count=len(track_rows),
        trajectories_count=len(traj_rows),
        events_count=len(event_rows),
        zone_counts_rows=len(zc_rows),
        line_counts_rows=len(lc_rows),
    )
