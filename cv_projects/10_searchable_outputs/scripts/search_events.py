"""Phase 10 — Searchable Outputs: SQLite ingestion (intelligence.db), CLI and
programmatic queries for events, tracks, and visual evidence, and worked examples.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    crop_path TEXT NOT NULL,
    video_clip_path TEXT NOT NULL DEFAULT '',
    storyboard_path TEXT NOT NULL DEFAULT ''
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
    write_subproject_docs: bool = True

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
    video_clip_path: str = ""
    storyboard_path: str = ""

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
            "video_clip_path": self.video_clip_path,
            "storyboard_path": self.storyboard_path,
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
    resolved_video_clip_path: str = ""
    resolved_storyboard_path: str = ""
    video_clip_exists: bool = False
    storyboard_exists: bool = False
    video_clip_size_bytes: int = 0
    storyboard_size_bytes: int = 0

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
            "resolved_video_clip_path": self.resolved_video_clip_path,
            "resolved_storyboard_path": self.resolved_storyboard_path,
            "video_clip_exists": self.video_clip_exists,
            "storyboard_exists": self.storyboard_exists,
            "video_clip_size_bytes": self.video_clip_size_bytes,
            "storyboard_size_bytes": self.storyboard_size_bytes,
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
        write_subproject_docs=bool(
            raw.get("write_subproject_docs", defaults.write_subproject_docs)
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
        "write_subproject_docs": config.write_subproject_docs,
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
            conn.execute(f"DELETE FROM {tbl}")

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


def query_events(
    db_path: Path | str,
    *,
    event_type: str | None = None,
    zone_id: str | None = None,
    track_id: int | None = None,
    min_time: float | None = None,
    max_time: float | None = None,
    min_duration: float | None = None,
    limit: int = 50,
) -> list[EventSearchResult]:
    """Query events from intelligence.db by type, zone/line, track_id, time range, or duration."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    clauses: list[str] = []
    params: list[Any] = []

    if event_type is not None and event_type.strip():
        clauses.append("event_type = ?")
        params.append(event_type.strip())

    if zone_id is not None and zone_id.strip():
        z = zone_id.strip()
        clauses.append("(zone_id = ? OR zone_id LIKE ?)")
        params.extend([z, f"{z}:%"])

    if track_id is not None:
        clauses.append("track_id = ?")
        params.append(int(track_id))

    if min_time is not None:
        clauses.append("end_time >= ?")
        params.append(float(min_time))

    if max_time is not None:
        clauses.append("start_time <= ?")
        params.append(float(max_time))

    if min_duration is not None:
        clauses.append("duration_seconds >= ?")
        params.append(float(min_duration))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT event_id, event_type, track_id, zone_id, start_time, end_time, "
        "duration_seconds, snapshot_path, crop_path "
        f"FROM events {where_sql} "
        "ORDER BY start_time ASC, event_id ASC LIMIT ?"
    )
    params.append(int(limit))

    results: list[EventSearchResult] = []
    with closing(sqlite3.connect(path)) as conn:
        for row in conn.execute(sql, params).fetchall():
            results.append(
                EventSearchResult(
                    event_id=str(row[0]),
                    event_type=str(row[1]),
                    track_id=int(row[2]),
                    zone_id=str(row[3]),
                    start_time=float(row[4]),
                    end_time=float(row[5]),
                    duration_seconds=float(row[6]),
                    snapshot_path=str(row[7]),
                    crop_path=str(row[8]),
                )
            )
    return results


def query_tracks(
    db_path: Path | str,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
    min_path_length_px: float | None = None,
    track_id: int | None = None,
    limit: int = 50,
) -> list[TrackSearchResult]:
    """Query tracks from intelligence.db by duration range, path length, or track_id."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    clauses: list[str] = []
    params: list[Any] = []

    if min_duration is not None:
        clauses.append("t.duration_seconds >= ?")
        params.append(float(min_duration))

    if max_duration is not None:
        clauses.append("t.duration_seconds <= ?")
        params.append(float(max_duration))

    if min_path_length_px is not None:
        clauses.append("t.path_length_px >= ?")
        params.append(float(min_path_length_px))

    if track_id is not None:
        clauses.append("t.track_id = ?")
        params.append(int(track_id))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT t.track_id, t.start_frame, t.end_frame, t.duration_frames, "
        "t.duration_seconds, t.observed_frames, t.path_length_px, "
        "COUNT(e.event_id) AS event_count "
        "FROM tracks t "
        "LEFT JOIN events e ON e.track_id = t.track_id "
        f"{where_sql} "
        "GROUP BY t.track_id "
        "ORDER BY t.duration_seconds DESC, t.track_id ASC LIMIT ?"
    )
    params.append(int(limit))

    results: list[TrackSearchResult] = []
    with closing(sqlite3.connect(path)) as conn:
        for row in conn.execute(sql, params).fetchall():
            results.append(
                TrackSearchResult(
                    track_id=int(row[0]),
                    start_frame=int(row[1]),
                    end_frame=int(row[2]),
                    duration_frames=int(row[3]),
                    duration_seconds=float(row[4]),
                    observed_frames=int(row[5]),
                    path_length_px=float(row[6]),
                    event_count=int(row[7]),
                )
            )
    return results


def lookup_event_evidence(
    db_path: Path | str,
    event_id: str,
    *,
    evidence_root: Path | str | None = None,
) -> EvidenceLookupResult:
    """Look up an event by event_id and resolve/verify its snapshot_path and crop_path on disk."""
    if not event_id.strip():
        raise ValueError("event_id must be non-empty")
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            "SELECT event_id, event_type, track_id, zone_id, start_time, end_time, "
            "duration_seconds, snapshot_path, crop_path "
            "FROM events WHERE event_id = ?",
            (event_id.strip(),),
        ).fetchone()

    if row is None:
        raise KeyError(f"Event ID not found in intelligence.db: {event_id!r}")

    ev = EventSearchResult(
        event_id=str(row[0]),
        event_type=str(row[1]),
        track_id=int(row[2]),
        zone_id=str(row[3]),
        start_time=float(row[4]),
        end_time=float(row[5]),
        duration_seconds=float(row[6]),
        snapshot_path=str(row[7]),
        crop_path=str(row[8]),
    )

    root = (
        Path(evidence_root)
        if evidence_root is not None
        else (
            _PROJECT_ROOT
            / "cv_projects"
            / "07_visual_evidence"
            / "outputs"
            / "people_detection"
        )
    )
    snap_file = (
        Path(ev.snapshot_path)
        if Path(ev.snapshot_path).is_absolute()
        else (root / ev.snapshot_path)
    )
    crop_file = (
        Path(ev.crop_path)
        if Path(ev.crop_path).is_absolute()
        else (root / ev.crop_path)
    )

    snap_exists = snap_file.is_file()
    crop_exists = crop_file.is_file()
    snap_size = snap_file.stat().st_size if snap_exists else 0
    crop_size = crop_file.stat().st_size if crop_exists else 0

    return EvidenceLookupResult(
        event=ev,
        resolved_snapshot_path=str(snap_file),
        resolved_crop_path=str(crop_file),
        snapshot_exists=snap_exists,
        crop_exists=crop_exists,
        snapshot_size_bytes=snap_size,
        crop_size_bytes=crop_size,
    )


def generate_schema_markdown(summary: DatabaseSummary) -> str:
    """Render schema.md documenting all 7 SQLite tables, indexes, and row counts."""
    lines: list[str] = [
        "# Phase 10 SQLite Schema — `intelligence.db`",
        "",
        f"Source video: **`{summary.source_name}`**",
        "",
        "## 1. Table Row Count Summary",
        "",
        "| Table Name | Primary Key | Secondary Indexes | Row Count | Description |",
        "| --- | --- | --- | ---: | --- |",
        f"| `detections` | `detection_id` | `idx_detections_frame_id (frame_id)` | `{summary.detections_count}` | Phase 2 person detections (`x1, y1, x2, y2, confidence, class_name`) |",
        f"| `track_observations` | `observation_id` | `idx_track_obs_track_frame (track_id, frame_id)` | `{summary.track_observations_count}` | Phase 3 per-frame tracked bounding boxes |",
        f"| `tracks` | `track_id` | `idx_tracks_duration (duration_seconds DESC)` | `{summary.tracks_count}` | Per-identity lifespan and ground-plane `path_length_px` |",
        f"| `trajectories` | `(track_id, frame_id)` | `idx_trajectories_timestamp (timestamp)` | `{summary.trajectories_count}` | Phase 4 smoothed `bottom_center` foot-anchor coordinates |",
        f"| `events` | `event_id` | `idx_events_type_zone`, `idx_events_track_id`, `idx_events_time_range` | `{summary.events_count}` | Phase 6/7 structured events with `snapshot_path` and `crop_path` |",
        f"| `zone_counts` | `zone_id` | — | `{summary.zone_counts_rows}` | Phase 8 per-zone entries, exits, total dwell, and peak occupancy |",
        f"| `line_counts` | `line_id` | — | `{summary.line_counts_rows}` | Phase 8 per-line directional crossings (`A->B`, `B->A`, `total`) |",
        "",
        "---",
        "",
        "## 2. SQLite DDL (`SCHEMA_DDL`)",
        "",
        "```sql",
        SCHEMA_DDL.strip(),
        "```",
    ]
    return "\n".join(lines)


def generate_search_examples_markdown(
    db_path: Path | str,
    *,
    evidence_root: Path | str | None = None,
) -> str:
    """Execute 5 canonical CLI/SQL queries against intelligence.db and render search_examples.md."""
    q1 = query_events(db_path, event_type="dwell_time_event", zone_id="right_aisle")
    q2 = query_events(
        db_path,
        event_type="line_crossing",
        zone_id="doorway_line",
        min_time=15.0,
        max_time=20.0,
    )
    q3 = query_events(db_path, track_id=1)
    q4 = query_tracks(db_path, min_duration=5.0)

    all_events = query_events(db_path, limit=500)
    sample_event_id = (
        all_events[1].event_id
        if len(all_events) > 1
        else (all_events[0].event_id if all_events else "")
    )
    ev_lookup = (
        lookup_event_evidence(db_path, sample_event_id, evidence_root=evidence_root)
        if sample_event_id
        else None
    )

    lines: list[str] = [
        "# Phase 10 Worked Search Examples — `search_examples.md`",
        "",
        "Every query below was executed directly against `intelligence.db` via `scripts/search_events.py`.",
        "",
        "---",
        "",
        "## Query 1 — Dwell-Time Events in `right_aisle`",
        "",
        "```bash",
        "uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \\",
        "    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \\",
        "    --event-type dwell_time_event --zone right_aisle",
        "```",
        "",
        f"**Returned `{len(q1)}` events:**",
        "",
        "| `event_id` | `event_type` | `track_id` | `zone_id` | `start_time` | `end_time` | `duration_seconds` | `snapshot_path` | `crop_path` |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for e in q1:
        lines.append(
            f"| `{e.event_id}` | `{e.event_type}` | `{e.track_id}` | `{e.zone_id}` | "
            f"`{e.start_time:.4f}` | `{e.end_time:.4f}` | `{e.duration_seconds:.4f}` | "
            f"`{e.snapshot_path}` | `{e.crop_path}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Query 2 — `doorway_line` Crossings Between `15.0s` and `20.0s`",
            "",
            "```bash",
            "uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \\",
            "    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \\",
            "    --event-type line_crossing --zone doorway_line --min-time 15.0 --max-time 20.0",
            "```",
            "",
            f"**Returned `{len(q2)}` events:**",
            "",
            "| `event_id` | `event_type` | `track_id` | `zone_id` | `start_time` | `snapshot_path` |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for e in q2:
        lines.append(
            f"| `{e.event_id}` | `{e.event_type}` | `{e.track_id}` | `{e.zone_id}` | "
            f"`{e.start_time:.4f}` | `{e.snapshot_path}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Query 3 — Complete Event Timeline for `track_id = 1`",
            "",
            "```bash",
            "uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \\",
            "    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \\",
            "    --track-id 1",
            "```",
            "",
            f"**Returned `{len(q3)}` events:**",
            "",
            "| `event_id` | `event_type` | `zone_id` | `start_time` | `end_time` | `duration_seconds` |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for e in q3:
        lines.append(
            f"| `{e.event_id}` | `{e.event_type}` | `{e.zone_id}` | "
            f"`{e.start_time:.4f}` | `{e.end_time:.4f}` | `{e.duration_seconds:.4f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Query 4 — Tracks with Duration `>= 5.0` Seconds",
            "",
            "```bash",
            "uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \\",
            "    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \\",
            "    --tracks --min-duration 5.0",
            "```",
            "",
            f"**Returned `{len(q4)}` tracks:**",
            "",
            "| `track_id` | `start_frame` | `end_frame` | `duration_frames` | `duration_seconds` | `path_length_px` | `event_count` |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for t in q4:
        lines.append(
            f"| `{t.track_id}` | `{t.start_frame}` | `{t.end_frame}` | `{t.duration_frames}` | "
            f"`{t.duration_seconds:.4f}` | `{t.path_length_px:.2f}` | `{t.event_count}` |"
        )

    if ev_lookup is not None:
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Query 5 — Visual Evidence Lookup for `{ev_lookup.event.event_id}`",
                "",
                "```bash",
                "uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \\",
                "    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \\",
                f"    --evidence-event-id {ev_lookup.event.event_id}",
                "```",
                "",
                (
                    f"- **Event:** `{ev_lookup.event.event_id}` (`{ev_lookup.event.event_type}`, "
                    f"`track_id={ev_lookup.event.track_id}`, `zone_id={ev_lookup.event.zone_id}`, "
                    f"`duration={ev_lookup.event.duration_seconds:.4f}s`)"
                ),
                (
                    f"- **Relative Snapshot Path:** `{ev_lookup.event.snapshot_path}` "
                    f"(`exists={ev_lookup.snapshot_exists}`, `{ev_lookup.snapshot_size_bytes}` bytes)"
                ),
                (
                    f"- **Relative Crop Path:** `{ev_lookup.event.crop_path}` "
                    f"(`exists={ev_lookup.crop_exists}`, `{ev_lookup.crop_size_bytes}` bytes)"
                ),
                f"- **Evidence Valid:** `{ev_lookup.is_valid}`",
            ]
        )

    return "\n".join(lines)


def run_searchable_outputs(
    config: SearchConfig,
    output_dir: Path | str,
    *,
    project_root: Path | None = None,
) -> DatabaseSummary:
    """Ingest upstream CSVs into intelligence.db and write schema, examples, and JSON summary."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = project_root if project_root is not None else _PROJECT_ROOT

    db_path = out_dir / "intelligence.db"
    summary = ingest_intelligence_db(db_path, config, project_root=root)

    ev_root = _resolve_optional_path(config.evidence_root, root)
    schema_md = generate_schema_markdown(summary)
    examples_md = generate_search_examples_markdown(db_path, evidence_root=ev_root)

    (out_dir / "schema.md").write_text(schema_md + "\n", encoding="utf-8")
    (out_dir / "search_examples.md").write_text(examples_md + "\n", encoding="utf-8")
    save_search_config(config, out_dir / "search.yaml")

    if config.write_subproject_docs:
        subproject_dir = root / "cv_projects" / "10_searchable_outputs"
        if subproject_dir.is_dir():
            (subproject_dir / "schema.md").write_text(
                schema_md + "\n", encoding="utf-8"
            )
            (subproject_dir / "search_examples.md").write_text(
                examples_md + "\n",
                encoding="utf-8",
            )

    all_events = query_events(db_path, limit=5000)
    valid_evidence_count = 0
    for ev in all_events:
        lookup = lookup_event_evidence(db_path, ev.event_id, evidence_root=ev_root)
        if lookup.is_valid:
            valid_evidence_count += 1

    summary_payload = {
        **summary.to_dict(),
        "valid_evidence_events": valid_evidence_count,
        "all_evidence_valid": valid_evidence_count == len(all_events),
    }
    (out_dir / "search_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for Phase 10 searchable outputs."""
    parser = argparse.ArgumentParser(
        description=(
            "Ingest upstream CV artifacts into SQLite (intelligence.db) and query "
            "events, tracks, and visual evidence."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path for --build ingestion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cv_projects/10_searchable_outputs/outputs/people_detection"),
        help="Output directory for intelligence.db, schema.md, and search_examples.md.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to an existing intelligence.db (defaults to <output-dir>/intelligence.db).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Ingest upstream CSVs into intelligence.db and generate schema/examples docs.",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        default=None,
        help="Filter events by event_type (e.g. zone_entry, dwell_time_event, line_crossing).",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Filter events by zone_id (exact or prefix match for directed lines).",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="Filter events by integer track_id.",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        help="Minimum timestamp in seconds for overlapping events.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Maximum timestamp in seconds for overlapping events.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum duration in seconds for events or tracks.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum duration in seconds for tracks.",
    )
    parser.add_argument(
        "--min-path-length",
        type=float,
        default=None,
        help="Minimum ground-plane path length in pixels for tracks.",
    )
    parser.add_argument(
        "--tracks",
        action="store_true",
        help="Query tracks table instead of events table.",
    )
    parser.add_argument(
        "--evidence-event-id",
        type=str,
        default=None,
        help="Look up visual evidence (snapshot_path and crop_path) for an event_id.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Optional root directory to resolve relative snapshot_path and crop_path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of rows to return (default: 100).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Format query results as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 10 searchable outputs."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = (
        load_search_config(args.config) if args.config is not None else SearchConfig()
    )
    if args.build:
        summary = run_searchable_outputs(config, args.output_dir)
        if args.json and not (
            args.tracks
            or args.evidence_event_id
            or args.event_type
            or args.zone
            or args.track_id is not None
            or args.min_time is not None
            or args.max_time is not None
            or args.min_duration is not None
        ):
            print(json.dumps(summary.to_dict(), indent=2))
            return 0
        if not (
            args.tracks
            or args.evidence_event_id
            or args.event_type
            or args.zone
            or args.track_id is not None
            or args.min_time is not None
            or args.max_time is not None
            or args.min_duration is not None
        ):
            print(
                f"Ingested intelligence.db ({summary.source_name}): "
                f"detections={summary.detections_count}, "
                f"track_obs={summary.track_observations_count}, "
                f"tracks={summary.tracks_count}, "
                f"trajectories={summary.trajectories_count}, "
                f"events={summary.events_count}, "
                f"zones={summary.zone_counts_rows}, "
                f"lines={summary.line_counts_rows}"
            )
            return 0

    db_path = args.db if args.db is not None else (args.output_dir / "intelligence.db")

    if args.evidence_event_id is not None:
        ev_root = (
            args.evidence_root
            if args.evidence_root is not None
            else _resolve_optional_path(config.evidence_root, _PROJECT_ROOT)
        )
        result = lookup_event_evidence(
            db_path,
            args.evidence_event_id,
            evidence_root=ev_root,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(
                f"Event {result.event.event_id} ({result.event.event_type}, "
                f"track={result.event.track_id}, zone={result.event.zone_id}): "
                f"snapshot={result.event.snapshot_path} (exists={result.snapshot_exists}, "
                f"{result.snapshot_size_bytes}B), "
                f"crop={result.event.crop_path} (exists={result.crop_exists}, "
                f"{result.crop_size_bytes}B)"
            )
        return 0

    if args.tracks:
        tracks = query_tracks(
            db_path,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_path_length_px=args.min_path_length,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps([t.to_dict() for t in tracks], indent=2))
        else:
            print(f"Matched {len(tracks)} track(s):")
            for t in tracks:
                print(
                    f"  track_id={t.track_id} frames={t.start_frame}..{t.end_frame} "
                    f"duration={t.duration_seconds:.4f}s "
                    f"path_length={t.path_length_px:.2f}px events={t.event_count}"
                )
        return 0

    events = query_events(
        db_path,
        event_type=args.event_type,
        zone_id=args.zone,
        track_id=args.track_id,
        min_time=args.min_time,
        max_time=args.max_time,
        min_duration=args.min_duration,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps([e.to_dict() for e in events], indent=2))
    else:
        print(f"Matched {len(events)} event(s):")
        for e in events:
            print(
                f"  {e.event_id} [{e.event_type}] track={e.track_id} zone={e.zone_id} "
                f"time={e.start_time:.4f}..{e.end_time:.4f}s "
                f"(dur={e.duration_seconds:.4f}s) snap={e.snapshot_path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
