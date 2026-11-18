"""Multi-run comparison, spatial density diffing, and cross-run baseline analytics engine."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml

from event_intelligence.compare.types import RunSnapshot


def _resolve_run_and_db_paths(run_or_db_path: Path | str) -> tuple[Path, Path]:
    """Resolve either a run directory or direct intelligence.db file into (run_dir, db_path)."""
    p = Path(run_or_db_path).resolve()
    if p.is_dir():
        db_path = p / "intelligence.db"
        if not db_path.is_file():
            raise FileNotFoundError(
                f"No intelligence.db found inside run directory: {p}"
            )
        return p, db_path
    if p.is_file():
        return p.parent, p
    raise FileNotFoundError(f"Run directory or SQLite database not found: {p}")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_has_column(conn: sqlite3.Connection, table_name: str, col_name: str) -> bool:
    if not _table_exists(conn, table_name):
        return False
    cols = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    return col_name in cols


def load_run_snapshot(
    run_or_db_path: Path | str,
    *,
    run_id: str | None = None,
    camera_id: str | None = None,
) -> RunSnapshot:
    """Load KPI metrics and frozen run_config.yaml from a pipeline run directory or SQLite DB."""
    run_dir, db_path = _resolve_run_and_db_paths(run_or_db_path)
    resolved_run_id = (
        run_id.strip() if run_id and run_id.strip() else run_dir.name or db_path.stem
    )
    cam_filter = camera_id.strip() if camera_id and camera_id.strip() else None

    config_snapshot: dict[str, Any] = {}
    cfg_file = run_dir / "run_config.yaml"
    if cfg_file.is_file():
        try:
            loaded_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            if isinstance(loaded_cfg, dict):
                config_snapshot = loaded_cfg
        except (yaml.YAMLError, OSError):
            pass

    def _where(conn: sqlite3.Connection, table: str) -> tuple[str, tuple[Any, ...]]:
        if cam_filter and _table_has_column(conn, table, "camera_id"):
            return " WHERE camera_id = ?", (cam_filter,)
        return "", ()

    with closing(sqlite3.connect(db_path)) as conn:
        if _table_exists(conn, "cameras"):
            w_sql, w_params = _where(conn, "cameras")
            cameras_count = int(
                conn.execute(
                    f"SELECT count(*) FROM cameras{w_sql}", w_params
                ).fetchone()[0]
            )
        else:
            cameras_count = 1

        w_sql, w_params = _where(conn, "detections")
        detections_count = (
            int(
                conn.execute(
                    f"SELECT count(*) FROM detections{w_sql}", w_params
                ).fetchone()[0]
            )
            if _table_exists(conn, "detections")
            else 0
        )

        if _table_exists(conn, "tracks"):
            w_sql, w_params = _where(conn, "tracks")
            trk_row = conn.execute(
                f"SELECT count(*), coalesce(avg(duration_seconds), 0.0), "
                f"coalesce(avg(path_length_px), 0.0) FROM tracks{w_sql}",
                w_params,
            ).fetchone()
            tracks_count = int(trk_row[0])
            mean_track_dur = float(trk_row[1])
            mean_path_len = float(trk_row[2])
        else:
            tracks_count = 0
            mean_track_dur = 0.0
            mean_path_len = 0.0

        w_sql, w_params = _where(conn, "trajectories")
        trajectories_count = (
            int(
                conn.execute(
                    f"SELECT count(*) FROM trajectories{w_sql}", w_params
                ).fetchone()[0]
            )
            if _table_exists(conn, "trajectories")
            else 0
        )

        if _table_exists(conn, "events"):
            w_sql, w_params = _where(conn, "events")
            ev_row = conn.execute(
                f"SELECT count(*), coalesce(sum(CASE WHEN event_type IN ('dwell_time_event', 'zone_dwell') "
                f"THEN duration_seconds ELSE 0.0 END), 0.0) FROM events{w_sql}",
                w_params,
            ).fetchone()
            events_count = int(ev_row[0])
            total_dwell = float(ev_row[1])
        else:
            events_count = 0
            total_dwell = 0.0

        if _table_exists(conn, "insights"):
            w_sql, w_params = _where(conn, "insights")
            ins_row = conn.execute(
                f"SELECT count(*), coalesce(sum(CASE WHEN lower(severity) IN ('high', 'critical') "
                f"THEN 1 ELSE 0 END), 0) FROM insights{w_sql}",
                w_params,
            ).fetchone()
            insights_count = int(ins_row[0])
            high_crit_count = int(ins_row[1])
        else:
            insights_count = 0
            high_crit_count = 0

    return RunSnapshot(
        run_id=resolved_run_id,
        run_dir=str(run_dir),
        db_path=str(db_path),
        cameras_count=max(0, cameras_count),
        detections_count=detections_count,
        tracks_count=tracks_count,
        trajectories_count=trajectories_count,
        events_count=events_count,
        insights_count=insights_count,
        high_critical_insights_count=high_crit_count,
        total_dwell_seconds=round(total_dwell, 4),
        mean_track_duration_seconds=round(mean_track_dur, 4),
        mean_path_length_px=round(mean_path_len, 2),
        config_snapshot=config_snapshot,
    )
