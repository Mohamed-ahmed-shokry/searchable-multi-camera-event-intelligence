"""Multi-run comparison, spatial density diffing, and cross-run baseline analytics engine."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import yaml

from event_intelligence.compare.types import (
    ConfigDiffEntry,
    InsightTypeDelta,
    LineMetricDelta,
    RunSnapshot,
    TemporalBucketDelta,
    ZoneMetricDelta,
)
from event_intelligence.storage.search_events import (
    query_insights,
    query_line_counts,
    query_temporal_activity,
    query_zone_counts,
)


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


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML dict keys using dot notation for readable config diffs."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        full_key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            out.update(_flatten_mapping(v, prefix=full_key))
        else:
            out[full_key] = dict(v) if isinstance(v, Mapping) else v
    return out


def diff_run_configs(
    baseline_cfg: Mapping[str, Any],
    candidate_cfg: Mapping[str, Any],
) -> list[ConfigDiffEntry]:
    """Return sorted ConfigDiffEntry rows for every parameter differing between two configs."""
    flat_base = _flatten_mapping(baseline_cfg)
    flat_cand = _flatten_mapping(candidate_cfg)
    all_keys = sorted(set(flat_base.keys()) | set(flat_cand.keys()))
    diffs: list[ConfigDiffEntry] = []
    for key in all_keys:
        b_val = flat_base.get(key)
        c_val = flat_cand.get(key)
        if b_val != c_val:
            diffs.append(
                ConfigDiffEntry(
                    key=key,
                    baseline_value=b_val,
                    candidate_value=c_val,
                )
            )
    return diffs


def compute_zone_deltas(
    baseline_db: Path | str,
    candidate_db: Path | str,
    *,
    camera_id: str | None = None,
) -> list[ZoneMetricDelta]:
    """Compute signed per-zone entry, exit, dwell, and peak occupancy deltas between two DBs."""
    _, b_db = _resolve_run_and_db_paths(baseline_db)
    _, c_db = _resolve_run_and_db_paths(candidate_db)

    b_rows = query_zone_counts(b_db, camera_id=camera_id, limit=1000)
    c_rows = query_zone_counts(c_db, camera_id=camera_id, limit=1000)

    b_cams = {r.camera_id for r in b_rows}
    c_cams = {r.camera_id for r in c_rows}
    align_by_zone_only = (
        camera_id is None and len(b_cams) == 1 and len(c_cams) == 1 and b_cams != c_cams
    )

    b_map = {("" if align_by_zone_only else r.camera_id, r.zone_id): r for r in b_rows}
    c_map = {("" if align_by_zone_only else r.camera_id, r.zone_id): r for r in c_rows}

    deltas: list[ZoneMetricDelta] = []
    for cam, zid in sorted(set(b_map.keys()) | set(c_map.keys())):
        br = b_map.get((cam, zid))
        cr = c_map.get((cam, zid))
        deltas.append(
            ZoneMetricDelta(
                camera_id=cam,
                zone_id=zid,
                baseline_entries=br.entry_count if br else 0,
                candidate_entries=cr.entry_count if cr else 0,
                baseline_exits=br.exit_count if br else 0,
                candidate_exits=cr.exit_count if cr else 0,
                baseline_dwell_seconds=br.total_dwell_seconds if br else 0.0,
                candidate_dwell_seconds=cr.total_dwell_seconds if cr else 0.0,
                baseline_peak_occupancy=br.peak_occupancy if br else 0,
                candidate_peak_occupancy=cr.peak_occupancy if cr else 0,
            )
        )
    return deltas


def compute_line_deltas(
    baseline_db: Path | str,
    candidate_db: Path | str,
    *,
    camera_id: str | None = None,
) -> list[LineMetricDelta]:
    """Compute signed per-line directional crossing deltas between two DBs."""
    _, b_db = _resolve_run_and_db_paths(baseline_db)
    _, c_db = _resolve_run_and_db_paths(candidate_db)

    b_rows = query_line_counts(b_db, camera_id=camera_id, limit=1000)
    c_rows = query_line_counts(c_db, camera_id=camera_id, limit=1000)

    b_cams = {r.camera_id for r in b_rows}
    c_cams = {r.camera_id for r in c_rows}
    align_by_line_only = (
        camera_id is None and len(b_cams) == 1 and len(c_cams) == 1 and b_cams != c_cams
    )

    b_map = {("" if align_by_line_only else r.camera_id, r.line_id): r for r in b_rows}
    c_map = {("" if align_by_line_only else r.camera_id, r.line_id): r for r in c_rows}

    deltas: list[LineMetricDelta] = []
    for cam, lid in sorted(set(b_map.keys()) | set(c_map.keys())):
        br = b_map.get((cam, lid))
        cr = c_map.get((cam, lid))
        deltas.append(
            LineMetricDelta(
                camera_id=cam,
                line_id=lid,
                baseline_ab=br.crossings_ab if br else 0,
                candidate_ab=cr.crossings_ab if cr else 0,
                baseline_ba=br.crossings_ba if br else 0,
                candidate_ba=cr.crossings_ba if cr else 0,
                baseline_total=br.total if br else 0,
                candidate_total=cr.total if cr else 0,
            )
        )
    return deltas


def compute_insight_deltas(
    baseline_db: Path | str,
    candidate_db: Path | str,
    *,
    camera_id: str | None = None,
) -> list[InsightTypeDelta]:
    """Compute signed per-insight-type count and high/critical severity deltas between two DBs."""
    _, b_db = _resolve_run_and_db_paths(baseline_db)
    _, c_db = _resolve_run_and_db_paths(candidate_db)

    def _safe_insights(db: Path) -> list[Any]:
        with closing(sqlite3.connect(db)) as conn:
            if not _table_exists(conn, "insights"):
                return []
        return query_insights(db, camera_id=camera_id, limit=10000)

    b_ins = _safe_insights(b_db)
    c_ins = _safe_insights(c_db)

    b_counts: dict[str, int] = {}
    b_hc: dict[str, int] = {}
    for item in b_ins:
        b_counts[item.insight_type] = b_counts.get(item.insight_type, 0) + 1
        if item.severity.lower() in {"high", "critical"}:
            b_hc[item.insight_type] = b_hc.get(item.insight_type, 0) + 1

    c_counts: dict[str, int] = {}
    c_hc: dict[str, int] = {}
    for item in c_ins:
        c_counts[item.insight_type] = c_counts.get(item.insight_type, 0) + 1
        if item.severity.lower() in {"high", "critical"}:
            c_hc[item.insight_type] = c_hc.get(item.insight_type, 0) + 1

    all_types = sorted(set(b_counts.keys()) | set(c_counts.keys()))
    return [
        InsightTypeDelta(
            insight_type=itype,
            baseline_count=b_counts.get(itype, 0),
            candidate_count=c_counts.get(itype, 0),
            baseline_high_critical=b_hc.get(itype, 0),
            candidate_high_critical=c_hc.get(itype, 0),
        )
        for itype in all_types
    ]


def compute_temporal_bucket_deltas(
    baseline_db: Path | str,
    candidate_db: Path | str,
    *,
    bucket_seconds: float = 5.0,
    camera_id: str | None = None,
) -> list[TemporalBucketDelta]:
    """Compare time-windowed activity histograms between two runs bucket-by-bucket."""
    if bucket_seconds <= 0.0:
        raise ValueError(f"bucket_seconds must be > 0.0, got {bucket_seconds}")
    _, b_db = _resolve_run_and_db_paths(baseline_db)
    _, c_db = _resolve_run_and_db_paths(candidate_db)

    b_buckets = query_temporal_activity(
        b_db, bucket_seconds=bucket_seconds, camera_id=camera_id
    )
    c_buckets = query_temporal_activity(
        c_db, bucket_seconds=bucket_seconds, camera_id=camera_id
    )

    b_by_idx = {b.bucket_index: b for b in b_buckets}
    c_by_idx = {b.bucket_index: b for b in c_buckets}
    all_idxs = sorted(set(b_by_idx.keys()) | set(c_by_idx.keys()))

    deltas: list[TemporalBucketDelta] = []
    for idx in all_idxs:
        bb = b_by_idx.get(idx)
        cb = c_by_idx.get(idx)
        start_t = round(idx * bucket_seconds, 4)
        end_t = round((idx + 1) * bucket_seconds, 4)
        deltas.append(
            TemporalBucketDelta(
                bucket_index=idx,
                start_time=start_t,
                end_time=end_t,
                baseline_active_tracks=bb.active_tracks_count if bb else 0,
                candidate_active_tracks=cb.active_tracks_count if cb else 0,
                baseline_events=bb.events_count if bb else 0,
                candidate_events=cb.events_count if cb else 0,
                baseline_insights=bb.insights_count if bb else 0,
                candidate_insights=cb.insights_count if cb else 0,
            )
        )
    return deltas
