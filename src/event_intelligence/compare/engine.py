"""Multi-run comparison, spatial density diffing, and cross-run baseline analytics engine."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from event_intelligence.analytics import (
    accumulate_trajectory_grid,
    smooth_and_normalize_heatmap,
)
from event_intelligence.compare.report import write_comparison_artifacts
from event_intelligence.compare.types import (
    ConfigDiffEntry,
    InsightTypeDelta,
    LineMetricDelta,
    MultiRunTrendReport,
    RegressionGateResult,
    RegressionPolicy,
    RegressionViolation,
    RunBaselineAnomaly,
    RunComparisonReport,
    RunSnapshot,
    SpatialShiftSummary,
    TemporalBucketDelta,
    ZoneMetricDelta,
)
from event_intelligence.storage.search_events import (
    query_insights,
    query_line_counts,
    query_temporal_activity,
    query_trajectories,
    query_zone_counts,
)
from event_intelligence.trajectories import TrajectoryPoint


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


def _infer_frame_dimensions(
    db_paths: list[Path],
    trajectories: list[Any],
    *,
    camera_id: str | None = None,
) -> tuple[int, int]:
    """Infer (width, height) from video_info.json, cameras table, or trajectory bounds."""
    for db_p in db_paths:
        vinfo_candidates = [db_p.parent / "video_info.json"]
        if camera_id and camera_id.strip():
            vinfo_candidates.insert(
                0, db_p.parent / "cameras" / camera_id.strip() / "video_info.json"
            )
        for vinfo_path in vinfo_candidates:
            if vinfo_path.is_file():
                try:
                    raw = json.loads(vinfo_path.read_text(encoding="utf-8"))
                    entry = raw[0] if isinstance(raw, list) and raw else raw
                    if isinstance(entry, dict):
                        vw = int(entry.get("width", 0))
                        vh = int(entry.get("height", 0))
                        if vw > 0 and vh > 0:
                            return vw, vh
                except (json.JSONDecodeError, ValueError, TypeError, OSError):
                    pass

    for db_p in db_paths:
        with closing(sqlite3.connect(db_p)) as conn:
            if not _table_exists(conn, "cameras"):
                continue
            if camera_id and camera_id.strip():
                row = conn.execute(
                    "SELECT width, height FROM cameras WHERE camera_id = ?",
                    (camera_id.strip(),),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT width, height FROM cameras ORDER BY camera_id LIMIT 1"
                ).fetchone()
            if row and int(row[0]) > 0 and int(row[1]) > 0:
                return int(row[0]), int(row[1])

    max_x = max((float(p.x) for p in trajectories), default=0.0)
    max_y = max((float(p.y) for p in trajectories), default=0.0)
    return max(64, int(max_x) + 16), max(64, int(max_y) + 16)


def _render_diverging_diff_image(
    delta: np.ndarray,
    *,
    reference_frame: np.ndarray | None = None,
    hotspot_increase_xy: tuple[float, float] | None = None,
    hotspot_decrease_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a diverging blue (decreased) vs red (increased) spatial density diff image."""
    height, width = int(delta.shape[0]), int(delta.shape[1])
    if reference_frame is not None and reference_frame.ndim == 3:
        base = cv2.resize(
            reference_frame, (width, height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
    else:
        base = np.full((height, width, 3), (28.0, 28.0, 32.0), dtype=np.float32)

    pos = np.clip(delta, 0.0, 1.0)
    neg = np.clip(-delta, 0.0, 1.0)

    overlay = base.copy()
    pos_w = np.where(pos > 0.02, 0.72 * (pos**0.7), 0.0)[..., np.newaxis]
    neg_w = np.where(neg > 0.02, 0.72 * (neg**0.7), 0.0)[..., np.newaxis]

    red_layer = np.full_like(base, (40.0, 45.0, 255.0))
    blue_layer = np.full_like(base, (255.0, 140.0, 35.0))

    overlay = overlay * (1.0 - pos_w) + red_layer * pos_w
    overlay = overlay * (1.0 - neg_w) + blue_layer * neg_w
    img = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    if hotspot_increase_xy is not None:
        ix, iy = round(hotspot_increase_xy[0]), round(hotspot_increase_xy[1])
        cv2.circle(img, (ix, iy), 8, (60, 60, 255), 2, cv2.LINE_AA)
    if hotspot_decrease_xy is not None:
        dx, dy = round(hotspot_decrease_xy[0]), round(hotspot_decrease_xy[1])
        cv2.circle(img, (dx, dy), 8, (255, 160, 50), 2, cv2.LINE_AA)

    if width >= 180 and height >= 60:
        hud_w = min(width - 16, 330)
        cv2.rectangle(img, (8, 8), (8 + hud_w, 44), (18, 18, 22), -1)
        cv2.rectangle(img, (8, 8), (8 + hud_w, 44), (110, 110, 120), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            "Spatial Diff: Blue=Decrease | Red=Increase",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )
    return img


def compute_spatial_density_diff(
    baseline_db: Path | str,
    candidate_db: Path | str,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    grid_bin_size_px: int = 16,
    gaussian_sigma_px: float = 24.0,
    camera_id: str | None = None,
    output_image_path: Path | str | None = None,
    reference_frame: np.ndarray | None = None,
) -> SpatialShiftSummary:
    """Compute normalized 2D spatial trajectory density difference (candidate - baseline)."""
    _, b_db = _resolve_run_and_db_paths(baseline_db)
    _, c_db = _resolve_run_and_db_paths(candidate_db)

    b_rows = query_trajectories(b_db, camera_id=camera_id, limit=200000)
    c_rows = query_trajectories(c_db, camera_id=camera_id, limit=200000)

    if frame_width is not None and frame_height is not None:
        w, h = int(frame_width), int(frame_height)
    else:
        w, h = _infer_frame_dimensions(
            [c_db, b_db], [*b_rows, *c_rows], camera_id=camera_id
        )

    b_pts = [
        TrajectoryPoint(r.track_id, r.frame_id, r.timestamp, r.x, r.y) for r in b_rows
    ]
    c_pts = [
        TrajectoryPoint(r.track_id, r.frame_id, r.timestamp, r.x, r.y) for r in c_rows
    ]

    b_grid, b_full = accumulate_trajectory_grid(
        b_pts, frame_width=w, frame_height=h, grid_bin_size_px=grid_bin_size_px
    )
    _, c_full = accumulate_trajectory_grid(
        c_pts, frame_width=w, frame_height=h, grid_bin_size_px=grid_bin_size_px
    )

    b_norm = smooth_and_normalize_heatmap(b_full, gaussian_sigma_px=gaussian_sigma_px)
    c_norm = smooth_and_normalize_heatmap(c_full, gaussian_sigma_px=gaussian_sigma_px)
    delta = (c_norm - b_norm).astype(np.float32)

    mean_abs = float(np.mean(np.abs(delta))) if delta.size > 0 else 0.0
    max_inc = float(delta.max()) if delta.size > 0 else 0.0
    max_dec = float(delta.min()) if delta.size > 0 else 0.0

    inc_xy: tuple[float, float] | None = None
    if max_inc > 1e-6:
        iy, ix = np.unravel_index(int(np.argmax(delta)), delta.shape)
        inc_xy = (float(ix), float(iy))

    dec_xy: tuple[float, float] | None = None
    if max_dec < -1e-6:
        dy, dx = np.unravel_index(int(np.argmin(delta)), delta.shape)
        dec_xy = (float(dx), float(dy))

    diff_img_str = ""
    if output_image_path is not None:
        out_img = Path(output_image_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        rendered = _render_diverging_diff_image(
            delta,
            reference_frame=reference_frame,
            hotspot_increase_xy=inc_xy,
            hotspot_decrease_xy=dec_xy,
        )
        cv2.imwrite(str(out_img), rendered)
        diff_img_str = str(out_img)

    return SpatialShiftSummary(
        grid_shape=(int(b_grid.shape[0]), int(b_grid.shape[1])),
        frame_width=w,
        frame_height=h,
        mean_abs_delta=mean_abs,
        max_increase=max_inc,
        max_decrease=max_dec,
        hotspot_increase_xy=inc_xy,
        hotspot_decrease_xy=dec_xy,
        diff_image_path=diff_img_str,
    )


_TREND_METRICS: tuple[str, ...] = (
    "detections_count",
    "tracks_count",
    "events_count",
    "insights_count",
    "high_critical_insights_count",
    "total_dwell_seconds",
)


def discover_runs(runs_root: Path | str) -> list[Path]:
    """Discover all subdirectories under runs_root containing an intelligence.db file."""
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Runs root directory not found: {root}")
    found = [
        child
        for child in sorted(root.iterdir(), key=lambda p: p.name)
        if child.is_dir() and (child / "intelligence.db").is_file()
    ]
    return found


def analyze_multi_run_trends(
    run_paths: Sequence[Path | str],
    *,
    camera_id: str | None = None,
    z_score_threshold: float = 1.5,
) -> MultiRunTrendReport:
    """Compute cross-run KPI baseline statistics and flag statistical z-score anomalies."""
    if z_score_threshold <= 0.0:
        raise ValueError(f"z_score_threshold must be > 0.0, got {z_score_threshold}")

    snapshots = [load_run_snapshot(rp, camera_id=camera_id) for rp in run_paths]
    if not snapshots:
        return MultiRunTrendReport(
            runs=[],
            z_score_threshold=z_score_threshold,
            metric_baselines={},
            anomalies=[],
            camera_filter=camera_id,
        )

    metric_baselines: dict[str, dict[str, float]] = {}
    anomalies: list[RunBaselineAnomaly] = []

    for metric_name in _TREND_METRICS:
        values = [float(getattr(s, metric_name)) for s in snapshots]
        mean_v = statistics.mean(values)
        std_v = statistics.pstdev(values) if len(values) >= 2 else 0.0
        metric_baselines[metric_name] = {
            "mean": round(mean_v, 4),
            "std": round(std_v, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
        if len(values) >= 2 and std_v > 1e-9:
            for snap, val in zip(snapshots, values, strict=True):
                z = (val - mean_v) / std_v
                if abs(z) >= z_score_threshold:
                    anomalies.append(
                        RunBaselineAnomaly(
                            run_id=snap.run_id,
                            metric_name=metric_name,
                            value=val,
                            baseline_mean=mean_v,
                            baseline_std=std_v,
                            z_score=z,
                            direction="above" if z > 0 else "below",
                        )
                    )

    anomalies.sort(key=lambda a: (-abs(a.z_score), a.run_id, a.metric_name))
    return MultiRunTrendReport(
        runs=snapshots,
        z_score_threshold=z_score_threshold,
        metric_baselines=metric_baselines,
        anomalies=anomalies,
        camera_filter=camera_id,
    )


def compare_runs(
    baseline_path: Path | str,
    candidate_path: Path | str,
    *,
    camera_id: str | None = None,
    bucket_seconds: float = 5.0,
    output_dir: Path | str | None = None,
    render_heatmap_diff: bool = True,
    grid_bin_size_px: int = 16,
    gaussian_sigma_px: float = 24.0,
) -> RunComparisonReport:
    """Orchestrate a complete pairwise comparison between a baseline run and candidate run."""
    base_snap = load_run_snapshot(baseline_path, camera_id=camera_id)
    cand_snap = load_run_snapshot(candidate_path, camera_id=camera_id)

    config_diffs = diff_run_configs(
        base_snap.config_snapshot, cand_snap.config_snapshot
    )
    zone_deltas = compute_zone_deltas(
        base_snap.db_path, cand_snap.db_path, camera_id=camera_id
    )
    line_deltas = compute_line_deltas(
        base_snap.db_path, cand_snap.db_path, camera_id=camera_id
    )
    insight_deltas = compute_insight_deltas(
        base_snap.db_path, cand_snap.db_path, camera_id=camera_id
    )
    temporal_deltas = compute_temporal_bucket_deltas(
        base_snap.db_path,
        cand_snap.db_path,
        bucket_seconds=bucket_seconds,
        camera_id=camera_id,
    )

    out_path = Path(output_dir).resolve() if output_dir is not None else None
    diff_img_path = (
        (out_path / "heatmap_diff.png")
        if (out_path is not None and render_heatmap_diff)
        else None
    )
    spatial_shift = compute_spatial_density_diff(
        base_snap.db_path,
        cand_snap.db_path,
        grid_bin_size_px=grid_bin_size_px,
        gaussian_sigma_px=gaussian_sigma_px,
        camera_id=camera_id,
        output_image_path=diff_img_path,
    )

    report = RunComparisonReport(
        baseline=base_snap,
        candidate=cand_snap,
        camera_filter=camera_id,
        config_diffs=config_diffs,
        zone_deltas=zone_deltas,
        line_deltas=line_deltas,
        insight_deltas=insight_deltas,
        temporal_deltas=temporal_deltas,
        spatial_shift=spatial_shift,
    )
    if out_path is not None:
        write_comparison_artifacts(report, out_path)
    return report


def load_regression_policy(path: Path | str | None = None) -> RegressionPolicy:
    """Load a RegressionPolicy from a YAML file, or return defaults if None."""
    if path is None:
        return RegressionPolicy()
    policy_path = Path(path)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Regression policy file not found: {policy_path}")
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if raw is None:
        return RegressionPolicy()
    if not isinstance(raw, dict):
        raise TypeError("Regression policy YAML root must be a mapping")

    defaults = RegressionPolicy()

    def _opt_float(key: str, default: float | None) -> float | None:
        if key not in raw:
            return default
        val = raw[key]
        return float(val) if val is not None else None

    def _opt_int(key: str, default: int | None) -> int | None:
        if key not in raw:
            return default
        val = raw[key]
        return int(val) if val is not None else None

    return RegressionPolicy(
        max_tracks_drop_pct=_opt_float(
            "max_tracks_drop_pct", defaults.max_tracks_drop_pct
        ),
        max_events_drop_pct=_opt_float(
            "max_events_drop_pct", defaults.max_events_drop_pct
        ),
        max_dwell_drop_pct=_opt_float(
            "max_dwell_drop_pct", defaults.max_dwell_drop_pct
        ),
        max_high_critical_insights_increase=_opt_int(
            "max_high_critical_insights_increase",
            defaults.max_high_critical_insights_increase,
        ),
        max_spatial_shift_mean_abs=_opt_float(
            "max_spatial_shift_mean_abs", defaults.max_spatial_shift_mean_abs
        ),
        min_detections_ratio=_opt_float(
            "min_detections_ratio", defaults.min_detections_ratio
        ),
        allow_config_drift=bool(
            raw.get("allow_config_drift", defaults.allow_config_drift)
        ),
    )


def save_regression_policy(policy: RegressionPolicy, path: Path | str) -> Path:
    """Serialize a RegressionPolicy to YAML."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(policy.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def evaluate_regression_gate(
    report: RunComparisonReport,
    policy: RegressionPolicy | None = None,
) -> RegressionGateResult:
    """Evaluate a RunComparisonReport against configurable CV regression gate rules."""
    pol = policy if policy is not None else RegressionPolicy()
    base = report.baseline
    cand = report.candidate
    violations: list[RegressionViolation] = []
    checks = 0

    if pol.max_tracks_drop_pct is not None:
        checks += 1
        if base.tracks_count > 0:
            drop_pct = (
                (base.tracks_count - cand.tracks_count) / base.tracks_count
            ) * 100.0
            if drop_pct > pol.max_tracks_drop_pct:
                violations.append(
                    RegressionViolation(
                        rule_name="max_tracks_drop_pct",
                        severity="error",
                        metric_name="tracks_count",
                        baseline_value=float(base.tracks_count),
                        candidate_value=float(cand.tracks_count),
                        threshold=pol.max_tracks_drop_pct,
                        message=(
                            f"Track count dropped by {drop_pct:.1f}% "
                            f"({base.tracks_count} -> {cand.tracks_count}, "
                            f"max allowed {pol.max_tracks_drop_pct:.1f}%)"
                        ),
                    )
                )

    if pol.max_events_drop_pct is not None:
        checks += 1
        if base.events_count > 0:
            drop_pct = (
                (base.events_count - cand.events_count) / base.events_count
            ) * 100.0
            if drop_pct > pol.max_events_drop_pct:
                violations.append(
                    RegressionViolation(
                        rule_name="max_events_drop_pct",
                        severity="error",
                        metric_name="events_count",
                        baseline_value=float(base.events_count),
                        candidate_value=float(cand.events_count),
                        threshold=pol.max_events_drop_pct,
                        message=(
                            f"Event count dropped by {drop_pct:.1f}% "
                            f"({base.events_count} -> {cand.events_count}, "
                            f"max allowed {pol.max_events_drop_pct:.1f}%)"
                        ),
                    )
                )

    if pol.max_dwell_drop_pct is not None:
        checks += 1
        if base.total_dwell_seconds > 0.0:
            drop_pct = (
                (base.total_dwell_seconds - cand.total_dwell_seconds)
                / base.total_dwell_seconds
            ) * 100.0
            if drop_pct > pol.max_dwell_drop_pct:
                violations.append(
                    RegressionViolation(
                        rule_name="max_dwell_drop_pct",
                        severity="error",
                        metric_name="total_dwell_seconds",
                        baseline_value=base.total_dwell_seconds,
                        candidate_value=cand.total_dwell_seconds,
                        threshold=pol.max_dwell_drop_pct,
                        message=(
                            f"Total dwell time dropped by {drop_pct:.1f}% "
                            f"({base.total_dwell_seconds:.2f}s -> {cand.total_dwell_seconds:.2f}s, "
                            f"max allowed {pol.max_dwell_drop_pct:.1f}%)"
                        ),
                    )
                )

    if pol.max_high_critical_insights_increase is not None:
        checks += 1
        inc = cand.high_critical_insights_count - base.high_critical_insights_count
        if inc > pol.max_high_critical_insights_increase:
            violations.append(
                RegressionViolation(
                    rule_name="max_high_critical_insights_increase",
                    severity="error",
                    metric_name="high_critical_insights_count",
                    baseline_value=float(base.high_critical_insights_count),
                    candidate_value=float(cand.high_critical_insights_count),
                    threshold=float(pol.max_high_critical_insights_increase),
                    message=(
                        f"High/critical insights increased by +{inc} "
                        f"({base.high_critical_insights_count} -> {cand.high_critical_insights_count}, "
                        f"max allowed +{pol.max_high_critical_insights_increase})"
                    ),
                )
            )

    if pol.max_spatial_shift_mean_abs is not None and report.spatial_shift is not None:
        checks += 1
        shift_val = report.spatial_shift.mean_abs_delta
        if shift_val > pol.max_spatial_shift_mean_abs:
            violations.append(
                RegressionViolation(
                    rule_name="max_spatial_shift_mean_abs",
                    severity="error",
                    metric_name="spatial_shift.mean_abs_delta",
                    baseline_value=0.0,
                    candidate_value=shift_val,
                    threshold=pol.max_spatial_shift_mean_abs,
                    message=(
                        f"Spatial density mean_abs_delta {shift_val:.4f} exceeded "
                        f"threshold {pol.max_spatial_shift_mean_abs:.4f}"
                    ),
                )
            )

    if pol.min_detections_ratio is not None:
        checks += 1
        if base.detections_count > 0:
            ratio = cand.detections_count / base.detections_count
            if ratio < pol.min_detections_ratio:
                violations.append(
                    RegressionViolation(
                        rule_name="min_detections_ratio",
                        severity="error",
                        metric_name="detections_count",
                        baseline_value=float(base.detections_count),
                        candidate_value=float(cand.detections_count),
                        threshold=pol.min_detections_ratio,
                        message=(
                            f"Detections ratio {ratio:.3f} "
                            f"({cand.detections_count}/{base.detections_count}) fell below "
                            f"minimum {pol.min_detections_ratio:.3f}"
                        ),
                    )
                )

    if not pol.allow_config_drift:
        checks += 1
        if report.config_diffs:
            changed_keys = ", ".join(cd.key for cd in report.config_diffs[:5])
            violations.append(
                RegressionViolation(
                    rule_name="allow_config_drift",
                    severity="error",
                    metric_name="config_diffs",
                    baseline_value=0.0,
                    candidate_value=float(len(report.config_diffs)),
                    threshold=0.0,
                    message=(
                        f"Config drift disallowed but {len(report.config_diffs)} "
                        f"parameter(s) changed: {changed_keys}"
                    ),
                )
            )

    passed = not any(v.severity == "error" for v in violations)
    return RegressionGateResult(
        passed=passed,
        policy=pol,
        checks_evaluated=checks,
        violations=violations,
    )
