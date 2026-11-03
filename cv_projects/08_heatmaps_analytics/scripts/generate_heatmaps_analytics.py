"""Phase 8: Heatmaps and Analytics 2D spatial density accumulator, Gaussian smoother,
reference-frame heatmap renderer, per-zone and per-line activity aggregators,
track duration distribution calculator, and CLI entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from extract_trajectories import TrajectoryPoint, read_trajectories_csv
except ImportError:
    _subproject_4_scripts = (
        Path(__file__).resolve().parents[2] / "04_trajectory_extraction" / "scripts"
    )
    if str(_subproject_4_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_4_scripts))
    from extract_trajectories import TrajectoryPoint, read_trajectories_csv

try:
    from reason_zones_lines import (
        SceneConfig,
        VideoSource,
        draw_zones_and_lines,
        load_scene_config,
        point_in_zone,
    )
except ImportError:
    _subproject_5_scripts = (
        Path(__file__).resolve().parents[2] / "05_zone_line_reasoning" / "scripts"
    )
    if str(_subproject_5_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_5_scripts))
    from reason_zones_lines import (
        SceneConfig,
        VideoSource,
        draw_zones_and_lines,
        load_scene_config,
        point_in_zone,
    )

try:
    from generate_dwell_events import EVENTS_CSV_HEADER, read_events_csv
except ImportError:
    _subproject_6_scripts = (
        Path(__file__).resolve().parents[2] / "06_dwell_time_events" / "scripts"
    )
    if str(_subproject_6_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_6_scripts))
    from generate_dwell_events import EVENTS_CSV_HEADER, read_events_csv

try:
    from extract_visual_evidence import (
        EVIDENCE_EVENTS_CSV_HEADER,
        read_evidence_events_csv,
    )
except ImportError:
    _subproject_7_scripts = (
        Path(__file__).resolve().parents[2] / "07_visual_evidence" / "scripts"
    )
    if str(_subproject_7_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_7_scripts))
    from extract_visual_evidence import (
        EVIDENCE_EVENTS_CSV_HEADER,
        read_evidence_events_csv,
    )

ZONE_COUNTS_CSV_HEADER: list[str] = [
    "zone_id",
    "entry_count",
    "exit_count",
    "total_dwell_seconds",
    "peak_occupancy",
]

LINE_COUNTS_CSV_HEADER: list[str] = [
    "line_id",
    "crossings_ab",
    "crossings_ba",
    "total",
]

VALID_NORMALIZATION_MODES: tuple[str, ...] = ("log", "percentile", "linear")
VALID_COLORMAPS: tuple[str, ...] = ("jet", "inferno", "turbo", "hot")


@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for Phase 8 heatmap rendering and statistical aggregation."""

    grid_bin_size_px: int = 16
    gaussian_sigma_px: float = 24.0
    normalization_mode: str = "log"
    percentile_clip: float = 99.0
    max_points_per_track_cell: int = 6
    heatmap_alpha: float = 0.55
    colormap: str = "jet"
    overlay_scene_geometry: bool = True
    reference_frame_index: int = 35
    duration_bucket_edges_seconds: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 10.0)

    def __post_init__(self) -> None:
        if self.grid_bin_size_px < 1:
            raise ValueError(
                f"grid_bin_size_px must be >= 1, got {self.grid_bin_size_px}"
            )
        if self.gaussian_sigma_px < 0.0:
            raise ValueError(
                f"gaussian_sigma_px must be >= 0.0, got {self.gaussian_sigma_px}"
            )
        if self.normalization_mode not in VALID_NORMALIZATION_MODES:
            raise ValueError(
                f"normalization_mode must be one of {VALID_NORMALIZATION_MODES}, "
                f"got {self.normalization_mode!r}"
            )
        if not (0.0 < self.percentile_clip <= 100.0):
            raise ValueError(
                f"percentile_clip must be in (0, 100], got {self.percentile_clip}"
            )
        if self.max_points_per_track_cell < 0:
            raise ValueError(
                f"max_points_per_track_cell must be >= 0, got {self.max_points_per_track_cell}"
            )
        if not (0.0 < self.heatmap_alpha <= 1.0):
            raise ValueError(
                f"heatmap_alpha must be in (0, 1], got {self.heatmap_alpha}"
            )
        if self.colormap not in VALID_COLORMAPS:
            raise ValueError(
                f"colormap must be one of {VALID_COLORMAPS}, got {self.colormap!r}"
            )
        if self.reference_frame_index < 0:
            raise ValueError(
                f"reference_frame_index must be >= 0, got {self.reference_frame_index}"
            )
        if len(self.duration_bucket_edges_seconds) < 2:
            raise ValueError("duration_bucket_edges_seconds must have at least 2 edges")


@dataclass(frozen=True)
class ZoneCountRecord:
    """Per-zone activity statistics row for zone_counts.csv."""

    zone_id: str
    entry_count: int
    exit_count: int
    total_dwell_seconds: float
    peak_occupancy: int

    def __post_init__(self) -> None:
        if not self.zone_id or not self.zone_id.strip():
            raise ValueError("zone_id must be a non-empty string")
        if self.entry_count < 0:
            raise ValueError(f"entry_count must be >= 0, got {self.entry_count}")
        if self.exit_count < 0:
            raise ValueError(f"exit_count must be >= 0, got {self.exit_count}")
        if self.total_dwell_seconds < 0.0:
            raise ValueError(
                f"total_dwell_seconds must be >= 0.0, got {self.total_dwell_seconds}"
            )
        if self.peak_occupancy < 0:
            raise ValueError(f"peak_occupancy must be >= 0, got {self.peak_occupancy}")

    def to_row(self) -> list[str]:
        return [
            self.zone_id,
            str(self.entry_count),
            str(self.exit_count),
            f"{self.total_dwell_seconds:.4f}",
            str(self.peak_occupancy),
        ]


@dataclass(frozen=True)
class LineCountRecord:
    """Per-line directional crossing statistics row for line_counts.csv."""

    line_id: str
    crossings_ab: int
    crossings_ba: int
    total: int

    def __post_init__(self) -> None:
        if not self.line_id or not self.line_id.strip():
            raise ValueError("line_id must be a non-empty string")
        if self.crossings_ab < 0:
            raise ValueError(f"crossings_ab must be >= 0, got {self.crossings_ab}")
        if self.crossings_ba < 0:
            raise ValueError(f"crossings_ba must be >= 0, got {self.crossings_ba}")
        if self.total != self.crossings_ab + self.crossings_ba:
            raise ValueError(
                f"total ({self.total}) must equal crossings_ab ({self.crossings_ab}) "
                f"+ crossings_ba ({self.crossings_ba})"
            )

    def to_row(self) -> list[str]:
        return [
            self.line_id,
            str(self.crossings_ab),
            str(self.crossings_ba),
            str(self.total),
        ]


@dataclass(frozen=True)
class TrackDurationStats:
    """Distribution statistics for per-track durations and path lengths."""

    track_count: int
    min_duration_seconds: float
    max_duration_seconds: float
    mean_duration_seconds: float
    median_duration_seconds: float
    p90_duration_seconds: float
    mean_path_length_px: float
    max_path_length_px: float
    duration_buckets: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_count": self.track_count,
            "min_duration_seconds": round(self.min_duration_seconds, 4),
            "max_duration_seconds": round(self.max_duration_seconds, 4),
            "mean_duration_seconds": round(self.mean_duration_seconds, 4),
            "median_duration_seconds": round(self.median_duration_seconds, 4),
            "p90_duration_seconds": round(self.p90_duration_seconds, 4),
            "mean_path_length_px": round(self.mean_path_length_px, 2),
            "max_path_length_px": round(self.max_path_length_px, 2),
            "duration_buckets": dict(self.duration_buckets),
        }


def load_analytics_config(config_path: Path | str) -> AnalyticsConfig:
    """Load an AnalyticsConfig from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Analytics config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)

    if raw is None:
        return AnalyticsConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")

    defaults = AnalyticsConfig()
    raw_edges = raw.get(
        "duration_bucket_edges_seconds", defaults.duration_bucket_edges_seconds
    )
    edges = tuple(float(v) for v in raw_edges)

    return AnalyticsConfig(
        grid_bin_size_px=int(raw.get("grid_bin_size_px", defaults.grid_bin_size_px)),
        gaussian_sigma_px=float(
            raw.get("gaussian_sigma_px", defaults.gaussian_sigma_px)
        ),
        normalization_mode=str(
            raw.get("normalization_mode", defaults.normalization_mode)
        ),
        percentile_clip=float(raw.get("percentile_clip", defaults.percentile_clip)),
        max_points_per_track_cell=int(
            raw.get("max_points_per_track_cell", defaults.max_points_per_track_cell)
        ),
        heatmap_alpha=float(raw.get("heatmap_alpha", defaults.heatmap_alpha)),
        colormap=str(raw.get("colormap", defaults.colormap)),
        overlay_scene_geometry=bool(
            raw.get("overlay_scene_geometry", defaults.overlay_scene_geometry)
        ),
        reference_frame_index=int(
            raw.get("reference_frame_index", defaults.reference_frame_index)
        ),
        duration_bucket_edges_seconds=edges,
    )


def save_analytics_config(config: AnalyticsConfig, config_path: Path | str) -> Path:
    """Save an AnalyticsConfig to a YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["duration_bucket_edges_seconds"] = list(
        config.duration_bucket_edges_seconds
    )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def write_zone_counts_csv(
    records: Sequence[ZoneCountRecord], csv_path: Path | str
) -> Path:
    """Write ZoneCountRecord rows to zone_counts.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ZONE_COUNTS_CSV_HEADER)
        for rec in records:
            writer.writerow(rec.to_row())
    return path


def read_zone_counts_csv(csv_path: Path | str) -> list[ZoneCountRecord]:
    """Read ZoneCountRecord rows from zone_counts.csv."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"zone_counts.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ZONE_COUNTS_CSV_HEADER:
            raise ValueError(
                f"Unexpected zone_counts.csv header in {path}: got {header}, "
                f"expected {ZONE_COUNTS_CSV_HEADER}"
            )

        records: list[ZoneCountRecord] = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(ZONE_COUNTS_CSV_HEADER):
                raise ValueError(
                    f"Malformed row in {path}: expected {len(ZONE_COUNTS_CSV_HEADER)} "
                    f"columns, got {len(row)}: {row}"
                )
            records.append(
                ZoneCountRecord(
                    zone_id=row[0].strip(),
                    entry_count=int(row[1].strip()),
                    exit_count=int(row[2].strip()),
                    total_dwell_seconds=float(row[3].strip()),
                    peak_occupancy=int(row[4].strip()),
                )
            )
    return records


def write_line_counts_csv(
    records: Sequence[LineCountRecord], csv_path: Path | str
) -> Path:
    """Write LineCountRecord rows to line_counts.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LINE_COUNTS_CSV_HEADER)
        for rec in records:
            writer.writerow(rec.to_row())
    return path


def read_line_counts_csv(csv_path: Path | str) -> list[LineCountRecord]:
    """Read LineCountRecord rows from line_counts.csv."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"line_counts.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != LINE_COUNTS_CSV_HEADER:
            raise ValueError(
                f"Unexpected line_counts.csv header in {path}: got {header}, "
                f"expected {LINE_COUNTS_CSV_HEADER}"
            )

        records: list[LineCountRecord] = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(LINE_COUNTS_CSV_HEADER):
                raise ValueError(
                    f"Malformed row in {path}: expected {len(LINE_COUNTS_CSV_HEADER)} "
                    f"columns, got {len(row)}: {row}"
                )
            records.append(
                LineCountRecord(
                    line_id=row[0].strip(),
                    crossings_ab=int(row[1].strip()),
                    crossings_ba=int(row[2].strip()),
                    total=int(row[3].strip()),
                )
            )
    return records


def _flatten_trajectory_points(
    points: Mapping[int, Sequence[TrajectoryPoint]] | Sequence[TrajectoryPoint],
) -> list[TrajectoryPoint]:
    """Normalize either a track_id->points mapping or a flat sequence into a list of TrajectoryPoint."""
    if isinstance(points, Mapping):
        flat: list[TrajectoryPoint] = []
        for track_id in sorted(points):
            flat.extend(points[track_id])
        return flat
    return list(points)


def accumulate_trajectory_grid(
    points: Mapping[int, Sequence[TrajectoryPoint]] | Sequence[TrajectoryPoint],
    *,
    frame_width: int,
    frame_height: int,
    grid_bin_size_px: int = 16,
    max_points_per_track_cell: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate trajectory coordinates into a 2D spatial grid and upsample to frame size.

    Returns ``(raw_grid, full_res_density)`` where ``raw_grid`` has shape
    ``(grid_rows, grid_cols)`` and ``full_res_density`` has exact shape
    ``(frame_height, frame_width)``.
    """
    if frame_width < 1 or frame_height < 1:
        raise ValueError(
            f"frame_width ({frame_width}) and frame_height ({frame_height}) must be >= 1"
        )
    if grid_bin_size_px < 1:
        raise ValueError(f"grid_bin_size_px must be >= 1, got {grid_bin_size_px}")
    if max_points_per_track_cell < 0:
        raise ValueError(
            f"max_points_per_track_cell must be >= 0, got {max_points_per_track_cell}"
        )

    flat_points = _flatten_trajectory_points(points)
    grid_cols = max(1, math.ceil(frame_width / grid_bin_size_px))
    grid_rows = max(1, math.ceil(frame_height / grid_bin_size_px))
    raw_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    per_track_cell_counts: dict[tuple[int, int, int], int] = {}
    for pt in flat_points:
        clamped_x = max(0.0, min(float(frame_width - 1), pt.x))
        clamped_y = max(0.0, min(float(frame_height - 1), pt.y))
        col_idx = min(grid_cols - 1, int(clamped_x // grid_bin_size_px))
        row_idx = min(grid_rows - 1, int(clamped_y // grid_bin_size_px))

        if max_points_per_track_cell > 0:
            key = (pt.track_id, row_idx, col_idx)
            seen = per_track_cell_counts.get(key, 0)
            if seen >= max_points_per_track_cell:
                continue
            per_track_cell_counts[key] = seen + 1

        raw_grid[row_idx, col_idx] += 1.0

    full_res_density = cv2.resize(
        raw_grid,
        (frame_width, frame_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return raw_grid, full_res_density.astype(np.float32)


def smooth_and_normalize_heatmap(
    full_res_density: np.ndarray,
    *,
    gaussian_sigma_px: float = 24.0,
    normalization_mode: str = "log",
    percentile_clip: float = 99.0,
) -> np.ndarray:
    """Apply Gaussian smoothing and normalize a 2D density array into ``[0.0, 1.0]``."""
    if normalization_mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(
            f"normalization_mode must be one of {VALID_NORMALIZATION_MODES}, "
            f"got {normalization_mode!r}"
        )
    if gaussian_sigma_px < 0.0:
        raise ValueError(f"gaussian_sigma_px must be >= 0.0, got {gaussian_sigma_px}")

    density = full_res_density.astype(np.float32)
    if gaussian_sigma_px > 0.0:
        smoothed = cv2.GaussianBlur(
            density,
            (0, 0),
            sigmaX=gaussian_sigma_px,
            sigmaY=gaussian_sigma_px,
        )
    else:
        smoothed = density.copy()

    smoothed = np.maximum(0.0, smoothed)
    max_val = float(smoothed.max()) if smoothed.size > 0 else 0.0
    if max_val <= 1e-9:
        return np.zeros_like(smoothed, dtype=np.float32)

    if normalization_mode == "log":
        normalized = np.log1p(smoothed) / float(np.log1p(max_val))
    elif normalization_mode == "percentile":
        nonzero = smoothed[smoothed > 1e-9]
        clip_val = (
            float(np.percentile(nonzero, percentile_clip))
            if nonzero.size > 0
            else max_val
        )
        clip_val = max(clip_val, 1e-9)
        normalized = np.clip(smoothed / clip_val, 0.0, 1.0)
    else:  # "linear"
        normalized = smoothed / max_val

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _resolve_cv2_colormap(colormap: str) -> int:
    mapping: dict[str, int] = {
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
    }
    if colormap not in mapping:
        raise ValueError(f"colormap must be one of {VALID_COLORMAPS}, got {colormap!r}")
    return mapping[colormap]


def render_heatmap_overlay(
    reference_frame: np.ndarray,
    normalized_heatmap: np.ndarray,
    *,
    heatmap_alpha: float = 0.55,
    colormap: str = "jet",
    scene_config: SceneConfig | None = None,
    title: str = "Spatial Activity Heatmap",
) -> np.ndarray:
    """Render a Gaussian-smoothed normalized heatmap overlaid on a reference frame."""
    if reference_frame.ndim != 3 or reference_frame.shape[2] != 3:
        raise ValueError(
            f"reference_frame must have shape (H, W, 3), got {reference_frame.shape}"
        )
    if not (0.0 < heatmap_alpha <= 1.0):
        raise ValueError(f"heatmap_alpha must be in (0, 1], got {heatmap_alpha}")

    height, width = int(reference_frame.shape[0]), int(reference_frame.shape[1])
    norm = normalized_heatmap.astype(np.float32)
    if norm.shape[:2] != (height, width):
        norm = cv2.resize(norm, (width, height), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(norm, 0.0, 1.0)

    cv_cmap = _resolve_cv2_colormap(colormap)
    heat_u8 = np.round(norm * 255.0).astype(np.uint8)
    colored_heat = cv2.applyColorMap(heat_u8, cv_cmap)

    base = reference_frame.astype(np.float32)
    heat_f = colored_heat.astype(np.float32)
    weight = np.where(norm > 0.015, heatmap_alpha * (norm**0.65), 0.0)[..., np.newaxis]
    blended = np.clip(base * (1.0 - weight) + heat_f * weight, 0.0, 255.0).astype(
        np.uint8
    )

    if scene_config is not None:
        blended = draw_zones_and_lines(blended, scene_config)

    # Compact HUD banner + colorbar legend
    bar_w = min(180, max(80, width - 32))
    bar_h = 12
    hud_w = min(width - 16, max(bar_w + 24, 260))
    hud_h = 54
    if width >= 120 and height >= 70:
        cv2.rectangle(blended, (8, 8), (8 + hud_w, 8 + hud_h), (20, 20, 20), -1)
        cv2.rectangle(
            blended, (8, 8), (8 + hud_w, 8 + hud_h), (100, 100, 100), 1, cv2.LINE_AA
        )
        cv2.putText(
            blended,
            title[:34],
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        grad = np.linspace(0, 255, bar_w, dtype=np.uint8)[np.newaxis, :]
        grad_bar = cv2.applyColorMap(np.repeat(grad, bar_h, axis=0), cv_cmap)
        blended[36 : 36 + bar_h, 16 : 16 + bar_w] = grad_bar
        cv2.putText(
            blended,
            "Low",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            "High",
            (16 + bar_w - 26, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return blended


def _interval_peak_occupancy(intervals: Sequence[tuple[int, float, float]]) -> int:
    """Compute peak simultaneous distinct track_ids across [start_time, end_time] intervals."""
    if not intervals:
        return 0
    endpoints: list[tuple[float, int, int]] = []
    for track_id, start_t, end_t in intervals:
        # Sort starts (-1) before ends (+1) at identical timestamps
        endpoints.append((start_t, -1, track_id))
        endpoints.append((end_t, 1, track_id))
    endpoints.sort()

    active_tracks: dict[int, int] = {}
    peak = 0
    for _, kind, track_id in endpoints:
        if kind == -1:
            active_tracks[track_id] = active_tracks.get(track_id, 0) + 1
            peak = max(peak, len(active_tracks))
        else:
            remaining = active_tracks.get(track_id, 0) - 1
            if remaining <= 0:
                active_tracks.pop(track_id, None)
            else:
                active_tracks[track_id] = remaining
    return peak


def compute_zone_counts(
    events: Sequence[Any],
    *,
    trajectory_points: Mapping[int, Sequence[TrajectoryPoint]]
    | Sequence[TrajectoryPoint] = (),
    scene_config: SceneConfig | None = None,
) -> list[ZoneCountRecord]:
    """Aggregate per-zone entry_count, exit_count, total_dwell_seconds, and peak_occupancy."""
    flat_points = _flatten_trajectory_points(trajectory_points)
    zone_order: list[str] = []
    zone_map = {}
    if scene_config is not None:
        for z in scene_config.zones:
            if z.id not in zone_order:
                zone_order.append(z.id)
                zone_map[z.id] = z

    for ev in events:
        if (
            ev.event_type in {"zone_entry", "zone_exit", "dwell_time_event"}
            and ev.zone_id not in zone_order
        ):
            zone_order.append(ev.zone_id)

    # Pre-compute frame-by-frame polygon occupancy when scene_config + trajectories are available
    frame_peak_by_zone: dict[str, int] = {zid: 0 for zid in zone_order}
    if zone_map and flat_points:
        by_frame: dict[int, list[TrajectoryPoint]] = {}
        for pt in flat_points:
            by_frame.setdefault(pt.frame_id, []).append(pt)
        for frame_pts in by_frame.values():
            for zid, poly_zone in zone_map.items():
                occupants = {
                    pt.track_id
                    for pt in frame_pts
                    if point_in_zone(pt.x, pt.y, poly_zone)
                }
                frame_peak_by_zone[zid] = max(frame_peak_by_zone[zid], len(occupants))

    records: list[ZoneCountRecord] = []
    for zid in zone_order:
        entries = sum(
            1 for ev in events if ev.event_type == "zone_entry" and ev.zone_id == zid
        )
        exits = sum(
            1 for ev in events if ev.event_type == "zone_exit" and ev.zone_id == zid
        )
        dwell_events = [
            ev
            for ev in events
            if ev.event_type == "dwell_time_event" and ev.zone_id == zid
        ]
        total_dwell = sum(float(ev.duration_seconds) for ev in dwell_events)
        intervals = [
            (int(ev.track_id), float(ev.start_time), float(ev.end_time))
            for ev in dwell_events
        ]
        interval_peak = _interval_peak_occupancy(intervals)
        fallback_peak = 1 if (entries > 0 or dwell_events) else 0
        peak_occ = max(frame_peak_by_zone.get(zid, 0), interval_peak, fallback_peak)

        records.append(
            ZoneCountRecord(
                zone_id=zid,
                entry_count=entries,
                exit_count=exits,
                total_dwell_seconds=round(total_dwell, 4),
                peak_occupancy=peak_occ,
            )
        )
    return records


def compute_line_counts(
    events: Sequence[Any],
    *,
    scene_config: SceneConfig | None = None,
) -> list[LineCountRecord]:
    """Aggregate per-line directional crossings (A->B, B->A, total) from line_crossing events."""
    line_order: list[str] = []
    if scene_config is not None:
        for ln in scene_config.lines:
            if ln.id not in line_order:
                line_order.append(ln.id)

    ab_counts: dict[str, int] = {}
    ba_counts: dict[str, int] = {}

    for ev in events:
        if ev.event_type != "line_crossing":
            continue
        raw_zone = str(ev.zone_id)
        if ":" in raw_zone:
            line_id, direction = raw_zone.split(":", 1)
        else:
            line_id, direction = raw_zone, "A->B"
        if line_id not in line_order:
            line_order.append(line_id)
        if direction.strip() == "B->A":
            ba_counts[line_id] = ba_counts.get(line_id, 0) + 1
        else:
            ab_counts[line_id] = ab_counts.get(line_id, 0) + 1

    records: list[LineCountRecord] = []
    for lid in line_order:
        ab = ab_counts.get(lid, 0)
        ba = ba_counts.get(lid, 0)
        records.append(
            LineCountRecord(
                line_id=lid,
                crossings_ab=ab,
                crossings_ba=ba,
                total=ab + ba,
            )
        )
    return records


def compute_track_duration_stats(
    points: Mapping[int, Sequence[TrajectoryPoint]] | Sequence[TrajectoryPoint],
    *,
    bucket_edges_seconds: Sequence[float] = (0.0, 2.0, 4.0, 6.0, 10.0),
) -> TrackDurationStats:
    """Compute track duration and path-length distribution metrics across all tracks."""
    flat_points = _flatten_trajectory_points(points)
    edges = sorted(float(v) for v in bucket_edges_seconds)
    if len(edges) < 2:
        raise ValueError("bucket_edges_seconds must contain at least 2 values")

    buckets: dict[str, int] = {}
    for a, b in itertools.pairwise(edges):
        buckets[f"{a:.1f}-{b:.1f}s"] = 0
    buckets[f">={edges[-1]:.1f}s"] = 0

    if not flat_points:
        return TrackDurationStats(
            track_count=0,
            min_duration_seconds=0.0,
            max_duration_seconds=0.0,
            mean_duration_seconds=0.0,
            median_duration_seconds=0.0,
            p90_duration_seconds=0.0,
            mean_path_length_px=0.0,
            max_path_length_px=0.0,
            duration_buckets=buckets,
        )

    by_track: dict[int, list[TrajectoryPoint]] = {}
    for pt in flat_points:
        by_track.setdefault(pt.track_id, []).append(pt)

    durations: list[float] = []
    path_lengths: list[float] = []

    for track_pts in by_track.values():
        ordered = sorted(track_pts, key=lambda p: p.frame_id)
        dur = max(0.0, float(ordered[-1].timestamp - ordered[0].timestamp))
        plen = sum(
            math.hypot(p2.x - p1.x, p2.y - p1.y)
            for p1, p2 in itertools.pairwise(ordered)
        )
        durations.append(dur)
        path_lengths.append(plen)

        placed = False
        for a, b in itertools.pairwise(edges):
            if a <= dur < b:
                buckets[f"{a:.1f}-{b:.1f}s"] += 1
                placed = True
                break
        if not placed:
            buckets[f">={edges[-1]:.1f}s"] += 1

    dur_arr = np.asarray(durations, dtype=np.float64)
    len_arr = np.asarray(path_lengths, dtype=np.float64)

    return TrackDurationStats(
        track_count=len(durations),
        min_duration_seconds=float(dur_arr.min()),
        max_duration_seconds=float(dur_arr.max()),
        mean_duration_seconds=float(dur_arr.mean()),
        median_duration_seconds=float(np.median(dur_arr)),
        p90_duration_seconds=float(np.percentile(dur_arr, 90.0)),
        mean_path_length_px=float(len_arr.mean()),
        max_path_length_px=float(len_arr.max()),
        duration_buckets=buckets,
    )


def _load_any_events_csv(events_csv_path: Path) -> list[Any]:
    """Load either Phase 6 (7-column) or Phase 7 (9-column) events.csv."""
    if not events_csv_path.is_file():
        raise FileNotFoundError(f"events.csv not found: {events_csv_path}")
    with events_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header == EVIDENCE_EVENTS_CSV_HEADER:
        return list(read_evidence_events_csv(events_csv_path))
    if header == EVENTS_CSV_HEADER:
        return list(read_events_csv(events_csv_path))
    raise ValueError(f"Unrecognized events.csv header in {events_csv_path}: {header}")


def run_heatmaps_analytics(
    *,
    source_path: Path,
    trajectories_csv_path: Path,
    events_csv_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    scene_config_path: Path | None = None,
    grid_bin_size_override: int | None = None,
    gaussian_sigma_override: float | None = None,
    normalization_mode_override: str | None = None,
    max_points_per_track_cell_override: int | None = None,
    colormap_override: str | None = None,
) -> dict[str, Any]:
    """Execute the complete Phase 8 heatmaps and analytics pipeline."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = (
        load_analytics_config(config_path)
        if config_path is not None and config_path.is_file()
        else AnalyticsConfig()
    )
    cfg = AnalyticsConfig(
        grid_bin_size_px=(
            grid_bin_size_override
            if grid_bin_size_override is not None
            else base_cfg.grid_bin_size_px
        ),
        gaussian_sigma_px=(
            gaussian_sigma_override
            if gaussian_sigma_override is not None
            else base_cfg.gaussian_sigma_px
        ),
        normalization_mode=(
            normalization_mode_override
            if normalization_mode_override is not None
            else base_cfg.normalization_mode
        ),
        percentile_clip=base_cfg.percentile_clip,
        max_points_per_track_cell=(
            max_points_per_track_cell_override
            if max_points_per_track_cell_override is not None
            else base_cfg.max_points_per_track_cell
        ),
        heatmap_alpha=base_cfg.heatmap_alpha,
        colormap=(
            colormap_override if colormap_override is not None else base_cfg.colormap
        ),
        overlay_scene_geometry=base_cfg.overlay_scene_geometry,
        reference_frame_index=base_cfg.reference_frame_index,
        duration_bucket_edges_seconds=base_cfg.duration_bucket_edges_seconds,
    )

    scene_cfg: SceneConfig | None = None
    if scene_config_path is not None and scene_config_path.is_file():
        scene_cfg = load_scene_config(scene_config_path)

    trajectory_points = _flatten_trajectory_points(
        read_trajectories_csv(trajectories_csv_path)
    )
    events = _load_any_events_csv(events_csv_path)

    with VideoSource(source_path) as vs:
        meta = vs.metadata
        frame_width = int(meta.width)
        frame_height = int(meta.height)
        frame_count = int(meta.frame_count)
        fps = float(meta.fps)
        ref_idx = min(cfg.reference_frame_index, max(0, frame_count - 1))
        reference_frame = vs.read_frame(ref_idx)

    raw_grid, full_res_density = accumulate_trajectory_grid(
        trajectory_points,
        frame_width=frame_width,
        frame_height=frame_height,
        grid_bin_size_px=cfg.grid_bin_size_px,
        max_points_per_track_cell=cfg.max_points_per_track_cell,
    )
    normalized_heatmap = smooth_and_normalize_heatmap(
        full_res_density,
        gaussian_sigma_px=cfg.gaussian_sigma_px,
        normalization_mode=cfg.normalization_mode,
        percentile_clip=cfg.percentile_clip,
    )
    overlay_img = render_heatmap_overlay(
        reference_frame,
        normalized_heatmap,
        heatmap_alpha=cfg.heatmap_alpha,
        colormap=cfg.colormap,
        scene_config=scene_cfg if cfg.overlay_scene_geometry else None,
    )

    heatmap_path = output_dir / "heatmap.png"
    cv2.imwrite(str(heatmap_path), overlay_img)

    zone_counts = compute_zone_counts(
        events,
        trajectory_points=trajectory_points,
        scene_config=scene_cfg,
    )
    line_counts = compute_line_counts(events, scene_config=scene_cfg)
    track_stats = compute_track_duration_stats(
        trajectory_points,
        bucket_edges_seconds=cfg.duration_bucket_edges_seconds,
    )

    zone_counts_path = write_zone_counts_csv(
        zone_counts, output_dir / "zone_counts.csv"
    )
    line_counts_path = write_line_counts_csv(
        line_counts, output_dir / "line_counts.csv"
    )
    config_out_path = save_analytics_config(cfg, output_dir / "analytics.yaml")

    peak_row, peak_col = (
        tuple(
            int(v) for v in np.unravel_index(int(np.argmax(raw_grid)), raw_grid.shape)
        )
        if raw_grid.size > 0
        else (0, 0)
    )
    peak_x_px = round((peak_col + 0.5) * cfg.grid_bin_size_px, 1)
    peak_y_px = round((peak_row + 0.5) * cfg.grid_bin_size_px, 1)

    event_counts_by_type: dict[str, int] = {}
    for ev in events:
        event_counts_by_type[ev.event_type] = (
            event_counts_by_type.get(ev.event_type, 0) + 1
        )

    elapsed_s = time.perf_counter() - t0
    summary: dict[str, Any] = {
        "source_path": str(source_path),
        "trajectories_csv_input": str(trajectories_csv_path),
        "events_csv_input": str(events_csv_path),
        "heatmap_output": str(heatmap_path),
        "zone_counts_output": str(zone_counts_path),
        "line_counts_output": str(line_counts_path),
        "analytics_config_output": str(config_out_path),
        "video_metadata": {
            "width": frame_width,
            "height": frame_height,
            "fps": fps,
            "frame_count": frame_count,
            "reference_frame_index": ref_idx,
        },
        "heatmap_stats": {
            "grid_shape": [int(raw_grid.shape[0]), int(raw_grid.shape[1])],
            "grid_bin_size_px": cfg.grid_bin_size_px,
            "gaussian_sigma_px": cfg.gaussian_sigma_px,
            "normalization_mode": cfg.normalization_mode,
            "max_points_per_track_cell": cfg.max_points_per_track_cell,
            "total_trajectory_points": len(trajectory_points),
            "active_grid_cells": int(np.count_nonzero(raw_grid > 0)),
            "peak_grid_cell": {
                "row": peak_row,
                "col": peak_col,
                "center_x_px": peak_x_px,
                "center_y_px": peak_y_px,
                "capped_count": float(raw_grid[peak_row, peak_col])
                if raw_grid.size > 0
                else 0.0,
            },
        },
        "event_counts_by_type": event_counts_by_type,
        "zone_counts": [
            {
                "zone_id": z.zone_id,
                "entry_count": z.entry_count,
                "exit_count": z.exit_count,
                "total_dwell_seconds": z.total_dwell_seconds,
                "peak_occupancy": z.peak_occupancy,
            }
            for z in zone_counts
        ],
        "line_counts": [
            {
                "line_id": l.line_id,
                "crossings_ab": l.crossings_ab,
                "crossings_ba": l.crossings_ba,
                "total": l.total,
            }
            for l in line_counts
        ],
        "track_duration_distribution": track_stats.to_dict(),
        "elapsed_seconds": round(elapsed_s, 4),
    }

    summary_path = output_dir / "analytics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Phase 8 heatmaps and analytics."""
    parser = argparse.ArgumentParser(
        description="Phase 8: Generate spatial activity heatmap and zone/line analytics."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source video file or MOT sequence directory.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Path to Phase 4 trajectories.csv input file.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Path to Phase 6 or Phase 7 events.csv input file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "analytics.yaml",
        help="Path to analytics.yaml configuration file.",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "05_zone_line_reasoning"
            / "configs"
            / "scene_config.yaml"
        ),
        help="Optional path to Phase 5 scene_config.yaml for zone/line overlays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "default_run",
        help="Directory to write heatmap.png, zone_counts.csv, line_counts.csv, and analytics_summary.json.",
    )
    parser.add_argument(
        "--grid-bin-size",
        type=int,
        default=None,
        help="Optional override for grid_bin_size_px.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Optional override for gaussian_sigma_px.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=list(VALID_NORMALIZATION_MODES),
        default=None,
        help="Optional override for normalization_mode.",
    )
    parser.add_argument(
        "--max-points-per-track-cell",
        type=int,
        default=None,
        help="Optional override for max_points_per_track_cell.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        choices=list(VALID_COLORMAPS),
        default=None,
        help="Optional override for colormap.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 8 heatmaps and analytics."""
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_heatmaps_analytics(
        source_path=args.source,
        trajectories_csv_path=args.trajectories,
        events_csv_path=args.events,
        output_dir=args.output_dir,
        config_path=args.config,
        scene_config_path=args.scene_config,
        grid_bin_size_override=args.grid_bin_size,
        gaussian_sigma_override=args.sigma,
        normalization_mode_override=args.normalization,
        max_points_per_track_cell_override=args.max_points_per_track_cell,
        colormap_override=args.colormap,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
