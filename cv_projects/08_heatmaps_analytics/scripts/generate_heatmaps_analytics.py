"""Phase 8: Heatmaps and Analytics 2D spatial density accumulator, Gaussian smoother,
reference-frame heatmap renderer, per-zone and per-line activity aggregators,
track duration distribution calculator, and CLI entrypoint.
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from extract_trajectories import TrajectoryPoint
except ImportError:
    _subproject_4_scripts = (
        Path(__file__).resolve().parents[2] / "04_trajectory_extraction" / "scripts"
    )
    if str(_subproject_4_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_4_scripts))
    from extract_trajectories import TrajectoryPoint

try:
    from reason_zones_lines import SceneConfig, draw_zones_and_lines
except ImportError:
    _subproject_5_scripts = (
        Path(__file__).resolve().parents[2] / "05_zone_line_reasoning" / "scripts"
    )
    if str(_subproject_5_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_5_scripts))
    from reason_zones_lines import SceneConfig, draw_zones_and_lines

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


def accumulate_trajectory_grid(
    points: Sequence[TrajectoryPoint],
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

    grid_cols = max(1, math.ceil(frame_width / grid_bin_size_px))
    grid_rows = max(1, math.ceil(frame_height / grid_bin_size_px))
    raw_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)

    per_track_cell_counts: dict[tuple[int, int, int], int] = {}
    for pt in points:
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
