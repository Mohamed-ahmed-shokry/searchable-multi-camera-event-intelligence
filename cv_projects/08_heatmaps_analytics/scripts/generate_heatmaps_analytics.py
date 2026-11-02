"""Phase 8: Heatmaps and Analytics 2D spatial density accumulator, Gaussian smoother,
reference-frame heatmap renderer, per-zone and per-line activity aggregators,
track duration distribution calculator, and CLI entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
