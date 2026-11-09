"""Phase 8 promotion: spatial density heatmaps, zone/line count summaries, and duration stats."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "08_heatmaps_analytics"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_heatmaps_analytics import (
    LINE_COUNTS_CSV_HEADER,
    VALID_COLORMAPS,
    VALID_NORMALIZATION_MODES,
    ZONE_COUNTS_CSV_HEADER,
    AnalyticsConfig,
    LineCountRecord,
    TrackDurationStats,
    ZoneCountRecord,
    accumulate_trajectory_grid,
    compute_line_counts,
    compute_track_duration_stats,
    compute_zone_counts,
    load_analytics_config,
    read_line_counts_csv,
    read_zone_counts_csv,
    render_heatmap_overlay,
    run_heatmaps_analytics,
    save_analytics_config,
    smooth_and_normalize_heatmap,
    write_line_counts_csv,
    write_zone_counts_csv,
)

__all__ = [
    "LINE_COUNTS_CSV_HEADER",
    "VALID_COLORMAPS",
    "VALID_NORMALIZATION_MODES",
    "ZONE_COUNTS_CSV_HEADER",
    "AnalyticsConfig",
    "LineCountRecord",
    "TrackDurationStats",
    "ZoneCountRecord",
    "accumulate_trajectory_grid",
    "compute_line_counts",
    "compute_track_duration_stats",
    "compute_zone_counts",
    "load_analytics_config",
    "read_line_counts_csv",
    "read_zone_counts_csv",
    "render_heatmap_overlay",
    "run_heatmaps_analytics",
    "save_analytics_config",
    "smooth_and_normalize_heatmap",
    "write_line_counts_csv",
    "write_zone_counts_csv",
]
