"""Multi-run comparison, baseline diffing, and cross-run analytics package."""

from __future__ import annotations

from event_intelligence.compare.engine import (
    analyze_multi_run_trends,
    compare_runs,
    compute_insight_deltas,
    compute_line_deltas,
    compute_spatial_density_diff,
    compute_temporal_bucket_deltas,
    compute_zone_deltas,
    diff_run_configs,
    discover_runs,
    load_run_snapshot,
)
from event_intelligence.compare.report import (
    generate_comparison_markdown,
    generate_trends_markdown,
    write_comparison_artifacts,
    write_trends_artifacts,
)
from event_intelligence.compare.types import (
    ConfigDiffEntry,
    InsightTypeDelta,
    LineMetricDelta,
    MultiRunTrendReport,
    RunBaselineAnomaly,
    RunComparisonReport,
    RunSnapshot,
    SpatialShiftSummary,
    TemporalBucketDelta,
    ZoneMetricDelta,
)

__all__ = [
    "ConfigDiffEntry",
    "InsightTypeDelta",
    "LineMetricDelta",
    "MultiRunTrendReport",
    "RunBaselineAnomaly",
    "RunComparisonReport",
    "RunSnapshot",
    "SpatialShiftSummary",
    "TemporalBucketDelta",
    "ZoneMetricDelta",
    "analyze_multi_run_trends",
    "compare_runs",
    "compute_insight_deltas",
    "compute_line_deltas",
    "compute_spatial_density_diff",
    "compute_temporal_bucket_deltas",
    "compute_zone_deltas",
    "diff_run_configs",
    "discover_runs",
    "generate_comparison_markdown",
    "generate_trends_markdown",
    "load_run_snapshot",
    "write_comparison_artifacts",
    "write_trends_artifacts",
]
