"""Phase 4 promotion: ground-plane trajectory extraction, smoothing, and path length."""

from __future__ import annotations

from event_intelligence.trajectories.compare_smoothing import compare_smoothing
from event_intelligence.trajectories.extract_trajectories import (
    TRAJECTORIES_CSV_HEADER,
    TRAJECTORY_SUMMARY_CSV_HEADER,
    VALID_ANCHORS,
    TrajectoryPoint,
    TrajectorySummary,
    analyze_trajectory_quality,
    compute_anchor_point,
    compute_trajectory_summaries,
    draw_trajectory_trails,
    load_trajectory_config,
    read_trajectories_csv,
    read_trajectory_summary_csv,
    render_trajectory_map,
    run_trajectory_extraction,
    smooth_trajectories,
    tracks_to_trajectories,
    write_trajectories_csv,
    write_trajectory_summary_csv,
)

__all__ = [
    "TRAJECTORIES_CSV_HEADER",
    "TRAJECTORY_SUMMARY_CSV_HEADER",
    "VALID_ANCHORS",
    "TrajectoryPoint",
    "TrajectorySummary",
    "analyze_trajectory_quality",
    "compare_smoothing",
    "compute_anchor_point",
    "compute_trajectory_summaries",
    "draw_trajectory_trails",
    "load_trajectory_config",
    "read_trajectories_csv",
    "read_trajectory_summary_csv",
    "render_trajectory_map",
    "run_trajectory_extraction",
    "smooth_trajectories",
    "tracks_to_trajectories",
    "write_trajectories_csv",
    "write_trajectory_summary_csv",
]
