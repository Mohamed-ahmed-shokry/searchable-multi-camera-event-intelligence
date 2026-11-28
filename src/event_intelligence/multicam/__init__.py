"""Multi-camera spatial-temporal correlation and cross-camera transition analytics."""

from __future__ import annotations

from event_intelligence.multicam.engine import (
    analyze_multi_camera_correlation,
    apply_calibrated_offsets,
    build_cross_camera_journeys,
    compute_co_activity_windows,
    correlate_cross_camera_transitions,
    estimate_camera_clock_offsets,
    extract_track_endpoints,
    infer_default_topology,
    load_camera_topology,
    save_camera_topology,
    summarize_camera_pair_flows,
    validate_camera_topology,
)
from event_intelligence.multicam.report import (
    generate_correlation_markdown,
    write_correlation_artifacts,
)
from event_intelligence.multicam.types import (
    CameraClockOffsetEstimate,
    CameraPairFlow,
    CameraTopology,
    CameraTransitionRule,
    ClockCalibrationReport,
    CoActivityWindow,
    CrossCameraJourney,
    CrossCameraTransition,
    MultiCameraCorrelationReport,
    TopologyValidationIssue,
    TopologyValidationReport,
    TrackEndpoint,
)

__all__ = [
    "CameraClockOffsetEstimate",
    "CameraPairFlow",
    "CameraTopology",
    "CameraTransitionRule",
    "ClockCalibrationReport",
    "CoActivityWindow",
    "CrossCameraJourney",
    "CrossCameraTransition",
    "MultiCameraCorrelationReport",
    "TopologyValidationIssue",
    "TopologyValidationReport",
    "TrackEndpoint",
    "analyze_multi_camera_correlation",
    "apply_calibrated_offsets",
    "build_cross_camera_journeys",
    "compute_co_activity_windows",
    "correlate_cross_camera_transitions",
    "estimate_camera_clock_offsets",
    "extract_track_endpoints",
    "generate_correlation_markdown",
    "infer_default_topology",
    "load_camera_topology",
    "save_camera_topology",
    "summarize_camera_pair_flows",
    "validate_camera_topology",
    "write_correlation_artifacts",
]
