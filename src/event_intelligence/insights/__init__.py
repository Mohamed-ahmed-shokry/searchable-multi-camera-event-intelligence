"""Behavioral insight and anomaly detection intelligence module."""

from __future__ import annotations

from event_intelligence.insights.detectors import (
    detect_cross_camera_transitions,
    detect_loitering_insights,
    detect_speed_anomaly_insights,
    detect_stationary_and_drop_insights,
    detect_wrong_way_insights,
    generate_all_insights,
)
from event_intelligence.insights.report import (
    generate_intelligence_report_markdown,
    write_insights_csv,
    write_insights_json,
    write_intelligence_report,
)
from event_intelligence.insights.types import (
    VALID_INSIGHT_TYPES,
    VALID_SEVERITIES,
    InsightConfig,
    InsightRecord,
    InsightType,
    Severity,
)

__all__ = [
    "VALID_INSIGHT_TYPES",
    "VALID_SEVERITIES",
    "InsightConfig",
    "InsightRecord",
    "InsightType",
    "Severity",
    "detect_cross_camera_transitions",
    "detect_loitering_insights",
    "detect_speed_anomaly_insights",
    "detect_stationary_and_drop_insights",
    "detect_wrong_way_insights",
    "generate_all_insights",
    "generate_intelligence_report_markdown",
    "write_insights_csv",
    "write_insights_json",
    "write_intelligence_report",
]
