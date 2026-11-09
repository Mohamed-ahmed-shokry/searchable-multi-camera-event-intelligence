"""Phase 9 promotion: quantitative MOT17 and qualitative surveillance CV evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3] / "cv_projects" / "09_cv_evaluation" / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluate_pipeline import (
    DetectionMetrics,
    EvaluationConfig,
    FailureCaseRecord,
    QualitativeSpotCheck,
    StageFpsRecord,
    TrackingMetrics,
    collect_stage_fps_table,
    evaluate_detection_precision_recall,
    evaluate_qualitative_events,
    evaluate_tracking_metrics,
    export_failure_examples,
    generate_evaluation_report_markdown,
    load_evaluation_config,
    run_cv_evaluation,
    save_evaluation_config,
)

__all__ = [
    "DetectionMetrics",
    "EvaluationConfig",
    "FailureCaseRecord",
    "QualitativeSpotCheck",
    "StageFpsRecord",
    "TrackingMetrics",
    "collect_stage_fps_table",
    "evaluate_detection_precision_recall",
    "evaluate_qualitative_events",
    "evaluate_tracking_metrics",
    "export_failure_examples",
    "generate_evaluation_report_markdown",
    "load_evaluation_config",
    "run_cv_evaluation",
    "save_evaluation_config",
]
