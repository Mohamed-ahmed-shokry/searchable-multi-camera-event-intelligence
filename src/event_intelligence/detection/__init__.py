"""Phase 2 promotion: YOLO11 person detection, CSV serialization, and frame annotation."""

from __future__ import annotations

from event_intelligence.detection.compare_models import (
    DEFAULT_MODELS,
    benchmark_model,
    compare_models,
    print_comparison_table,
)
from event_intelligence.detection.detect_people import (
    DETECTIONS_CSV_HEADER,
    Detection,
    FrameDetections,
    PersonDetector,
    create_video_writer,
    draw_detections,
    load_config,
    read_detections_csv,
    run_detection,
    select_default_device,
    write_detections_csv,
)

__all__ = [
    "DEFAULT_MODELS",
    "DETECTIONS_CSV_HEADER",
    "Detection",
    "FrameDetections",
    "PersonDetector",
    "benchmark_model",
    "compare_models",
    "create_video_writer",
    "draw_detections",
    "load_config",
    "print_comparison_table",
    "read_detections_csv",
    "run_detection",
    "select_default_device",
    "write_detections_csv",
]
