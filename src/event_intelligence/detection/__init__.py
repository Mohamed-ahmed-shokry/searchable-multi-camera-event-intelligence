"""Phase 2 promotion: YOLO11 person detection, CSV serialization, and frame annotation."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "02_person_detection"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from detect_people import (
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
    "DETECTIONS_CSV_HEADER",
    "Detection",
    "FrameDetections",
    "PersonDetector",
    "create_video_writer",
    "draw_detections",
    "load_config",
    "read_detections_csv",
    "run_detection",
    "select_default_device",
    "write_detections_csv",
]
