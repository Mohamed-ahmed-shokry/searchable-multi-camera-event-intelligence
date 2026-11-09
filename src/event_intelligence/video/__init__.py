"""Phase 1 promotion: video inspection, VideoSource, and dataset validation."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "01_video_inspection"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from inspect_video import (  # noqa: E402
    CONTACT_SHEET_COLS,
    MAX_SAMPLE_COUNT,
    SAMPLE_COUNT,
    THUMB_SIZE,
    build_contact_sheet,
    inspect_source,
    make_thumbnail,
    run_inspection,
    sample_frame_indices,
    source_name,
    write_image,
    write_json,
)
from validate_dataset import (  # noqa: E402
    MOT17_SPECS,
    check_clip_entry,
    check_mot17_entry,
    mot17_base_sequence,
    validate_video_info,
)
from video_source import VideoMetadata, VideoSource  # noqa: E402

__all__ = [
    "CONTACT_SHEET_COLS",
    "MAX_SAMPLE_COUNT",
    "MOT17_SPECS",
    "SAMPLE_COUNT",
    "THUMB_SIZE",
    "VideoMetadata",
    "VideoSource",
    "build_contact_sheet",
    "check_clip_entry",
    "check_mot17_entry",
    "inspect_source",
    "make_thumbnail",
    "mot17_base_sequence",
    "run_inspection",
    "sample_frame_indices",
    "source_name",
    "validate_video_info",
    "write_image",
    "write_json",
]
