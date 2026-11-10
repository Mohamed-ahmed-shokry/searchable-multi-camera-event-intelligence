"""Phase 1 promotion: video inspection, VideoSource, and dataset validation."""

from __future__ import annotations

from event_intelligence.video.inspect_video import (
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
from event_intelligence.video.validate_dataset import (
    MOT17_SPECS,
    check_clip_entry,
    check_mot17_entry,
    mot17_base_sequence,
    validate_video_info,
)
from event_intelligence.video.video_source import VideoMetadata, VideoSource

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
