"""Phase 3 promotion: ByteTrack / BoT-SORT person tracking and CLEAR-MOT evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "03_person_tracking"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evaluate_mot import (
    box_iou,
    compute_iou_matrix,
    evaluate_tracking,
    load_mot17_gt,
)
from track_people import (
    TRACK_COLOR_PALETTE,
    TRACK_SUMMARY_CSV_HEADER,
    TRACKS_CSV_HEADER,
    FrameTracks,
    PersonTracker,
    TrackDetection,
    TrackSummary,
    analyze_track_anomalies,
    compute_track_summaries,
    create_video_writer,
    draw_tracks,
    read_track_summary_csv,
    read_tracks_csv,
    run_tracking,
    track_color,
    write_track_summary_csv,
    write_tracks_csv,
)

__all__ = [
    "TRACKS_CSV_HEADER",
    "TRACK_COLOR_PALETTE",
    "TRACK_SUMMARY_CSV_HEADER",
    "FrameTracks",
    "PersonTracker",
    "TrackDetection",
    "TrackSummary",
    "analyze_track_anomalies",
    "box_iou",
    "compute_iou_matrix",
    "compute_track_summaries",
    "create_video_writer",
    "draw_tracks",
    "evaluate_tracking",
    "load_mot17_gt",
    "read_track_summary_csv",
    "read_tracks_csv",
    "run_tracking",
    "track_color",
    "write_track_summary_csv",
    "write_tracks_csv",
]
