"""Phase 3 promotion: ByteTrack / BoT-SORT person tracking and CLEAR-MOT evaluation."""

from __future__ import annotations

from event_intelligence.tracking.compare_trackers import (
    DEFAULT_TRACKERS,
    compare_trackers,
)
from event_intelligence.tracking.evaluate_mot import (
    box_iou,
    compute_iou_matrix,
    evaluate_tracking,
    load_mot17_gt,
)
from event_intelligence.tracking.track_people import (
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
    "DEFAULT_TRACKERS",
    "TRACKS_CSV_HEADER",
    "TRACK_COLOR_PALETTE",
    "TRACK_SUMMARY_CSV_HEADER",
    "FrameTracks",
    "PersonTracker",
    "TrackDetection",
    "TrackSummary",
    "analyze_track_anomalies",
    "box_iou",
    "compare_trackers",
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
