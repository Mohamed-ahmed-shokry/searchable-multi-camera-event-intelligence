"""Phase 7 promotion: best-frame selection, padded person crops, and event snapshots."""

from __future__ import annotations

from event_intelligence.evidence.extract_visual_evidence import (
    EVIDENCE_EVENTS_CSV_CLIPS_HEADER,
    EVIDENCE_EVENTS_CSV_HEADER,
    VALID_BEST_FRAME_STRATEGIES,
    BestFrameSelection,
    EventClipRecord,
    EvidenceConfig,
    EvidenceEventRecord,
    extract_person_crop,
    index_track_detections,
    load_evidence_config,
    read_evidence_events_csv,
    render_event_snapshot,
    render_clip_frame_overlay,
    render_evidence_contact_sheet,
    run_visual_evidence,
    save_evidence_config,
    select_best_frame_for_event,
    verify_evidence_records,
    write_evidence_events_csv,
)

__all__ = [
    "EVIDENCE_EVENTS_CSV_CLIPS_HEADER",
    "EVIDENCE_EVENTS_CSV_HEADER",
    "VALID_BEST_FRAME_STRATEGIES",
    "BestFrameSelection",
    "EventClipRecord",
    "EvidenceConfig",
    "EvidenceEventRecord",
    "extract_person_crop",
    "index_track_detections",
    "load_evidence_config",
    "read_evidence_events_csv",
    "render_event_snapshot",
    "render_clip_frame_overlay",
    "render_evidence_contact_sheet",
    "run_visual_evidence",
    "save_evidence_config",
    "select_best_frame_for_event",
    "verify_evidence_records",
    "write_evidence_events_csv",
]
