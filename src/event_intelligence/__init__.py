"""Searchable Multi-Camera Event Intelligence core CV package (Phase 11).

Consolidates proven capabilities from cv_projects/01..10 into one coherent,
runnable package:
- video: VideoSource, frame inspection, and dataset validation
- detection: YOLO11 person detection and CSV I/O
- tracking: ByteTrack / BoT-SORT tracking and CLEAR-MOT evaluation
- trajectories: ground-plane trajectory extraction and smoothing
- spatial: polygon zone and directed tripwire line reasoning
- events: stateful zone dwell-time state machine with hysteresis
- evidence: best-frame event snapshots and padded person crops
- analytics: spatial density heatmaps and zone/line count summaries
- evaluation: quantitative MOT17 and qualitative surveillance evaluation
- storage: SQLite intelligence.db ingestion and CLI/programmatic search
- pipeline: single-command end-to-end CV pipeline runner
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
