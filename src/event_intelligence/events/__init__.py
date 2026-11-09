"""Phase 6 promotion: stateful zone dwell-time state machine with hysteresis."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "06_dwell_time_events"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from generate_dwell_events import (
    EVENTS_CSV_HEADER,
    VALID_EVENT_TYPES,
    VALID_STAY_STATES,
    EventConfig,
    EventRecord,
    StayDebugTrace,
    ZoneDwellStateMachine,
    draw_dwell_events_overlay,
    evaluate_dwell_events,
    load_events_config,
    read_events_csv,
    run_dwell_time_events,
    save_dwell_examples,
    save_events_config,
    write_event_debug_logs,
    write_events_csv,
)

__all__ = [
    "EVENTS_CSV_HEADER",
    "VALID_EVENT_TYPES",
    "VALID_STAY_STATES",
    "EventConfig",
    "EventRecord",
    "StayDebugTrace",
    "ZoneDwellStateMachine",
    "draw_dwell_events_overlay",
    "evaluate_dwell_events",
    "load_events_config",
    "read_events_csv",
    "run_dwell_time_events",
    "save_dwell_examples",
    "save_events_config",
    "write_event_debug_logs",
    "write_events_csv",
]
