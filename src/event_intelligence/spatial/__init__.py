"""Phase 5 promotion: polygon zones, directed tripwire lines, and spatial signal reasoning."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "05_zone_line_reasoning"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from reason_zones_lines import (
    LINE_PALETTE_BGR,
    VALID_DIRECTIONS,
    VALID_SIGNAL_TYPES,
    ZONE_LINE_EVENTS_CSV_HEADER,
    ZONE_PALETTE_BGR,
    LineSpec,
    SceneConfig,
    ZoneLineSignal,
    ZoneSpec,
    analyze_boundary_flicker,
    check_line_crossing,
    detect_line_signals,
    detect_zone_signals,
    draw_zones_and_lines,
    evaluate_scene_signals,
    load_scene_config,
    point_in_zone,
    read_zone_line_events_csv,
    run_zone_line_reasoning,
    save_scene_config,
    write_zone_line_events_csv,
)

__all__ = [
    "LINE_PALETTE_BGR",
    "VALID_DIRECTIONS",
    "VALID_SIGNAL_TYPES",
    "ZONE_LINE_EVENTS_CSV_HEADER",
    "ZONE_PALETTE_BGR",
    "LineSpec",
    "SceneConfig",
    "ZoneLineSignal",
    "ZoneSpec",
    "analyze_boundary_flicker",
    "check_line_crossing",
    "detect_line_signals",
    "detect_zone_signals",
    "draw_zones_and_lines",
    "evaluate_scene_signals",
    "load_scene_config",
    "point_in_zone",
    "read_zone_line_events_csv",
    "run_zone_line_reasoning",
    "save_scene_config",
    "write_zone_line_events_csv",
]
