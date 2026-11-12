"""Rule-based and statistical behavioral insight detectors."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from event_intelligence.insights.types import (
    InsightConfig,
    InsightRecord,
    InsightType,
    Severity,
)


def detect_loitering_insights(
    events: Sequence[Any],
    config: InsightConfig,
    *,
    camera_id: str = "default",
    start_index: int = 1,
) -> list[InsightRecord]:
    """Detect loitering and prolonged dwell anomalies exceeding normal or configured thresholds."""
    dwell_events = [
        ev
        for ev in events
        if getattr(ev, "event_type", None) in ("dwell_time_event", "zone_dwell")
        and getattr(ev, "zone_id", None)
    ]
    if not dwell_events:
        return []

    # Calculate median dwell per zone
    dwells_by_zone: dict[str, list[float]] = {}
    for ev in dwell_events:
        zone = str(ev.zone_id)
        dwells_by_zone.setdefault(zone, []).append(float(ev.duration))

    medians_by_zone = {
        zone: statistics.median(durations)
        for zone, durations in dwells_by_zone.items()
        if len(durations) >= 2
    }

    insights: list[InsightRecord] = []
    idx = start_index

    for ev in dwell_events:
        zone = str(ev.zone_id)
        dur = float(ev.duration)
        zone_threshold = config.per_zone_loitering_seconds.get(
            zone, config.min_loitering_seconds
        )
        median_dwell = medians_by_zone.get(zone, 0.0)
        is_statistical_loiter = (
            median_dwell > 0.0
            and dur >= config.loitering_factor * median_dwell
            and dur >= config.min_loitering_seconds
        )

        is_loiter = dur >= zone_threshold or is_statistical_loiter
        if not is_loiter:
            continue

        is_restricted = zone in config.restricted_zones
        if is_restricted:
            severity = (
                Severity.CRITICAL.value
                if dur >= 1.5 * zone_threshold
                else Severity.HIGH.value
            )
        elif dur >= 2.0 * zone_threshold:
            severity = Severity.HIGH.value
        else:
            severity = Severity.MEDIUM.value

        snap_path = getattr(ev, "evidence_snapshot_path", None) or getattr(
            ev, "snapshot_path", None
        )
        crop_path = getattr(ev, "evidence_crop_path", None) or getattr(
            ev, "crop_path", None
        )

        record = InsightRecord(
            insight_id=f"ins_{camera_id}_{idx:04d}",
            camera_id=camera_id,
            insight_type=InsightType.LOITERING.value,
            severity=severity,
            track_id=int(ev.track_id)
            if getattr(ev, "track_id", None) is not None
            else None,
            zone_id=zone,
            start_time=float(ev.start_time),
            end_time=float(ev.end_time),
            duration=dur,
            description=(
                f"Track {ev.track_id} loitering in {'restricted ' if is_restricted else ''}"
                f"zone '{zone}' for {dur:.1f}s (threshold {zone_threshold:.1f}s)"
            ),
            evidence_snapshot_path=str(snap_path) if snap_path else None,
            evidence_crop_path=str(crop_path) if crop_path else None,
            details={
                "zone_threshold_seconds": round(zone_threshold, 2),
                "zone_median_dwell_seconds": round(median_dwell, 2),
                "is_restricted_zone": is_restricted,
            },
        )
        insights.append(record)
        idx += 1

    return insights


def detect_wrong_way_insights(
    events: Sequence[Any],
    config: InsightConfig,
    *,
    camera_id: str = "default",
    start_index: int = 1,
) -> list[InsightRecord]:
    """Detect directional corridor violations and reverse line crossings."""
    if not config.allowed_line_directions and not config.disallowed_line_crossings:
        return []

    line_events = [
        ev
        for ev in events
        if getattr(ev, "event_type", None) == "line_crossing"
        and getattr(ev, "zone_id", None)
    ]
    if not line_events:
        return []

    insights: list[InsightRecord] = []
    idx = start_index

    for ev in line_events:
        full_spec = str(ev.zone_id)
        line_id, sep, direction = full_spec.partition(":")
        if not sep:
            line_id = full_spec
            direction = ""

        is_disallowed = full_spec in config.disallowed_line_crossings
        is_wrong_dir = (
            line_id in config.allowed_line_directions
            and direction != ""
            and config.allowed_line_directions[line_id] != direction
        )

        if not (is_disallowed or is_wrong_dir):
            continue

        is_restricted = (
            line_id in config.restricted_zones or full_spec in config.restricted_zones
        )
        severity = Severity.HIGH.value if is_restricted else Severity.MEDIUM.value

        snap_path = getattr(ev, "evidence_snapshot_path", None) or getattr(
            ev, "snapshot_path", None
        )
        crop_path = getattr(ev, "evidence_crop_path", None) or getattr(
            ev, "crop_path", None
        )

        expected = config.allowed_line_directions.get(line_id, "disallowed")
        record = InsightRecord(
            insight_id=f"ins_{camera_id}_{idx:04d}",
            camera_id=camera_id,
            insight_type=InsightType.WRONG_WAY.value,
            severity=severity,
            track_id=(
                int(ev.track_id) if getattr(ev, "track_id", None) is not None else None
            ),
            zone_id=full_spec,
            start_time=float(ev.start_time),
            end_time=float(ev.end_time),
            duration=float(
                getattr(ev, "duration", 0.0) or getattr(ev, "duration_seconds", 0.0)
            ),
            description=(
                f"Track {ev.track_id} wrong-way crossing on '{line_id}' "
                f"in direction '{direction}' (expected '{expected}')"
            ),
            evidence_snapshot_path=str(snap_path) if snap_path else None,
            evidence_crop_path=str(crop_path) if crop_path else None,
            details={
                "line_id": line_id,
                "direction": direction,
                "expected_direction": expected,
                "is_restricted": is_restricted,
            },
        )
        insights.append(record)
        idx += 1

    return insights
