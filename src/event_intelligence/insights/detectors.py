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
