"""Rule-based and statistical behavioral insight detectors."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Mapping, Sequence
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

    dwells_by_zone: dict[str, list[float]] = {}
    for ev in dwell_events:
        zone = str(ev.zone_id)
        ev_dur = float(getattr(ev, "duration_seconds", getattr(ev, "duration", 0.0)))
        dwells_by_zone.setdefault(zone, []).append(ev_dur)

    medians_by_zone = {
        zone: statistics.median(durations)
        for zone, durations in dwells_by_zone.items()
        if len(durations) >= 2
    }

    insights: list[InsightRecord] = []
    idx = start_index

    for ev in dwell_events:
        zone = str(ev.zone_id)
        dur = float(getattr(ev, "duration_seconds", getattr(ev, "duration", 0.0)))
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
            track_id=(
                int(ev.track_id) if getattr(ev, "track_id", None) is not None else None
            ),
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


def detect_speed_anomaly_insights(
    trajectories: Mapping[int, Sequence[Any]],
    config: InsightConfig,
    *,
    camera_id: str = "default",
    start_index: int = 1,
) -> list[InsightRecord]:
    """Detect rapid movement, sprinting, or high-velocity anomalies along trajectories."""
    threshold = config.speed_anomaly_threshold_px_s
    min_frames = config.min_speed_anomaly_frames
    insights: list[InsightRecord] = []
    idx = start_index

    for track_id, points in trajectories.items():
        if len(points) < min_frames + 1:
            continue

        sorted_pts = sorted(
            points,
            key=lambda p: (
                getattr(p, "frame_id", 0),
                getattr(p, "timestamp", 0.0),
            ),
        )

        current_burst: list[tuple[Any, float]] = []
        bursts: list[list[tuple[Any, float]]] = []

        for p1, p2 in itertools.pairwise(sorted_pts):
            t1 = float(getattr(p1, "timestamp", 0.0))
            t2 = float(getattr(p2, "timestamp", 0.0))
            dt = t2 - t1
            if dt <= 1e-4:
                continue

            x1, y1 = float(getattr(p1, "x", 0.0)), float(getattr(p1, "y", 0.0))
            x2, y2 = float(getattr(p2, "x", 0.0)), float(getattr(p2, "y", 0.0))
            dist = math.hypot(x2 - x1, y2 - y1)
            speed = dist / dt

            if speed >= threshold:
                if not current_burst:
                    current_burst.append((p1, speed))
                current_burst.append((p2, speed))
            else:
                if len(current_burst) >= min_frames + 1:
                    bursts.append(list(current_burst))
                current_burst.clear()

        if len(current_burst) >= min_frames + 1:
            bursts.append(list(current_burst))

        for burst in bursts:
            burst_pts = [b[0] for b in burst]
            burst_speeds = [b[1] for b in burst]
            max_spd = max(burst_speeds)
            mean_spd = sum(burst_speeds) / len(burst_speeds)
            t_start = float(getattr(burst_pts[0], "timestamp", 0.0))
            t_end = float(getattr(burst_pts[-1], "timestamp", 0.0))
            dur = max(0.0, t_end - t_start)

            if max_spd >= 2.5 * threshold:
                sev = Severity.CRITICAL.value
            elif max_spd >= 1.5 * threshold:
                sev = Severity.HIGH.value
            else:
                sev = Severity.MEDIUM.value

            rec = InsightRecord(
                insight_id=f"ins_{camera_id}_{idx:04d}",
                camera_id=camera_id,
                insight_type=InsightType.HIGH_SPEED.value,
                severity=sev,
                track_id=int(track_id),
                zone_id=None,
                start_time=t_start,
                end_time=t_end,
                duration=dur,
                description=(
                    f"Track {track_id} rapid movement / speed anomaly: "
                    f"max speed {max_spd:.1f} px/s (threshold {threshold:.1f} px/s) "
                    f"over {dur:.2f}s ({len(burst_pts)} frames)"
                ),
                details={
                    "max_speed_px_s": round(max_spd, 1),
                    "mean_speed_px_s": round(mean_spd, 1),
                    "threshold_px_s": threshold,
                    "burst_frames": len(burst_pts),
                },
            )
            insights.append(rec)
            idx += 1

    return insights


def detect_stationary_and_drop_insights(
    events: Sequence[Any],
    trajectories: Mapping[int, Sequence[Any]],
    config: InsightConfig,
    *,
    traces: Sequence[Any] | None = None,
    camera_id: str = "default",
    start_index: int = 1,
) -> list[InsightRecord]:
    """Detect stationary entities and unexpected track drop/loss inside sensitive zones."""
    insights: list[InsightRecord] = []
    idx = start_index

    # 1. Detect track loss in zone from traces or events
    lost_records: list[tuple[int, str, float, float, Any]] = []
    if traces is not None:
        for tr in traces:
            if getattr(tr, "track_lost_in_zone", False):
                t_id = int(getattr(tr, "track_id", 0))
                z_id = str(getattr(tr, "zone_id", ""))
                t_in = float(getattr(tr, "entry_timestamp", 0.0))
                t_out = float(getattr(tr, "exit_timestamp", t_in))
                lost_records.append((t_id, z_id, t_in, t_out, tr))

    # Also inspect events if event_type explicitly mentions track loss
    for ev in events:
        ev_type = str(getattr(ev, "event_type", ""))
        if "track_lost" in ev_type or "lost_in_zone" in ev_type:
            t_id = int(getattr(ev, "track_id", 0))
            z_id = str(getattr(ev, "zone_id", ""))
            t_in = float(getattr(ev, "start_time", 0.0))
            t_out = float(getattr(ev, "end_time", t_in))
            if not any(r[0] == t_id and r[1] == z_id for r in lost_records):
                lost_records.append((t_id, z_id, t_in, t_out, ev))

    for t_id, z_id, t_in, t_out, src_obj in lost_records:
        is_restricted = z_id in config.restricted_zones
        dur = max(0.0, t_out - t_in)
        sev = Severity.CRITICAL.value if is_restricted else Severity.HIGH.value

        snap_path = getattr(src_obj, "evidence_snapshot_path", None) or getattr(
            src_obj, "snapshot_path", None
        )
        crop_path = getattr(src_obj, "evidence_crop_path", None) or getattr(
            src_obj, "crop_path", None
        )

        insights.append(
            InsightRecord(
                insight_id=f"ins_{camera_id}_{idx:04d}",
                camera_id=camera_id,
                insight_type=InsightType.TRACK_DROP_IN_ZONE.value,
                severity=sev,
                track_id=t_id,
                zone_id=z_id,
                start_time=t_in,
                end_time=t_out,
                duration=dur,
                description=(
                    f"Track {t_id} terminated / lost inside {'restricted ' if is_restricted else ''}"
                    f"zone '{z_id}' at {t_out:.2f}s"
                ),
                evidence_snapshot_path=str(snap_path) if snap_path else None,
                evidence_crop_path=str(crop_path) if crop_path else None,
                details={
                    "is_restricted_zone": is_restricted,
                    "resolution_reason": "track_lost_in_zone",
                },
            )
        )
        idx += 1

    # 2. Detect stationary entities inside zones
    dwell_events = [
        ev
        for ev in events
        if getattr(ev, "event_type", None) in ("dwell_time_event", "zone_dwell")
        and getattr(ev, "zone_id", None)
    ]

    for ev in dwell_events:
        t_id = int(getattr(ev, "track_id", 0))
        z_id = str(getattr(ev, "zone_id", ""))
        dur = float(
            getattr(ev, "duration", 0.0) or getattr(ev, "duration_seconds", 0.0)
        )
        if dur < config.stationary_min_seconds:
            continue

        pts = [
            p
            for p in trajectories.get(t_id, [])
            if float(getattr(ev, "start_time", 0.0))
            <= float(getattr(p, "timestamp", 0.0))
            <= float(getattr(ev, "end_time", 0.0))
        ]
        if len(pts) < 3:
            continue

        speeds: list[float] = []
        for p1, p2 in itertools.pairwise(pts):
            dt = float(getattr(p2, "timestamp", 0.0)) - float(
                getattr(p1, "timestamp", 0.0)
            )
            if dt > 1e-4:
                dist = math.hypot(
                    float(getattr(p2, "x", 0.0)) - float(getattr(p1, "x", 0.0)),
                    float(getattr(p2, "y", 0.0)) - float(getattr(p1, "y", 0.0)),
                )
                speeds.append(dist / dt)

        avg_spd = sum(speeds) / len(speeds) if speeds else 0.0
        if avg_spd <= config.stationary_speed_threshold_px_s:
            is_restricted = z_id in config.restricted_zones
            sev = Severity.HIGH.value if is_restricted else Severity.MEDIUM.value

            snap_path = getattr(ev, "evidence_snapshot_path", None) or getattr(
                ev, "snapshot_path", None
            )
            crop_path = getattr(ev, "evidence_crop_path", None) or getattr(
                ev, "crop_path", None
            )

            insights.append(
                InsightRecord(
                    insight_id=f"ins_{camera_id}_{idx:04d}",
                    camera_id=camera_id,
                    insight_type=InsightType.STATIONARY_IN_ZONE.value,
                    severity=sev,
                    track_id=t_id,
                    zone_id=z_id,
                    start_time=float(ev.start_time),
                    end_time=float(ev.end_time),
                    duration=dur,
                    description=(
                        f"Track {t_id} remained stationary in {'restricted ' if is_restricted else ''}"
                        f"zone '{z_id}' for {dur:.1f}s (avg speed {avg_spd:.1f} px/s)"
                    ),
                    evidence_snapshot_path=str(snap_path) if snap_path else None,
                    evidence_crop_path=str(crop_path) if crop_path else None,
                    details={
                        "avg_speed_px_s": round(avg_spd, 2),
                        "speed_threshold_px_s": config.stationary_speed_threshold_px_s,
                        "is_restricted_zone": is_restricted,
                    },
                )
            )
            idx += 1

    return insights


def detect_cross_camera_transitions(
    events_by_camera: Mapping[str, Sequence[Any]],
    config: InsightConfig,
    *,
    start_index: int = 1,
) -> list[InsightRecord]:
    """Detect plausible entity transitions and temporal handoffs across multiple camera views."""
    cams = list(events_by_camera.keys())
    if len(cams) < 2:
        return []

    insights: list[InsightRecord] = []
    idx = start_index

    # Gather departures (zone exits or dwell event completions)
    departures: list[
        tuple[str, int, str, float, Any]
    ] = []  # (cam, track_id, zone, t_exit, obj)
    arrivals: list[
        tuple[str, int, str, float, Any]
    ] = []  # (cam, track_id, zone, t_entry, obj)

    for cam, evs in events_by_camera.items():
        for ev in evs:
            t_id = int(getattr(ev, "track_id", 0))
            z_id = str(getattr(ev, "zone_id", ""))
            ev_type = str(getattr(ev, "event_type", ""))

            if ev_type in ("zone_exit", "dwell_time_event"):
                t_exit = float(getattr(ev, "end_time", 0.0))
                departures.append((cam, t_id, z_id, t_exit, ev))
            if ev_type in ("zone_entry", "dwell_time_event"):
                t_entry = float(getattr(ev, "start_time", 0.0))
                arrivals.append((cam, t_id, z_id, t_entry, ev))

    for cam_a, t_a, z_a, t_dep, dep_obj in departures:
        for cam_b, t_b, z_b, t_arr, arr_obj in arrivals:
            if cam_a == cam_b:
                continue

            delta = t_arr - t_dep
            if (
                config.cross_camera_min_seconds
                <= delta
                <= config.cross_camera_max_seconds
            ):
                snap_path = getattr(arr_obj, "evidence_snapshot_path", None) or getattr(
                    arr_obj, "snapshot_path", None
                )
                crop_path = getattr(arr_obj, "evidence_crop_path", None) or getattr(
                    arr_obj, "crop_path", None
                )

                rec = InsightRecord(
                    insight_id=f"ins_multicam_{idx:04d}",
                    camera_id=f"{cam_a}->{cam_b}",
                    insight_type=InsightType.CROSS_CAMERA_TRANSITION.value,
                    severity=Severity.INFO.value,
                    track_id=t_b,
                    zone_id=f"{z_a}->{z_b}" if z_a and z_b else None,
                    start_time=t_dep,
                    end_time=t_arr,
                    duration=round(delta, 2),
                    description=(
                        f"Cross-camera transition from '{cam_a}' (track {t_a}) "
                        f"to '{cam_b}' (track {t_b}) in {delta:.2f}s"
                    ),
                    evidence_snapshot_path=str(snap_path) if snap_path else None,
                    evidence_crop_path=str(crop_path) if crop_path else None,
                    details={
                        "from_camera": cam_a,
                        "from_track_id": t_a,
                        "from_zone": z_a,
                        "to_camera": cam_b,
                        "to_track_id": t_b,
                        "to_zone": z_b,
                        "transition_seconds": round(delta, 2),
                    },
                )
                insights.append(rec)
                idx += 1

    return insights


def generate_all_insights(
    events: Sequence[Any],
    trajectories: Mapping[int, Sequence[Any]],
    config: InsightConfig | None = None,
    *,
    traces: Sequence[Any] | None = None,
    camera_id: str = "default",
    events_by_camera: Mapping[str, Sequence[Any]] | None = None,
) -> list[InsightRecord]:
    """Execute all behavioral detectors and return aggregated, chronologically ordered insights."""
    cfg = config if config is not None else InsightConfig()
    all_insights: list[InsightRecord] = []

    # 1. Loitering
    loitering = detect_loitering_insights(
        events, cfg, camera_id=camera_id, start_index=len(all_insights) + 1
    )
    all_insights.extend(loitering)

    # 2. Wrong-way
    wrong_way = detect_wrong_way_insights(
        events, cfg, camera_id=camera_id, start_index=len(all_insights) + 1
    )
    all_insights.extend(wrong_way)

    # 3. High-speed
    speed_insights = detect_speed_anomaly_insights(
        trajectories, cfg, camera_id=camera_id, start_index=len(all_insights) + 1
    )
    all_insights.extend(speed_insights)

    # 4. Stationary & track drop
    stationary_and_drop = detect_stationary_and_drop_insights(
        events,
        trajectories,
        cfg,
        traces=traces,
        camera_id=camera_id,
        start_index=len(all_insights) + 1,
    )
    all_insights.extend(stationary_and_drop)

    # 5. Cross-camera transitions (if multi-camera dictionary provided)
    if events_by_camera and len(events_by_camera) > 1:
        multicam_transitions = detect_cross_camera_transitions(
            events_by_camera, cfg, start_index=len(all_insights) + 1
        )
        all_insights.extend(multicam_transitions)

    # Sort deterministically by start_time, severity rank, insight_id
    severity_order = {
        Severity.CRITICAL.value: 0,
        Severity.HIGH.value: 1,
        Severity.MEDIUM.value: 2,
        Severity.LOW.value: 3,
        Severity.INFO.value: 4,
    }
    all_insights.sort(
        key=lambda ins: (
            ins.start_time,
            severity_order.get(ins.severity, 5),
            ins.insight_id,
        )
    )
    return all_insights
