"""Multi-camera spatial-temporal correlation engine and topology helpers."""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path

import yaml

from event_intelligence.multicam.types import (
    CameraPairFlow,
    CameraTopology,
    CameraTransitionRule,
    CoActivityWindow,
    CrossCameraJourney,
    CrossCameraTransition,
    MultiCameraCorrelationReport,
    TrackEndpoint,
)


def load_camera_topology(path: Path | str) -> CameraTopology:
    """Load a CameraTopology specification from a YAML file."""
    topology_path = Path(path)
    if not topology_path.exists():
        raise FileNotFoundError(f"Camera topology YAML not found: {topology_path}")
    raw = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    if raw is None:
        return CameraTopology()
    if not isinstance(raw, dict):
        raise TypeError(
            f"Camera topology YAML must contain a mapping, got {type(raw).__name__}"
        )
    return CameraTopology.from_dict(raw)


def save_camera_topology(topology: CameraTopology, path: Path | str) -> Path:
    """Write a CameraTopology specification to a YAML file."""
    topology_path = Path(path)
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.write_text(
        yaml.safe_dump(topology.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return topology_path


def infer_default_topology(
    camera_ids: Sequence[str],
    *,
    min_transit_seconds: float = 0.0,
    max_transit_seconds: float = 30.0,
    max_matches_per_track: int = 1,
) -> CameraTopology:
    """Build a bidirectional all-pairs topology for cameras when no YAML is provided."""
    unique_cameras = sorted({str(cam) for cam in camera_ids if str(cam)})
    rules: list[CameraTransitionRule] = []
    for src in unique_cameras:
        for dst in unique_cameras:
            if src == dst:
                continue
            rules.append(
                CameraTransitionRule(
                    from_camera=src,
                    to_camera=dst,
                    min_transit_seconds=min_transit_seconds,
                    max_transit_seconds=max_transit_seconds,
                    label=f"{src} -> {dst}",
                )
            )
    return CameraTopology(
        name="inferred_all_pairs",
        description=(
            f"Auto-inferred all-pairs topology across {len(unique_cameras)} camera(s)."
        ),
        transitions=tuple(rules),
        default_min_transit_seconds=min_transit_seconds,
        default_max_transit_seconds=max_transit_seconds,
        max_matches_per_track=max_matches_per_track,
    )


def extract_track_endpoints(
    db_path: Path | str,
    *,
    camera_time_offsets: Mapping[str, float] | None = None,
) -> list[TrackEndpoint]:
    """Extract per-camera track start/end times, coordinates, and zones from SQLite."""
    resolved = Path(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite intelligence DB not found: {resolved}")

    offsets = dict(camera_time_offsets or {})

    with closing(sqlite3.connect(str(resolved))) as conn:
        conn.row_factory = sqlite3.Row

        fps_by_cam: dict[str, float] = {}
        for row in conn.execute("SELECT camera_id, fps FROM cameras"):
            cam_fps = float(row["fps"]) if row["fps"] else 30.0
            fps_by_cam[str(row["camera_id"])] = cam_fps if cam_fps > 0 else 30.0

        traj_points: dict[tuple[str, int], list[tuple[float, float, float]]] = (
            defaultdict(list)
        )
        for row in conn.execute(
            "SELECT camera_id, track_id, timestamp, x, y FROM trajectories "
            "ORDER BY camera_id, track_id, frame_id"
        ):
            key = (str(row["camera_id"]), int(row["track_id"]))
            traj_points[key].append(
                (float(row["timestamp"]), float(row["x"]), float(row["y"]))
            )

        events_by_track: dict[tuple[str, int], list[tuple[float, float, str]]] = (
            defaultdict(list)
        )
        for row in conn.execute(
            "SELECT camera_id, track_id, start_time, end_time, zone_id FROM events "
            "ORDER BY camera_id, track_id, start_time, end_time"
        ):
            key = (str(row["camera_id"]), int(row["track_id"]))
            events_by_track[key].append(
                (float(row["start_time"]), float(row["end_time"]), str(row["zone_id"]))
            )

        endpoints: list[TrackEndpoint] = []
        seen_keys: set[tuple[str, int]] = set()
        for row in conn.execute(
            "SELECT camera_id, track_id, start_frame, end_frame, duration_seconds "
            "FROM tracks ORDER BY camera_id, track_id"
        ):
            cam_id = str(row["camera_id"])
            trk_id = int(row["track_id"])
            key = (cam_id, trk_id)
            seen_keys.add(key)

            offset = float(offsets.get(cam_id, 0.0))
            pts = traj_points.get(key, [])
            fps = fps_by_cam.get(cam_id, 30.0)

            if pts:
                start_t = pts[0][0] + offset
                end_t = pts[-1][0] + offset
                start_x, start_y = pts[0][1], pts[0][2]
                end_x, end_y = pts[-1][1], pts[-1][2]
            else:
                start_t = (int(row["start_frame"]) / fps) + offset
                end_t = (int(row["end_frame"]) / fps) + offset
                start_x = start_y = end_x = end_y = None

            duration = max(0.0, float(row["duration_seconds"]))
            if end_t < start_t:
                end_t = start_t + duration

            evts = events_by_track.get(key, [])
            entry_zones = _extract_ordered_zones(evts, from_end=False)
            exit_zones = _extract_ordered_zones(evts, from_end=True)

            endpoints.append(
                TrackEndpoint(
                    camera_id=cam_id,
                    track_id=trk_id,
                    start_time=start_t,
                    end_time=end_t,
                    duration_seconds=duration,
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    entry_zones=entry_zones,
                    exit_zones=exit_zones,
                    event_count=len(evts),
                )
            )

        # Fallback for DBs that have trajectories without tracks rows
        for key, pts in sorted(traj_points.items()):
            if key in seen_keys or not pts:
                continue
            cam_id, trk_id = key
            offset = float(offsets.get(cam_id, 0.0))
            start_t = pts[0][0] + offset
            end_t = pts[-1][0] + offset
            evts = events_by_track.get(key, [])
            endpoints.append(
                TrackEndpoint(
                    camera_id=cam_id,
                    track_id=trk_id,
                    start_time=start_t,
                    end_time=end_t,
                    duration_seconds=max(0.0, end_t - start_t),
                    start_x=pts[0][1],
                    start_y=pts[0][2],
                    end_x=pts[-1][1],
                    end_y=pts[-1][2],
                    entry_zones=_extract_ordered_zones(evts, from_end=False),
                    exit_zones=_extract_ordered_zones(evts, from_end=True),
                    event_count=len(evts),
                )
            )

    endpoints.sort(key=lambda ep: (ep.start_time, ep.camera_id, ep.track_id))
    return endpoints


def _extract_ordered_zones(
    events: Sequence[tuple[float, float, str]],
    *,
    from_end: bool,
) -> tuple[str, ...]:
    """Return unique non-empty zone IDs ordered from track entry or track exit."""
    if not events:
        return ()
    ordered = (
        sorted(events, key=lambda item: (item[1], item[0]), reverse=True)
        if from_end
        else sorted(events, key=lambda item: (item[0], item[1]))
    )
    seen: list[str] = []
    for _, _, zone_id in ordered:
        if zone_id and zone_id not in seen:
            seen.append(zone_id)
    return tuple(seen)


def correlate_cross_camera_transitions(
    endpoints: Sequence[TrackEndpoint],
    topology: CameraTopology,
    *,
    min_confidence: float = 0.25,
) -> list[CrossCameraTransition]:
    """Match candidate cross-camera track handoffs under a CameraTopology."""
    by_camera: dict[str, list[TrackEndpoint]] = defaultdict(list)
    for ep in endpoints:
        by_camera[ep.camera_id].append(ep)

    candidates: list[
        tuple[float, float, TrackEndpoint, TrackEndpoint, CameraTransitionRule, float]
    ] = []

    for rule in topology.transitions:
        src_list = by_camera.get(rule.from_camera, [])
        dst_list = by_camera.get(rule.to_camera, [])
        if not src_list or not dst_list:
            continue

        target = (
            rule.expected_transit_seconds
            if rule.expected_transit_seconds is not None
            else 0.5 * (rule.min_transit_seconds + rule.max_transit_seconds)
        )
        half_span = max(
            0.5 * (rule.max_transit_seconds - rule.min_transit_seconds), 1e-6
        )

        for src_ep in src_list:
            if rule.from_zone and rule.from_zone not in src_ep.exit_zones:
                continue
            for dst_ep in dst_list:
                if rule.to_zone and rule.to_zone not in dst_ep.entry_zones:
                    continue
                transit = dst_ep.start_time - src_ep.end_time
                if (
                    transit < rule.min_transit_seconds
                    or transit > rule.max_transit_seconds
                ):
                    continue

                confidence = _score_transition_candidate(
                    transit=transit,
                    target=target,
                    half_span=half_span,
                    rule=rule,
                    src_ep=src_ep,
                    dst_ep=dst_ep,
                )
                if confidence < min_confidence:
                    continue
                candidates.append(
                    (
                        confidence,
                        abs(transit - target),
                        src_ep,
                        dst_ep,
                        rule,
                        transit,
                    )
                )

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2].end_time,
            item[2].camera_id,
            item[2].track_id,
            item[3].camera_id,
            item[3].track_id,
        )
    )

    max_matches = max(1, int(topology.max_matches_per_track))
    out_counts: dict[tuple[str, int], int] = defaultdict(int)
    in_counts: dict[tuple[str, int], int] = defaultdict(int)
    seen_pairs: set[tuple[str, int, str, int]] = set()
    selected: list[
        tuple[float, TrackEndpoint, TrackEndpoint, CameraTransitionRule, float]
    ] = []

    for confidence, _delta, src_ep, dst_ep, rule, transit in candidates:
        src_key = (src_ep.camera_id, src_ep.track_id)
        dst_key = (dst_ep.camera_id, dst_ep.track_id)
        pair_key = (
            src_ep.camera_id,
            src_ep.track_id,
            dst_ep.camera_id,
            dst_ep.track_id,
        )
        if pair_key in seen_pairs:
            continue
        if out_counts[src_key] >= max_matches or in_counts[dst_key] >= max_matches:
            continue
        seen_pairs.add(pair_key)
        out_counts[src_key] += 1
        in_counts[dst_key] += 1
        selected.append((confidence, src_ep, dst_ep, rule, transit))

    selected.sort(
        key=lambda item: (
            item[1].end_time,
            item[2].start_time,
            item[1].camera_id,
            item[1].track_id,
            item[2].camera_id,
            item[2].track_id,
        )
    )

    transitions: list[CrossCameraTransition] = []
    for idx, (confidence, src_ep, dst_ep, rule, transit) in enumerate(
        selected, start=1
    ):
        from_zone = rule.from_zone or (
            src_ep.exit_zones[0] if src_ep.exit_zones else None
        )
        to_zone = rule.to_zone or (
            dst_ep.entry_zones[0] if dst_ep.entry_zones else None
        )
        reason_parts = [
            (
                f"dt={transit:.2f}s in [{rule.min_transit_seconds:.1f}s, "
                f"{rule.max_transit_seconds:.1f}s]"
            )
        ]
        if rule.from_zone:
            reason_parts.append(f"exit_zone={rule.from_zone}")
        if rule.to_zone:
            reason_parts.append(f"entry_zone={rule.to_zone}")
        transitions.append(
            CrossCameraTransition(
                transition_id=f"tr_{idx:04d}",
                from_camera=src_ep.camera_id,
                from_track_id=src_ep.track_id,
                to_camera=dst_ep.camera_id,
                to_track_id=dst_ep.track_id,
                exit_time=src_ep.end_time,
                entry_time=dst_ep.start_time,
                transit_seconds=transit,
                confidence=confidence,
                rule_label=rule.label or f"{rule.from_camera} -> {rule.to_camera}",
                from_zone=from_zone,
                to_zone=to_zone,
                evidence_reason="; ".join(reason_parts),
            )
        )
    return transitions


def _score_transition_candidate(
    *,
    transit: float,
    target: float,
    half_span: float,
    rule: CameraTransitionRule,
    src_ep: TrackEndpoint,
    dst_ep: TrackEndpoint,
) -> float:
    """Compute a deterministic spatial-temporal handoff confidence in [0.0, 1.0]."""
    temporal_score = max(0.0, 1.0 - (abs(transit - target) / half_span))
    score = 0.35 + 0.50 * temporal_score
    if rule.from_zone:
        score += 0.075
    if rule.to_zone:
        score += 0.075
    if (
        not rule.from_zone
        and not rule.to_zone
        and src_ep.exit_zones
        and dst_ep.entry_zones
    ):
        score += 0.05
    return min(1.0, max(0.0, score))


def build_cross_camera_journeys(
    transitions: Sequence[CrossCameraTransition],
) -> list[CrossCameraJourney]:
    """Stitch sequential cross-camera transitions into cycle-safe multi-hop journeys."""
    if not transitions:
        return []

    outgoing: dict[tuple[str, int], list[CrossCameraTransition]] = defaultdict(list)
    incoming_nodes: set[tuple[str, int]] = set()
    for tr in transitions:
        outgoing[(tr.from_camera, tr.from_track_id)].append(tr)
        incoming_nodes.add((tr.to_camera, tr.to_track_id))

    for tr_list in outgoing.values():
        tr_list.sort(
            key=lambda item: (-item.confidence, item.exit_time, item.transition_id)
        )

    used_ids: set[str] = set()
    chains: list[list[CrossCameraTransition]] = []

    ordered_starts = sorted(
        transitions,
        key=lambda item: (
            (item.from_camera, item.from_track_id) in incoming_nodes,
            item.exit_time,
            item.transition_id,
        ),
    )

    for start_tr in ordered_starts:
        if start_tr.transition_id in used_ids:
            continue
        chain: list[CrossCameraTransition] = [start_tr]
        used_ids.add(start_tr.transition_id)
        visited_nodes: set[tuple[str, int]] = {
            (start_tr.from_camera, start_tr.from_track_id),
            (start_tr.to_camera, start_tr.to_track_id),
        }
        curr_node = (start_tr.to_camera, start_tr.to_track_id)
        while True:
            next_tr = None
            for cand in outgoing.get(curr_node, []):
                dst_node = (cand.to_camera, cand.to_track_id)
                if cand.transition_id not in used_ids and dst_node not in visited_nodes:
                    next_tr = cand
                    break
            if next_tr is None:
                break
            chain.append(next_tr)
            used_ids.add(next_tr.transition_id)
            curr_node = (next_tr.to_camera, next_tr.to_track_id)
            visited_nodes.add(curr_node)
        chains.append(chain)

    chains.sort(key=lambda ch: (ch[0].exit_time, -len(ch), ch[0].transition_id))

    journeys: list[CrossCameraJourney] = []
    for idx, chain in enumerate(chains, start=1):
        cameras: list[str] = [chain[0].from_camera]
        for hop in chain:
            cameras.append(hop.to_camera)
        start_t = chain[0].exit_time
        end_t = chain[-1].entry_time
        avg_conf = sum(hop.confidence for hop in chain) / len(chain)
        journeys.append(
            CrossCameraJourney(
                journey_id=f"jr_{idx:04d}",
                hops=tuple(chain),
                cameras_visited=tuple(cameras),
                start_time=start_t,
                end_time=end_t,
                total_span_seconds=max(0.0, end_t - start_t),
                avg_confidence=avg_conf,
            )
        )
    return journeys


def summarize_camera_pair_flows(
    transitions: Sequence[CrossCameraTransition],
) -> list[CameraPairFlow]:
    """Aggregate directed camera-pair flow statistics from correlated transitions."""
    grouped: dict[tuple[str, str], list[CrossCameraTransition]] = defaultdict(list)
    for tr in transitions:
        grouped[(tr.from_camera, tr.to_camera)].append(tr)

    flows: list[CameraPairFlow] = []
    for (src_cam, dst_cam), items in grouped.items():
        transits = [item.transit_seconds for item in items]
        confs = [item.confidence for item in items]
        flows.append(
            CameraPairFlow(
                from_camera=src_cam,
                to_camera=dst_cam,
                transition_count=len(items),
                avg_transit_seconds=sum(transits) / len(transits),
                min_transit_seconds=min(transits),
                max_transit_seconds=max(transits),
                avg_confidence=sum(confs) / len(confs),
            )
        )
    flows.sort(
        key=lambda item: (-item.transition_count, item.from_camera, item.to_camera)
    )
    return flows


def compute_co_activity_windows(
    db_path: Path | str,
    *,
    camera_time_offsets: Mapping[str, float] | None = None,
    bucket_seconds: float = 5.0,
    surge_min_cameras: int = 2,
    surge_min_tracks: int = 2,
) -> list[CoActivityWindow]:
    """Compute synchronized multi-camera activity windows and flag simultaneous surges."""
    if bucket_seconds <= 0:
        raise ValueError(f"bucket_seconds must be positive, got {bucket_seconds}")
    resolved = Path(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite intelligence DB not found: {resolved}")

    offsets = dict(camera_time_offsets or {})
    tracks_by_bucket: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    events_by_bucket: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with closing(sqlite3.connect(str(resolved))) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT camera_id, track_id, timestamp FROM trajectories"
        ):
            cam_id = str(row["camera_id"])
            ts = float(row["timestamp"]) + float(offsets.get(cam_id, 0.0))
            b_idx = math.floor(ts / bucket_seconds)
            tracks_by_bucket[b_idx][cam_id].add(int(row["track_id"]))

        for row in conn.execute("SELECT camera_id, start_time FROM events"):
            cam_id = str(row["camera_id"])
            ts = float(row["start_time"]) + float(offsets.get(cam_id, 0.0))
            b_idx = math.floor(ts / bucket_seconds)
            events_by_bucket[b_idx][cam_id] += 1

    all_indices = sorted(set(tracks_by_bucket.keys()) | set(events_by_bucket.keys()))
    if not all_indices:
        return []

    windows: list[CoActivityWindow] = []
    for b_idx in range(all_indices[0], all_indices[-1] + 1):
        cam_tracks = {
            cam: len(trks)
            for cam, trks in sorted(tracks_by_bucket.get(b_idx, {}).items())
            if trks
        }
        cam_events = {
            cam: cnt
            for cam, cnt in sorted(events_by_bucket.get(b_idx, {}).items())
            if cnt > 0
        }
        active_cams = tuple(sorted(set(cam_tracks.keys()) | set(cam_events.keys())))
        total_tracks = sum(cam_tracks.values())
        total_events = sum(cam_events.values())
        is_surge = (
            len(active_cams) >= surge_min_cameras and total_tracks >= surge_min_tracks
        )
        windows.append(
            CoActivityWindow(
                bucket_index=b_idx,
                start_time=b_idx * bucket_seconds,
                end_time=(b_idx + 1) * bucket_seconds,
                active_cameras=active_cams,
                per_camera_active_tracks=cam_tracks,
                total_active_tracks=total_tracks,
                per_camera_events=cam_events,
                total_events=total_events,
                is_simultaneous_surge=is_surge,
            )
        )
    return windows


def _discover_db_cameras(db_path: Path) -> tuple[str, ...]:
    """Return sorted unique camera_ids present across cameras, tracks, and trajectories."""
    cams: set[str] = set()
    with closing(sqlite3.connect(str(db_path))) as conn:
        for table in ("cameras", "tracks", "trajectories", "events"):
            try:
                for (cam_id,) in conn.execute(
                    f"SELECT DISTINCT camera_id FROM {table}"
                ):
                    if cam_id:
                        cams.add(str(cam_id))
            except sqlite3.OperationalError:
                continue
    return tuple(sorted(cams))


def analyze_multi_camera_correlation(
    db_path: Path | str,
    *,
    topology: CameraTopology | None = None,
    bucket_seconds: float = 5.0,
    min_confidence: float = 0.25,
    surge_min_cameras: int = 2,
    surge_min_tracks: int = 2,
) -> MultiCameraCorrelationReport:
    """Run end-to-end spatial-temporal correlation across all cameras in intelligence.db."""
    resolved = Path(db_path)
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite intelligence DB not found: {resolved}")

    cameras = _discover_db_cameras(resolved)
    active_topology = topology or infer_default_topology(cameras)

    endpoints = extract_track_endpoints(
        resolved,
        camera_time_offsets=active_topology.camera_time_offsets,
    )
    transitions = correlate_cross_camera_transitions(
        endpoints,
        active_topology,
        min_confidence=min_confidence,
    )
    journeys = build_cross_camera_journeys(transitions)
    pair_flows = summarize_camera_pair_flows(transitions)
    windows = compute_co_activity_windows(
        resolved,
        camera_time_offsets=active_topology.camera_time_offsets,
        bucket_seconds=bucket_seconds,
        surge_min_cameras=surge_min_cameras,
        surge_min_tracks=surge_min_tracks,
    )

    peak_cams = max((len(w.active_cameras) for w in windows), default=0)
    peak_tracks = max((w.total_active_tracks for w in windows), default=0)
    surge_count = sum(1 for w in windows if w.is_simultaneous_surge)

    summary_text = (
        f"Analyzed {len(cameras)} camera(s) and {len(endpoints)} track endpoint(s) "
        f"under topology '{active_topology.name}': found {len(transitions)} "
        f"cross-camera transition(s), {len(journeys)} stitched journey(s), "
        f"peak {peak_cams} simultaneous camera(s) ({peak_tracks} active track(s)), "
        f"and {surge_count} simultaneous surge window(s)."
    )

    return MultiCameraCorrelationReport(
        db_path=str(resolved),
        topology_name=active_topology.name,
        cameras=cameras,
        track_endpoints_count=len(endpoints),
        transitions=tuple(transitions),
        journeys=tuple(journeys),
        pair_flows=tuple(pair_flows),
        co_activity_windows=tuple(windows),
        peak_simultaneous_cameras=peak_cams,
        peak_simultaneous_tracks=peak_tracks,
        summary_text=summary_text,
    )
