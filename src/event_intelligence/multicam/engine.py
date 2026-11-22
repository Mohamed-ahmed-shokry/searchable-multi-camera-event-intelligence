"""Multi-camera spatial-temporal correlation engine and topology helpers."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path

import yaml

from event_intelligence.multicam.types import (
    CameraTopology,
    CameraTransitionRule,
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
