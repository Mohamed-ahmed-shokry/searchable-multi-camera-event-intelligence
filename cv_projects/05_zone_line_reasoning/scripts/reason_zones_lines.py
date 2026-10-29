"""Phase 5: Zone and Line Reasoning data structures, YAML scene config loader,
point-in-polygon and directional line-crossing evaluators, boundary-flicker
analyzer, spatial video renderer, and CLI entrypoint.
"""

from __future__ import annotations

import csv
import itertools
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from shapely.geometry import LineString, Point, Polygon

try:
    from extract_trajectories import (
        TrajectoryPoint,
        draw_trajectory_trails,
        read_trajectories_csv,
    )
except ImportError:
    _subproject_4_scripts = (
        Path(__file__).resolve().parents[2] / "04_trajectory_extraction" / "scripts"
    )
    if str(_subproject_4_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_4_scripts))
    from extract_trajectories import (  # noqa: F401
        TrajectoryPoint,
        draw_trajectory_trails,
        read_trajectories_csv,
    )


@dataclass(frozen=True)
class ZoneSpec:
    """A named 2D polygon zone on the image ground plane."""

    id: str
    polygon: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("ZoneSpec.id must be a non-empty string")
        if len(self.polygon) < 3:
            raise ValueError(
                f"ZoneSpec {self.id!r} requires at least 3 vertices, got {len(self.polygon)}"
            )
        poly = Polygon(self.polygon)
        if not poly.is_valid or poly.area <= 0.0:
            raise ValueError(
                f"ZoneSpec {self.id!r} forms an invalid or zero-area polygon"
            )

    @property
    def shapely_polygon(self) -> Polygon:
        return Polygon(self.polygon)


@dataclass(frozen=True)
class LineSpec:
    """A named directed 2-point tripwire line segment (A -> B) on the image plane."""

    id: str
    points: tuple[tuple[float, float], tuple[float, float]]

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("LineSpec.id must be a non-empty string")
        if len(self.points) != 2:
            raise ValueError(
                f"LineSpec {self.id!r} requires exactly 2 endpoints, got {len(self.points)}"
            )
        line = LineString(self.points)
        if line.length <= 0.0:
            raise ValueError(f"LineSpec {self.id!r} has zero length")

    @property
    def shapely_line(self) -> LineString:
        return LineString(self.points)


@dataclass(frozen=True)
class SceneConfig:
    """Collection of named polygon zones and directed crossing lines for a scene."""

    zones: tuple[ZoneSpec, ...] = ()
    lines: tuple[LineSpec, ...] = ()


def load_scene_config(path: Path | str) -> SceneConfig:
    """Load and validate a Phase 5 scene_config.yaml file."""
    cfg_path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected mapping in {cfg_path}, got {type(raw).__name__}")

    seen_zones: set[str] = set()
    zones: list[ZoneSpec] = []
    for item in raw.get("zones") or []:
        zid = str(item["id"])
        if zid in seen_zones:
            raise ValueError(f"Duplicate zone id: {zid!r}")
        seen_zones.add(zid)
        coords = tuple((float(pt[0]), float(pt[1])) for pt in item["polygon"])
        zones.append(ZoneSpec(id=zid, polygon=coords))

    seen_lines: set[str] = set()
    lines: list[LineSpec] = []
    for item in raw.get("lines") or []:
        lid = str(item["id"])
        if lid in seen_lines:
            raise ValueError(f"Duplicate line id: {lid!r}")
        seen_lines.add(lid)
        pts_raw = item["points"]
        if len(pts_raw) != 2:
            raise ValueError(f"LineSpec {lid!r} requires 2 points, got {len(pts_raw)}")
        p0 = (float(pts_raw[0][0]), float(pts_raw[0][1]))
        p1 = (float(pts_raw[1][0]), float(pts_raw[1][1]))
        lines.append(LineSpec(id=lid, points=(p0, p1)))

    return SceneConfig(zones=tuple(zones), lines=tuple(lines))


def save_scene_config(path: Path | str, config: SceneConfig) -> Path:
    """Serialize a SceneConfig to YAML matching the Phase 5 scene_config.yaml schema."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "zones": [
            {
                "id": z.id,
                "polygon": [[round(x, 2), round(y, 2)] for x, y in z.polygon],
            }
            for z in config.zones
        ],
        "lines": [
            {
                "id": ln.id,
                "points": [[round(x, 2), round(y, 2)] for x, y in ln.points],
            }
            for ln in config.lines
        ],
    }
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


ZONE_LINE_EVENTS_CSV_HEADER: list[str] = [
    "frame_id",
    "track_id",
    "signal_type",
    "zone_id",
    "line_id",
    "direction",
    "timestamp",
]

VALID_SIGNAL_TYPES: tuple[str, ...] = ("zone_enter", "zone_exit", "line_cross")
VALID_DIRECTIONS: tuple[str, ...] = ("", "A->B", "B->A")


@dataclass(frozen=True)
class ZoneLineSignal:
    """A raw spatial signal (zone_enter, zone_exit, or line_cross) on a frame."""

    frame_id: int
    track_id: int
    signal_type: str
    zone_id: str = ""
    line_id: str = ""
    direction: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if self.signal_type not in VALID_SIGNAL_TYPES:
            raise ValueError(
                f"signal_type must be one of {VALID_SIGNAL_TYPES}, got {self.signal_type!r}"
            )
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got {self.direction!r}"
            )
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0, got {self.timestamp}")
        if self.signal_type in {"zone_enter", "zone_exit"} and not self.zone_id:
            raise ValueError(f"{self.signal_type} requires a non-empty zone_id")
        if self.signal_type == "line_cross" and (
            not self.line_id or not self.direction
        ):
            raise ValueError("line_cross requires both line_id and direction")

    def to_row(self) -> list[str]:
        return [
            str(self.frame_id),
            str(self.track_id),
            self.signal_type,
            self.zone_id,
            self.line_id,
            self.direction,
            f"{self.timestamp:.4f}",
        ]


def write_zone_line_events_csv(
    path: Path | str,
    signals: Sequence[ZoneLineSignal],
) -> Path:
    """Write raw spatial signals to CSV matching ZONE_LINE_EVENTS_CSV_HEADER."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        signals,
        key=lambda item: (
            item.frame_id,
            item.track_id,
            item.signal_type,
            item.zone_id,
            item.line_id,
        ),
    )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(ZONE_LINE_EVENTS_CSV_HEADER)
        for sig in ordered:
            writer.writerow(sig.to_row())
    return out_path


def read_zone_line_events_csv(path: Path | str) -> list[ZoneLineSignal]:
    """Read a zone_line_events.csv file into ordered ZoneLineSignal records."""
    in_path = Path(path)
    signals: list[ZoneLineSignal] = []
    with in_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ZONE_LINE_EVENTS_CSV_HEADER:
            raise ValueError(
                f"Expected CSV header {ZONE_LINE_EVENTS_CSV_HEADER}, got {header!r}"
            )
        for row in reader:
            if not row:
                continue
            signals.append(
                ZoneLineSignal(
                    frame_id=int(row[0]),
                    track_id=int(row[1]),
                    signal_type=row[2],
                    zone_id=row[3],
                    line_id=row[4],
                    direction=row[5],
                    timestamp=float(row[6]),
                )
            )
    return sorted(
        signals,
        key=lambda item: (
            item.frame_id,
            item.track_id,
            item.signal_type,
            item.zone_id,
            item.line_id,
        ),
    )


def point_in_zone(x: float, y: float, zone: ZoneSpec | Polygon) -> bool:
    """Return True if point (x, y) lies inside or on the boundary of `zone`."""
    poly = zone.shapely_polygon if isinstance(zone, ZoneSpec) else zone
    return bool(poly.covers(Point(x, y)))


def detect_zone_signals(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    zones: Sequence[ZoneSpec],
) -> list[ZoneLineSignal]:
    """Emit raw zone_enter and zone_exit signals per (track_id, zone_id)."""
    signals: list[ZoneLineSignal] = []
    zone_polys = [(z.id, z.shapely_polygon) for z in zones]

    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        if not pts:
            continue
        for zid, poly in zone_polys:
            was_inside = False
            for pt in pts:
                is_inside = point_in_zone(pt.x, pt.y, poly)
                if is_inside and not was_inside:
                    signals.append(
                        ZoneLineSignal(
                            frame_id=pt.frame_id,
                            track_id=pt.track_id,
                            signal_type="zone_enter",
                            zone_id=zid,
                            timestamp=pt.timestamp,
                        )
                    )
                elif not is_inside and was_inside:
                    signals.append(
                        ZoneLineSignal(
                            frame_id=pt.frame_id,
                            track_id=pt.track_id,
                            signal_type="zone_exit",
                            zone_id=zid,
                            timestamp=pt.timestamp,
                        )
                    )
                was_inside = is_inside

    return sorted(
        signals,
        key=lambda item: (item.frame_id, item.track_id, item.signal_type, item.zone_id),
    )


def _side_of_line(
    x: float,
    y: float,
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> float:
    """Return the signed 2D cross product of (p1 - p0) x ((x, y) - p0)."""
    return (p1[0] - p0[0]) * (y - p0[1]) - (p1[1] - p0[1]) * (x - p0[0])


def check_line_crossing(
    prev_xy: tuple[float, float],
    curr_xy: tuple[float, float],
    line: LineSpec,
) -> str | None:
    """Return 'A->B', 'B->A', or None for a consecutive step across a finite LineSpec.

    Side A has signed cross-product < 0; Side B has signed cross-product > 0.
    Also verifies that the finite trajectory step intersects the finite line segment.
    """
    if prev_xy == curr_xy:
        return None
    p0, p1 = line.points
    s_prev = _side_of_line(prev_xy[0], prev_xy[1], p0, p1)
    s_curr = _side_of_line(curr_xy[0], curr_xy[1], p0, p1)
    if s_prev == s_curr:
        return None

    # Require opposite signs (or one endpoint on the line while moving from/to a strict side)
    crossed_a_to_b = (s_prev < 0.0 and s_curr >= 0.0) or (
        s_prev <= 0.0 and s_curr > 0.0
    )
    crossed_b_to_a = (s_prev > 0.0 and s_curr <= 0.0) or (
        s_prev >= 0.0 and s_curr < 0.0
    )
    if not (crossed_a_to_b or crossed_b_to_a):
        return None

    step_seg = LineString([prev_xy, curr_xy])
    if not step_seg.intersects(line.shapely_line):
        return None

    return "A->B" if crossed_a_to_b else "B->A"


def detect_line_signals(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    lines: Sequence[LineSpec],
) -> list[ZoneLineSignal]:
    """Emit raw line_cross signals per (track_id, line_id) across consecutive trajectory steps."""
    signals: list[ZoneLineSignal] = []

    for tid in sorted(trajectories):
        pts = sorted(trajectories[tid], key=lambda item: item.frame_id)
        if len(pts) < 2:
            continue
        for line in lines:
            for cur, nxt in itertools.pairwise(pts):
                direction = check_line_crossing(
                    (cur.x, cur.y),
                    (nxt.x, nxt.y),
                    line,
                )
                if direction is not None:
                    signals.append(
                        ZoneLineSignal(
                            frame_id=nxt.frame_id,
                            track_id=nxt.track_id,
                            signal_type="line_cross",
                            line_id=line.id,
                            direction=direction,
                            timestamp=nxt.timestamp,
                        )
                    )

    return sorted(
        signals,
        key=lambda item: (item.frame_id, item.track_id, item.signal_type, item.line_id),
    )


def evaluate_scene_signals(
    trajectories: Mapping[int, Sequence[TrajectoryPoint]],
    scene_config: SceneConfig,
) -> list[ZoneLineSignal]:
    """Evaluate all zone enter/exit and directional line-crossing signals for a SceneConfig."""
    zone_sigs = detect_zone_signals(trajectories, scene_config.zones)
    line_sigs = detect_line_signals(trajectories, scene_config.lines)
    return sorted(
        [*zone_sigs, *line_sigs],
        key=lambda item: (
            item.frame_id,
            item.track_id,
            item.signal_type,
            item.zone_id,
            item.line_id,
        ),
    )


def analyze_boundary_flicker(
    signals: Sequence[ZoneLineSignal],
    *,
    max_flicker_frames: int = 5,
) -> dict[str, Any]:
    """Detect rapid zone enter/exit oscillations and back-and-forth line-crossing jitter."""
    if max_flicker_frames < 1:
        raise ValueError(f"max_flicker_frames must be >= 1, got {max_flicker_frames}")

    by_zone: dict[tuple[int, str], list[ZoneLineSignal]] = {}
    by_line: dict[tuple[int, str], list[ZoneLineSignal]] = {}

    for sig in sorted(signals, key=lambda item: (item.frame_id, item.track_id)):
        if sig.signal_type in {"zone_enter", "zone_exit"}:
            by_zone.setdefault((sig.track_id, sig.zone_id), []).append(sig)
        elif sig.signal_type == "line_cross":
            by_line.setdefault((sig.track_id, sig.line_id), []).append(sig)

    zone_flicker_cases: list[dict[str, Any]] = []
    for (tid, zid), group in sorted(by_zone.items()):
        for cur, nxt in itertools.pairwise(group):
            gap = nxt.frame_id - cur.frame_id
            if gap <= max_flicker_frames:
                zone_flicker_cases.append(
                    {
                        "track_id": tid,
                        "zone_id": zid,
                        "from_signal": cur.signal_type,
                        "to_signal": nxt.signal_type,
                        "from_frame": cur.frame_id,
                        "to_frame": nxt.frame_id,
                        "gap_frames": gap,
                    }
                )

    line_jitter_cases: list[dict[str, Any]] = []
    for (tid, lid), group in sorted(by_line.items()):
        for cur, nxt in itertools.pairwise(group):
            gap = nxt.frame_id - cur.frame_id
            if gap <= max_flicker_frames:
                line_jitter_cases.append(
                    {
                        "track_id": tid,
                        "line_id": lid,
                        "from_direction": cur.direction,
                        "to_direction": nxt.direction,
                        "from_frame": cur.frame_id,
                        "to_frame": nxt.frame_id,
                        "gap_frames": gap,
                    }
                )

    return {
        "max_flicker_frames": max_flicker_frames,
        "zone_flicker_count": len(zone_flicker_cases),
        "line_jitter_count": len(line_jitter_cases),
        "zone_flicker_cases": zone_flicker_cases,
        "line_jitter_cases": line_jitter_cases,
    }


ZONE_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (75, 180, 60),
    (0, 165, 255),
    (255, 105, 180),
    (255, 215, 0),
    (240, 50, 230),
)

LINE_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 220, 0),
    (220, 80, 255),
    (120, 255, 80),
    (50, 200, 255),
)


def draw_zones_and_lines(
    frame: np.ndarray,
    scene_config: SceneConfig,
    *,
    active_signals: Sequence[ZoneLineSignal] | None = None,
    alpha: float = 0.22,
) -> np.ndarray:
    """Render semi-transparent polygon zones, tripwire lines, and active signal highlights."""
    out = frame.copy()
    fill_layer = frame.copy()

    active_zones = {s.zone_id for s in (active_signals or ()) if s.zone_id}
    active_lines = {s.line_id for s in (active_signals or ()) if s.line_id}

    if scene_config.zones and alpha > 0.0:
        for idx, zone in enumerate(scene_config.zones):
            color = ZONE_PALETTE_BGR[idx % len(ZONE_PALETTE_BGR)]
            pts = np.array(
                [[round(x), round(y)] for x, y in zone.polygon],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.fillPoly(fill_layer, [pts], color)
        cv2.addWeighted(fill_layer, alpha, out, 1.0 - alpha, 0.0, dst=out)

    for idx, zone in enumerate(scene_config.zones):
        base_color = ZONE_PALETTE_BGR[idx % len(ZONE_PALETTE_BGR)]
        is_active = zone.id in active_zones
        border_color = (0, 255, 255) if is_active else base_color
        thickness = 3 if is_active else 2
        pts = np.array(
            [[round(x), round(y)] for x, y in zone.polygon],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(
            out,
            [pts],
            isClosed=True,
            color=border_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        lx = round(zone.polygon[0][0]) + 6
        ly = max(18, round(zone.polygon[0][1]) + 18)
        cv2.putText(
            out,
            zone.id,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            border_color,
            1,
            cv2.LINE_AA,
        )

    for idx, line in enumerate(scene_config.lines):
        base_color = LINE_PALETTE_BGR[idx % len(LINE_PALETTE_BGR)]
        is_active = line.id in active_lines
        line_color = (0, 0, 255) if is_active else base_color
        thickness = 4 if is_active else 2
        p0 = (round(line.points[0][0]), round(line.points[0][1]))
        p1 = (round(line.points[1][0]), round(line.points[1][1]))
        cv2.line(out, p0, p1, line_color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, p0, 4, line_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p1, 4, line_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "A",
            (p0[0] + 5, max(14, p0[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            line_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "B",
            (p1[0] + 5, max(14, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            line_color,
            1,
            cv2.LINE_AA,
        )
        mx = round((p0[0] + p1[0]) / 2.0) + 6
        my = round((p0[1] + p1[1]) / 2.0) - 6
        cv2.putText(
            out,
            line.id,
            (mx, max(16, my)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            line_color,
            1,
            cv2.LINE_AA,
        )

    for s_idx, sig in enumerate(active_signals or ()):
        target = f"{sig.zone_id}" if sig.zone_id else f"{sig.line_id} ({sig.direction})"
        banner = f"T{sig.track_id} {sig.signal_type}: {target}"
        cv2.putText(
            out,
            banner,
            (12, 52 + s_idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out
