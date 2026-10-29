"""Phase 5: Zone and Line Reasoning data structures, YAML scene config loader,
point-in-polygon and directional line-crossing evaluators, boundary-flicker
analyzer, spatial video renderer, and CLI entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import LineString, Polygon


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
        raise ValueError(f"Expected mapping in {cfg_path}, got {type(raw).__name__}")

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
