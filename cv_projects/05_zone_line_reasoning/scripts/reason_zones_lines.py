"""Phase 5: Zone and Line Reasoning data structures, YAML scene config loader,
point-in-polygon and directional line-crossing evaluators, boundary-flicker
analyzer, spatial video renderer, and CLI entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass

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
