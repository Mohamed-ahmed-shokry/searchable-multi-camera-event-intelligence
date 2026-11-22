"""Data models for multi-camera spatial-temporal correlation and transition analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CameraTransitionRule:
    """Directed topology edge describing a valid cross-camera handoff."""

    from_camera: str
    to_camera: str
    min_transit_seconds: float = 0.0
    max_transit_seconds: float = 30.0
    from_zone: str | None = None
    to_zone: str | None = None
    expected_transit_seconds: float | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize transition rule to a JSON/YAML-compatible dict."""
        return {
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "min_transit_seconds": self.min_transit_seconds,
            "max_transit_seconds": self.max_transit_seconds,
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "expected_transit_seconds": self.expected_transit_seconds,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraTransitionRule:
        """Deserialize transition rule from a mapping."""
        expected = data.get("expected_transit_seconds")
        return cls(
            from_camera=str(data["from_camera"]),
            to_camera=str(data["to_camera"]),
            min_transit_seconds=float(data.get("min_transit_seconds", 0.0)),
            max_transit_seconds=float(data.get("max_transit_seconds", 30.0)),
            from_zone=str(data["from_zone"]) if data.get("from_zone") else None,
            to_zone=str(data["to_zone"]) if data.get("to_zone") else None,
            expected_transit_seconds=float(expected) if expected is not None else None,
            label=str(data.get("label", "")),
        )


@dataclass(frozen=True)
class CameraTopology:
    """Multi-camera topology specification with clock offsets and transition edges."""

    name: str = "default_topology"
    description: str = ""
    camera_time_offsets: dict[str, float] = field(default_factory=dict)
    transitions: tuple[CameraTransitionRule, ...] = ()
    default_min_transit_seconds: float = 0.0
    default_max_transit_seconds: float = 30.0
    max_matches_per_track: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology to a JSON/YAML-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "camera_time_offsets": dict(self.camera_time_offsets),
            "default_min_transit_seconds": self.default_min_transit_seconds,
            "default_max_transit_seconds": self.default_max_transit_seconds,
            "max_matches_per_track": self.max_matches_per_track,
            "transitions": [rule.to_dict() for rule in self.transitions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraTopology:
        """Deserialize topology from a mapping."""
        raw_offsets = data.get("camera_time_offsets") or {}
        offsets = {str(k): float(v) for k, v in raw_offsets.items()}
        raw_rules = data.get("transitions") or []
        rules = tuple(CameraTransitionRule.from_dict(item) for item in raw_rules)
        return cls(
            name=str(data.get("name", "default_topology")),
            description=str(data.get("description", "")),
            camera_time_offsets=offsets,
            transitions=rules,
            default_min_transit_seconds=float(
                data.get("default_min_transit_seconds", 0.0)
            ),
            default_max_transit_seconds=float(
                data.get("default_max_transit_seconds", 30.0)
            ),
            max_matches_per_track=int(data.get("max_matches_per_track", 1)),
        )
