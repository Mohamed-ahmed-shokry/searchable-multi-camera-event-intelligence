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


@dataclass(frozen=True)
class TrackEndpoint:
    """Temporal and spatial boundary summary for a single camera track."""

    camera_id: str
    track_id: int
    start_time: float
    end_time: float
    duration_seconds: float
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    entry_zones: tuple[str, ...] = ()
    exit_zones: tuple[str, ...] = ()
    event_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize track endpoint to a JSON-compatible dict."""
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "start_x": round(self.start_x, 2) if self.start_x is not None else None,
            "start_y": round(self.start_y, 2) if self.start_y is not None else None,
            "end_x": round(self.end_x, 2) if self.end_x is not None else None,
            "end_y": round(self.end_y, 2) if self.end_y is not None else None,
            "entry_zones": list(self.entry_zones),
            "exit_zones": list(self.exit_zones),
            "event_count": self.event_count,
        }


@dataclass(frozen=True)
class CrossCameraTransition:
    """Candidate spatial-temporal handoff linking two tracks across cameras."""

    transition_id: str
    from_camera: str
    from_track_id: int
    to_camera: str
    to_track_id: int
    exit_time: float
    entry_time: float
    transit_seconds: float
    confidence: float
    rule_label: str = ""
    from_zone: str | None = None
    to_zone: str | None = None
    evidence_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize cross-camera transition to a JSON-compatible dict."""
        return {
            "transition_id": self.transition_id,
            "from_camera": self.from_camera,
            "from_track_id": self.from_track_id,
            "to_camera": self.to_camera,
            "to_track_id": self.to_track_id,
            "exit_time": round(self.exit_time, 3),
            "entry_time": round(self.entry_time, 3),
            "transit_seconds": round(self.transit_seconds, 3),
            "confidence": round(self.confidence, 4),
            "rule_label": self.rule_label,
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "evidence_reason": self.evidence_reason,
        }


@dataclass(frozen=True)
class CrossCameraJourney:
    """Multi-hop stitched journey across two or more cameras."""

    journey_id: str
    hops: tuple[CrossCameraTransition, ...]
    cameras_visited: tuple[str, ...]
    start_time: float
    end_time: float
    total_span_seconds: float
    avg_confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize multi-hop journey to a JSON-compatible dict."""
        return {
            "journey_id": self.journey_id,
            "hop_count": len(self.hops),
            "cameras_visited": list(self.cameras_visited),
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "total_span_seconds": round(self.total_span_seconds, 3),
            "avg_confidence": round(self.avg_confidence, 4),
            "hops": [hop.to_dict() for hop in self.hops],
        }


@dataclass(frozen=True)
class CameraPairFlow:
    """Aggregated transition statistics between a directed pair of cameras."""

    from_camera: str
    to_camera: str
    transition_count: int
    avg_transit_seconds: float
    min_transit_seconds: float
    max_transit_seconds: float
    avg_confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize pair flow summary to a JSON-compatible dict."""
        return {
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "transition_count": self.transition_count,
            "avg_transit_seconds": round(self.avg_transit_seconds, 3),
            "min_transit_seconds": round(self.min_transit_seconds, 3),
            "max_transit_seconds": round(self.max_transit_seconds, 3),
            "avg_confidence": round(self.avg_confidence, 4),
        }


@dataclass(frozen=True)
class CoActivityWindow:
    """Synchronized temporal activity bucket across all registered cameras."""

    bucket_index: int
    start_time: float
    end_time: float
    active_cameras: tuple[str, ...]
    per_camera_active_tracks: dict[str, int]
    total_active_tracks: int
    per_camera_events: dict[str, int]
    total_events: int
    is_simultaneous_surge: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize co-activity window to a JSON-compatible dict."""
        return {
            "bucket_index": self.bucket_index,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "active_camera_count": len(self.active_cameras),
            "active_cameras": list(self.active_cameras),
            "per_camera_active_tracks": dict(self.per_camera_active_tracks),
            "total_active_tracks": self.total_active_tracks,
            "per_camera_events": dict(self.per_camera_events),
            "total_events": self.total_events,
            "is_simultaneous_surge": self.is_simultaneous_surge,
        }


@dataclass(frozen=True)
class MultiCameraCorrelationReport:
    """Full structured multi-camera spatial-temporal correlation report."""

    db_path: str
    topology_name: str
    cameras: tuple[str, ...]
    track_endpoints_count: int
    transitions: tuple[CrossCameraTransition, ...]
    journeys: tuple[CrossCameraJourney, ...]
    pair_flows: tuple[CameraPairFlow, ...]
    co_activity_windows: tuple[CoActivityWindow, ...]
    peak_simultaneous_cameras: int
    peak_simultaneous_tracks: int
    summary_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize full multi-camera correlation report to a JSON-compatible dict."""
        return {
            "db_path": self.db_path,
            "topology_name": self.topology_name,
            "cameras": list(self.cameras),
            "camera_count": len(self.cameras),
            "track_endpoints_count": self.track_endpoints_count,
            "transition_count": len(self.transitions),
            "journey_count": len(self.journeys),
            "peak_simultaneous_cameras": self.peak_simultaneous_cameras,
            "peak_simultaneous_tracks": self.peak_simultaneous_tracks,
            "summary_text": self.summary_text,
            "pair_flows": [flow.to_dict() for flow in self.pair_flows],
            "transitions": [item.to_dict() for item in self.transitions],
            "journeys": [journey.to_dict() for journey in self.journeys],
            "co_activity_windows": [
                window.to_dict() for window in self.co_activity_windows
            ],
        }
