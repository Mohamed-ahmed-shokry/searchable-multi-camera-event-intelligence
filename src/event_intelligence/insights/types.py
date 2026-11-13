"""Domain types, enumerations, and configuration for behavior insights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class InsightType(StrEnum):
    """Categorical classification of surveillance behavioral insights."""

    LOITERING = "loitering"
    WRONG_WAY = "wrong_way"
    HIGH_SPEED = "high_speed"
    STATIONARY_IN_ZONE = "stationary_in_zone"
    TRACK_DROP_IN_ZONE = "track_drop_in_zone"
    CROSS_CAMERA_TRANSITION = "cross_camera_transition"


class Severity(StrEnum):
    """Priority and impact levels for behavioral insights."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


VALID_INSIGHT_TYPES = frozenset(t.value for t in InsightType)
VALID_SEVERITIES = frozenset(s.value for s in Severity)


@dataclass(frozen=True, slots=True)
class InsightRecord:
    """Structured record of a single detected behavioral insight or anomaly."""

    insight_id: str
    camera_id: str
    insight_type: str
    severity: str
    track_id: int | None
    zone_id: str | None
    start_time: float
    end_time: float
    duration: float
    description: str
    evidence_snapshot_path: str | None = None
    evidence_crop_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.insight_id.strip():
            raise ValueError("insight_id must be non-empty")
        if not self.camera_id.strip():
            raise ValueError("camera_id must be non-empty")
        if self.insight_type not in VALID_INSIGHT_TYPES:
            raise ValueError(
                f"insight_type must be one of {sorted(VALID_INSIGHT_TYPES)}, got {self.insight_type!r}"
            )
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(VALID_SEVERITIES)}, got {self.severity!r}"
            )
        if self.start_time < 0.0 or self.end_time < self.start_time:
            raise ValueError(
                f"timestamps must satisfy 0.0 <= start_time <= end_time, got start={self.start_time}, end={self.end_time}"
            )
        if self.duration < 0.0:
            raise ValueError(f"duration must be >= 0.0, got {self.duration}")

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InsightRecord:
        """Instantiate an InsightRecord from a dictionary."""
        return cls(
            insight_id=str(data["insight_id"]),
            camera_id=str(data.get("camera_id", "default")),
            insight_type=str(data["insight_type"]),
            severity=str(data.get("severity", "medium")),
            track_id=(
                int(data["track_id"]) if data.get("track_id") is not None else None
            ),
            zone_id=str(data["zone_id"]) if data.get("zone_id") is not None else None,
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            duration=float(data["duration"]),
            description=str(data.get("description", "")),
            evidence_snapshot_path=(
                str(data["evidence_snapshot_path"])
                if data.get("evidence_snapshot_path")
                else None
            ),
            evidence_crop_path=(
                str(data["evidence_crop_path"])
                if data.get("evidence_crop_path")
                else None
            ),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class InsightConfig:
    """Configurable thresholds and options for behavioral anomaly detection."""

    loitering_factor: float = 2.5
    min_loitering_seconds: float = 15.0
    per_zone_loitering_seconds: Mapping[str, float] = field(default_factory=dict)
    speed_anomaly_threshold_px_s: float = 180.0
    min_speed_anomaly_frames: int = 5
    stationary_speed_threshold_px_s: float = 5.0
    stationary_min_seconds: float = 10.0
    restricted_zones: tuple[str, ...] = ()
    allowed_line_directions: Mapping[str, str] = field(default_factory=dict)
    disallowed_line_crossings: tuple[str, ...] = ()
    cross_camera_min_seconds: float = 0.5
    cross_camera_max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.loitering_factor <= 1.0:
            raise ValueError(
                f"loitering_factor must be > 1.0, got {self.loitering_factor}"
            )
        if self.min_loitering_seconds < 0.0:
            raise ValueError("min_loitering_seconds must be >= 0.0")
        if self.speed_anomaly_threshold_px_s <= 0.0:
            raise ValueError("speed_anomaly_threshold_px_s must be > 0.0")
        if self.min_speed_anomaly_frames < 1:
            raise ValueError("min_speed_anomaly_frames must be >= 1")
        if self.stationary_speed_threshold_px_s < 0.0:
            raise ValueError("stationary_speed_threshold_px_s must be >= 0.0")
        if self.stationary_min_seconds < 0.0:
            raise ValueError("stationary_min_seconds must be >= 0.0")
        if (
            self.cross_camera_min_seconds < 0.0
            or self.cross_camera_max_seconds < self.cross_camera_min_seconds
        ):
            raise ValueError("cross camera window must be 0.0 <= min <= max")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "loitering_factor": self.loitering_factor,
            "min_loitering_seconds": self.min_loitering_seconds,
            "per_zone_loitering_seconds": dict(self.per_zone_loitering_seconds),
            "speed_anomaly_threshold_px_s": self.speed_anomaly_threshold_px_s,
            "min_speed_anomaly_frames": self.min_speed_anomaly_frames,
            "stationary_speed_threshold_px_s": self.stationary_speed_threshold_px_s,
            "stationary_min_seconds": self.stationary_min_seconds,
            "restricted_zones": list(self.restricted_zones),
            "allowed_line_directions": dict(self.allowed_line_directions),
            "disallowed_line_crossings": list(self.disallowed_line_crossings),
            "cross_camera_min_seconds": self.cross_camera_min_seconds,
            "cross_camera_max_seconds": self.cross_camera_max_seconds,
        }
