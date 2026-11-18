"""Domain types for multi-run comparison, baseline diffing, and cross-run trends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Aggregated KPI snapshot of a single processed event_intelligence run."""

    run_id: str
    run_dir: str
    db_path: str
    cameras_count: int
    detections_count: int
    tracks_count: int
    trajectories_count: int
    events_count: int
    insights_count: int
    high_critical_insights_count: int
    total_dwell_seconds: float
    mean_track_duration_seconds: float
    mean_path_length_px: float
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        for name, val in (
            ("cameras_count", self.cameras_count),
            ("detections_count", self.detections_count),
            ("tracks_count", self.tracks_count),
            ("trajectories_count", self.trajectories_count),
            ("events_count", self.events_count),
            ("insights_count", self.insights_count),
            ("high_critical_insights_count", self.high_critical_insights_count),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")
        if self.total_dwell_seconds < 0.0:
            raise ValueError("total_dwell_seconds must be >= 0.0")
        if self.mean_track_duration_seconds < 0.0:
            raise ValueError("mean_track_duration_seconds must be >= 0.0")
        if self.mean_path_length_px < 0.0:
            raise ValueError("mean_path_length_px must be >= 0.0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "db_path": self.db_path,
            "cameras_count": self.cameras_count,
            "detections_count": self.detections_count,
            "tracks_count": self.tracks_count,
            "trajectories_count": self.trajectories_count,
            "events_count": self.events_count,
            "insights_count": self.insights_count,
            "high_critical_insights_count": self.high_critical_insights_count,
            "total_dwell_seconds": round(self.total_dwell_seconds, 4),
            "mean_track_duration_seconds": round(self.mean_track_duration_seconds, 4),
            "mean_path_length_px": round(self.mean_path_length_px, 2),
            "config_snapshot": dict(self.config_snapshot),
        }


@dataclass(frozen=True, slots=True)
class ConfigDiffEntry:
    """Single flattened parameter change between baseline and candidate run_config.yaml."""

    key: str
    baseline_value: Any
    candidate_value: Any

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("key must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
        }


@dataclass(frozen=True, slots=True)
class ZoneMetricDelta:
    """Signed difference in zone activity counts and dwell time between two runs."""

    camera_id: str
    zone_id: str
    baseline_entries: int
    candidate_entries: int
    baseline_exits: int
    candidate_exits: int
    baseline_dwell_seconds: float
    candidate_dwell_seconds: float
    baseline_peak_occupancy: int
    candidate_peak_occupancy: int

    @property
    def delta_entries(self) -> int:
        return self.candidate_entries - self.baseline_entries

    @property
    def delta_exits(self) -> int:
        return self.candidate_exits - self.baseline_exits

    @property
    def delta_dwell_seconds(self) -> float:
        return round(self.candidate_dwell_seconds - self.baseline_dwell_seconds, 4)

    @property
    def delta_peak_occupancy(self) -> int:
        return self.candidate_peak_occupancy - self.baseline_peak_occupancy

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "baseline_entries": self.baseline_entries,
            "candidate_entries": self.candidate_entries,
            "delta_entries": self.delta_entries,
            "baseline_exits": self.baseline_exits,
            "candidate_exits": self.candidate_exits,
            "delta_exits": self.delta_exits,
            "baseline_dwell_seconds": round(self.baseline_dwell_seconds, 4),
            "candidate_dwell_seconds": round(self.candidate_dwell_seconds, 4),
            "delta_dwell_seconds": self.delta_dwell_seconds,
            "baseline_peak_occupancy": self.baseline_peak_occupancy,
            "candidate_peak_occupancy": self.candidate_peak_occupancy,
            "delta_peak_occupancy": self.delta_peak_occupancy,
        }


@dataclass(frozen=True, slots=True)
class LineMetricDelta:
    """Signed difference in directional tripwire crossings between two runs."""

    camera_id: str
    line_id: str
    baseline_ab: int
    candidate_ab: int
    baseline_ba: int
    candidate_ba: int
    baseline_total: int
    candidate_total: int

    @property
    def delta_ab(self) -> int:
        return self.candidate_ab - self.baseline_ab

    @property
    def delta_ba(self) -> int:
        return self.candidate_ba - self.baseline_ba

    @property
    def delta_total(self) -> int:
        return self.candidate_total - self.baseline_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "line_id": self.line_id,
            "baseline_ab": self.baseline_ab,
            "candidate_ab": self.candidate_ab,
            "delta_ab": self.delta_ab,
            "baseline_ba": self.baseline_ba,
            "candidate_ba": self.candidate_ba,
            "delta_ba": self.delta_ba,
            "baseline_total": self.baseline_total,
            "candidate_total": self.candidate_total,
            "delta_total": self.delta_total,
        }


@dataclass(frozen=True, slots=True)
class InsightTypeDelta:
    """Signed count difference per behavioral insight type between two runs."""

    insight_type: str
    baseline_count: int
    candidate_count: int
    baseline_high_critical: int = 0
    candidate_high_critical: int = 0

    @property
    def delta_count(self) -> int:
        return self.candidate_count - self.baseline_count

    @property
    def delta_high_critical(self) -> int:
        return self.candidate_high_critical - self.baseline_high_critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
            "delta_count": self.delta_count,
            "baseline_high_critical": self.baseline_high_critical,
            "candidate_high_critical": self.candidate_high_critical,
            "delta_high_critical": self.delta_high_critical,
        }


@dataclass(frozen=True, slots=True)
class TemporalBucketDelta:
    """Signed difference in time-windowed activity metrics between two runs."""

    bucket_index: int
    start_time: float
    end_time: float
    baseline_active_tracks: int
    candidate_active_tracks: int
    baseline_events: int
    candidate_events: int
    baseline_insights: int
    candidate_insights: int

    @property
    def delta_active_tracks(self) -> int:
        return self.candidate_active_tracks - self.baseline_active_tracks

    @property
    def delta_events(self) -> int:
        return self.candidate_events - self.baseline_events

    @property
    def delta_insights(self) -> int:
        return self.candidate_insights - self.baseline_insights

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket_index": self.bucket_index,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "baseline_active_tracks": self.baseline_active_tracks,
            "candidate_active_tracks": self.candidate_active_tracks,
            "delta_active_tracks": self.delta_active_tracks,
            "baseline_events": self.baseline_events,
            "candidate_events": self.candidate_events,
            "delta_events": self.delta_events,
            "baseline_insights": self.baseline_insights,
            "candidate_insights": self.candidate_insights,
            "delta_insights": self.delta_insights,
        }


@dataclass(frozen=True, slots=True)
class SpatialShiftSummary:
    """Quantitative summary of spatial trajectory density shift (candidate - baseline)."""

    grid_shape: tuple[int, int]
    frame_width: int
    frame_height: int
    mean_abs_delta: float
    max_increase: float
    max_decrease: float
    hotspot_increase_xy: tuple[float, float] | None = None
    hotspot_decrease_xy: tuple[float, float] | None = None
    diff_image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "mean_abs_delta": round(self.mean_abs_delta, 6),
            "max_increase": round(self.max_increase, 6),
            "max_decrease": round(self.max_decrease, 6),
            "hotspot_increase_xy": (
                [
                    round(self.hotspot_increase_xy[0], 2),
                    round(self.hotspot_increase_xy[1], 2),
                ]
                if self.hotspot_increase_xy is not None
                else None
            ),
            "hotspot_decrease_xy": (
                [
                    round(self.hotspot_decrease_xy[0], 2),
                    round(self.hotspot_decrease_xy[1], 2),
                ]
                if self.hotspot_decrease_xy is not None
                else None
            ),
            "diff_image_path": self.diff_image_path,
        }


@dataclass(frozen=True, slots=True)
class RunComparisonReport:
    """Complete structured diff between a baseline run and a candidate run."""

    baseline: RunSnapshot
    candidate: RunSnapshot
    camera_filter: str | None = None
    config_diffs: list[ConfigDiffEntry] = field(default_factory=list)
    zone_deltas: list[ZoneMetricDelta] = field(default_factory=list)
    line_deltas: list[LineMetricDelta] = field(default_factory=list)
    insight_deltas: list[InsightTypeDelta] = field(default_factory=list)
    temporal_deltas: list[TemporalBucketDelta] = field(default_factory=list)
    spatial_shift: SpatialShiftSummary | None = None

    @property
    def kpi_deltas(self) -> dict[str, int | float]:
        return {
            "detections_delta": self.candidate.detections_count
            - self.baseline.detections_count,
            "tracks_delta": self.candidate.tracks_count - self.baseline.tracks_count,
            "trajectories_delta": self.candidate.trajectories_count
            - self.baseline.trajectories_count,
            "events_delta": self.candidate.events_count - self.baseline.events_count,
            "insights_delta": self.candidate.insights_count
            - self.baseline.insights_count,
            "high_critical_insights_delta": self.candidate.high_critical_insights_count
            - self.baseline.high_critical_insights_count,
            "total_dwell_seconds_delta": round(
                self.candidate.total_dwell_seconds - self.baseline.total_dwell_seconds,
                4,
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_filter": self.camera_filter,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "kpi_deltas": self.kpi_deltas,
            "config_diffs": [c.to_dict() for c in self.config_diffs],
            "zone_deltas": [z.to_dict() for z in self.zone_deltas],
            "line_deltas": [ln.to_dict() for ln in self.line_deltas],
            "insight_deltas": [i.to_dict() for i in self.insight_deltas],
            "temporal_deltas": [t.to_dict() for t in self.temporal_deltas],
            "spatial_shift": (
                self.spatial_shift.to_dict() if self.spatial_shift is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RunBaselineAnomaly:
    """Outlier metric flagged against the multi-run baseline mean and standard deviation."""

    run_id: str
    metric_name: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in {"above", "below"}:
            raise ValueError(
                f"direction must be 'above' or 'below', got {self.direction!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "value": round(self.value, 4),
            "baseline_mean": round(self.baseline_mean, 4),
            "baseline_std": round(self.baseline_std, 4),
            "z_score": round(self.z_score, 4),
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class MultiRunTrendReport:
    """Cross-run KPI catalog, baseline statistics, and outlier flags across N runs."""

    runs: list[RunSnapshot]
    z_score_threshold: float
    metric_baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    anomalies: list[RunBaselineAnomaly] = field(default_factory=list)
    camera_filter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_count": len(self.runs),
            "camera_filter": self.camera_filter,
            "z_score_threshold": round(self.z_score_threshold, 4),
            "runs": [r.to_dict() for r in self.runs],
            "metric_baselines": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.metric_baselines.items()
            },
            "anomalies": [a.to_dict() for a in self.anomalies],
        }
