"""Phase 9 — CV Evaluation: quantitative MOT17 metrics, qualitative surveillance
spot-checks, per-stage FPS benchmarking, and annotated failure example extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configuration for Phase 9 quantitative and qualitative CV evaluation."""

    iou_threshold: float = 0.5
    min_gt_visibility: float = 0.0
    mot17_max_frames: int = 300
    mot17_sequence_path: str = "data/mot17/MOT17-11-FRCNN"
    mot17_gt_path: str = "data/mot17/MOT17-11-FRCNN/gt/gt.txt"
    mot17_bytetrack_csv: str = "cv_projects/03_person_tracking/outputs/mot17_11_comparison/bytetrack/tracks.csv"
    mot17_botsort_csv: str = (
        "cv_projects/03_person_tracking/outputs/mot17_11_comparison/botsort/tracks.csv"
    )
    surveillance_video_path: str = "data/fallback_clips/people-detection.mp4"
    surveillance_events_csv: str = (
        "cv_projects/07_visual_evidence/outputs/people_detection/events.csv"
    )
    surveillance_zone_counts_csv: str = (
        "cv_projects/08_heatmaps_analytics/outputs/people_detection/zone_counts.csv"
    )
    surveillance_line_counts_csv: str = (
        "cv_projects/08_heatmaps_analytics/outputs/people_detection/line_counts.csv"
    )
    surveillance_stay_traces_json: str = (
        "cv_projects/06_dwell_time_events/outputs/people_detection/"
        "event_debug_log/stay_traces.json"
    )
    dwell_tolerance_seconds: float = 0.1
    expected_zone_entries: dict[str, int] = field(
        default_factory=lambda: {
            "entrance_area": 7,
            "center_concourse": 6,
            "right_aisle": 3,
        }
    )
    expected_zone_exits: dict[str, int] = field(
        default_factory=lambda: {
            "entrance_area": 7,
            "center_concourse": 6,
            "right_aisle": 0,
        }
    )
    expected_line_crossings: dict[str, int] = field(
        default_factory=lambda: {
            "doorway_line:A->B": 0,
            "doorway_line:B->A": 7,
            "aisle_split_line:A->B": 1,
            "aisle_split_line:B->A": 1,
        }
    )
    expected_dwell_events_count: int = 10
    expected_track_lost_in_zone_count: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0.0, 1.0]")
        if not 0.0 <= self.min_gt_visibility <= 1.0:
            raise ValueError("min_gt_visibility must be in [0.0, 1.0]")
        if self.mot17_max_frames <= 0:
            raise ValueError("mot17_max_frames must be positive")
        if self.dwell_tolerance_seconds < 0.0:
            raise ValueError("dwell_tolerance_seconds must be non-negative")
        if self.expected_dwell_events_count < 0:
            raise ValueError("expected_dwell_events_count must be non-negative")
        if self.expected_track_lost_in_zone_count < 0:
            raise ValueError("expected_track_lost_in_zone_count must be non-negative")


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Quantitative frame-by-frame pedestrian detection metrics against MOT17 GT."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    num_gt: int
    num_predictions: int
    mean_iou: float
    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.precision <= 1.0:
            raise ValueError("precision must be in [0.0, 1.0]")
        if not 0.0 <= self.recall <= 1.0:
            raise ValueError("recall must be in [0.0, 1.0]")
        if not 0.0 <= self.f1 <= 1.0:
            raise ValueError("f1 must be in [0.0, 1.0]")
        if self.tp < 0 or self.fp < 0 or self.fn < 0:
            raise ValueError("tp, fp, and fn must be non-negative")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "num_gt": self.num_gt,
            "num_predictions": self.num_predictions,
            "mean_iou": round(self.mean_iou, 4),
            "iou_threshold": round(self.iou_threshold, 4),
        }


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    """Quantitative CLEAR-MOT, IDF1, and track lifespan coverage metrics on MOT17."""

    tracker_name: str
    mota: float
    motp: float
    idf1: float
    precision: float
    recall: float
    id_switches: int
    tp: int
    fp: int
    fn: int
    num_gt_boxes: int
    num_pred_boxes: int
    num_gt_tracks: int
    num_pred_tracks: int
    mostly_tracked: int
    partially_tracked: int
    mostly_lost: int
    fragmentation_count: int
    id_switch_events: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.tracker_name.strip():
            raise ValueError("tracker_name must be non-empty")
        if self.id_switches < 0 or self.fragmentation_count < 0:
            raise ValueError("id_switches and fragmentation_count must be non-negative")
        if (
            self.mostly_tracked < 0
            or self.partially_tracked < 0
            or self.mostly_lost < 0
        ):
            raise ValueError("MT/PT/ML counts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_name": self.tracker_name,
            "mota": round(self.mota, 4),
            "motp": round(self.motp, 4),
            "idf1": round(self.idf1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "id_switches": self.id_switches,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "num_gt_boxes": self.num_gt_boxes,
            "num_pred_boxes": self.num_pred_boxes,
            "num_gt_tracks": self.num_gt_tracks,
            "num_pred_tracks": self.num_pred_tracks,
            "mostly_tracked": self.mostly_tracked,
            "partially_tracked": self.partially_tracked,
            "mostly_lost": self.mostly_lost,
            "fragmentation_count": self.fragmentation_count,
            "id_switch_events": list(self.id_switch_events),
        }


@dataclass(frozen=True, slots=True)
class QualitativeSpotCheck:
    """Manual playback audit vs. automated event and analytics verification."""

    total_events: int
    zone_entry_accuracy: float
    zone_exit_accuracy: float
    line_crossing_accuracy: float
    dwell_count_matches: bool
    track_lost_in_zone_matches: bool
    max_dwell_duration_error_seconds: float
    dwell_within_tolerance: bool
    zone_entry_comparison: dict[str, dict[str, int | bool]]
    zone_exit_comparison: dict[str, dict[str, int | bool]]
    line_crossing_comparison: dict[str, dict[str, int | bool]]

    def __post_init__(self) -> None:
        for name, val in (
            ("zone_entry_accuracy", self.zone_entry_accuracy),
            ("zone_exit_accuracy", self.zone_exit_accuracy),
            ("line_crossing_accuracy", self.line_crossing_accuracy),
        ):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0]")
        if self.max_dwell_duration_error_seconds < 0.0:
            raise ValueError("max_dwell_duration_error_seconds must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "zone_entry_accuracy": round(self.zone_entry_accuracy, 4),
            "zone_exit_accuracy": round(self.zone_exit_accuracy, 4),
            "line_crossing_accuracy": round(self.line_crossing_accuracy, 4),
            "dwell_count_matches": self.dwell_count_matches,
            "track_lost_in_zone_matches": self.track_lost_in_zone_matches,
            "max_dwell_duration_error_seconds": round(
                self.max_dwell_duration_error_seconds, 4
            ),
            "dwell_within_tolerance": self.dwell_within_tolerance,
            "zone_entry_comparison": self.zone_entry_comparison,
            "zone_exit_comparison": self.zone_exit_comparison,
            "line_crossing_comparison": self.line_crossing_comparison,
        }


@dataclass(frozen=True, slots=True)
class StageFpsRecord:
    """Processing throughput and wall-clock latency for a single CV pipeline stage."""

    stage_id: str
    stage_name: str
    device: str
    frames_processed: int
    elapsed_seconds: float
    fps: float
    ms_per_frame: float
    runtime_share_pct: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.stage_id.strip() or not self.stage_name.strip():
            raise ValueError("stage_id and stage_name must be non-empty")
        if self.frames_processed <= 0:
            raise ValueError("frames_processed must be positive")
        if self.elapsed_seconds <= 0.0 or self.fps <= 0.0:
            raise ValueError("elapsed_seconds and fps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "device": self.device,
            "frames_processed": self.frames_processed,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "fps": round(self.fps, 2),
            "ms_per_frame": round(self.ms_per_frame, 2),
            "runtime_share_pct": round(self.runtime_share_pct, 2),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class FailureCaseRecord:
    """Concrete visual failure example with frame index, root cause, and mitigation."""

    case_id: str
    category: str
    source_name: str
    frame_id: int
    title: str
    symptom: str
    root_cause: str
    impact_and_mitigation: str
    image_path: str

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.title.strip():
            raise ValueError("case_id and title must be non-empty")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.image_path.strip():
            raise ValueError("image_path must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "source_name": self.source_name,
            "frame_id": self.frame_id,
            "title": self.title,
            "symptom": self.symptom,
            "root_cause": self.root_cause,
            "impact_and_mitigation": self.impact_and_mitigation,
            "image_path": self.image_path,
        }


def load_evaluation_config(path: Path | str | None = None) -> EvaluationConfig:
    """Load an EvaluationConfig from a YAML file, or return defaults if None."""
    if path is None:
        return EvaluationConfig()
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return EvaluationConfig()
    if not isinstance(raw, dict):
        raise TypeError("Evaluation config YAML root must be a mapping")

    defaults = EvaluationConfig()
    return EvaluationConfig(
        iou_threshold=float(raw.get("iou_threshold", defaults.iou_threshold)),
        min_gt_visibility=float(
            raw.get("min_gt_visibility", defaults.min_gt_visibility)
        ),
        mot17_max_frames=int(raw.get("mot17_max_frames", defaults.mot17_max_frames)),
        mot17_sequence_path=str(
            raw.get("mot17_sequence_path", defaults.mot17_sequence_path)
        ),
        mot17_gt_path=str(raw.get("mot17_gt_path", defaults.mot17_gt_path)),
        mot17_bytetrack_csv=str(
            raw.get("mot17_bytetrack_csv", defaults.mot17_bytetrack_csv)
        ),
        mot17_botsort_csv=str(raw.get("mot17_botsort_csv", defaults.mot17_botsort_csv)),
        surveillance_video_path=str(
            raw.get("surveillance_video_path", defaults.surveillance_video_path)
        ),
        surveillance_events_csv=str(
            raw.get("surveillance_events_csv", defaults.surveillance_events_csv)
        ),
        surveillance_zone_counts_csv=str(
            raw.get(
                "surveillance_zone_counts_csv", defaults.surveillance_zone_counts_csv
            )
        ),
        surveillance_line_counts_csv=str(
            raw.get(
                "surveillance_line_counts_csv", defaults.surveillance_line_counts_csv
            )
        ),
        surveillance_stay_traces_json=str(
            raw.get(
                "surveillance_stay_traces_json", defaults.surveillance_stay_traces_json
            )
        ),
        dwell_tolerance_seconds=float(
            raw.get("dwell_tolerance_seconds", defaults.dwell_tolerance_seconds)
        ),
        expected_zone_entries={
            str(k): int(v)
            for k, v in dict(
                raw.get("expected_zone_entries", defaults.expected_zone_entries)
            ).items()
        },
        expected_zone_exits={
            str(k): int(v)
            for k, v in dict(
                raw.get("expected_zone_exits", defaults.expected_zone_exits)
            ).items()
        },
        expected_line_crossings={
            str(k): int(v)
            for k, v in dict(
                raw.get("expected_line_crossings", defaults.expected_line_crossings)
            ).items()
        },
        expected_dwell_events_count=int(
            raw.get("expected_dwell_events_count", defaults.expected_dwell_events_count)
        ),
        expected_track_lost_in_zone_count=int(
            raw.get(
                "expected_track_lost_in_zone_count",
                defaults.expected_track_lost_in_zone_count,
            )
        ),
    )


def save_evaluation_config(config: EvaluationConfig, path: Path | str) -> Path:
    """Serialize an EvaluationConfig to YAML."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iou_threshold": config.iou_threshold,
        "min_gt_visibility": config.min_gt_visibility,
        "mot17_max_frames": config.mot17_max_frames,
        "mot17_sequence_path": config.mot17_sequence_path,
        "mot17_gt_path": config.mot17_gt_path,
        "mot17_bytetrack_csv": config.mot17_bytetrack_csv,
        "mot17_botsort_csv": config.mot17_botsort_csv,
        "surveillance_video_path": config.surveillance_video_path,
        "surveillance_events_csv": config.surveillance_events_csv,
        "surveillance_zone_counts_csv": config.surveillance_zone_counts_csv,
        "surveillance_line_counts_csv": config.surveillance_line_counts_csv,
        "surveillance_stay_traces_json": config.surveillance_stay_traces_json,
        "dwell_tolerance_seconds": config.dwell_tolerance_seconds,
        "expected_zone_entries": dict(config.expected_zone_entries),
        "expected_zone_exits": dict(config.expected_zone_exits),
        "expected_line_crossings": dict(config.expected_line_crossings),
        "expected_dwell_events_count": config.expected_dwell_events_count,
        "expected_track_lost_in_zone_count": config.expected_track_lost_in_zone_count,
    }
    out_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return out_path
