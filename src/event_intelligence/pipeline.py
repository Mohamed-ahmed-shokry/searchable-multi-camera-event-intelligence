"""Phase 11 — CV Pipeline Integration: unified end-to-end CV pipeline runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from event_intelligence.evidence import VALID_BEST_FRAME_STRATEGIES
from event_intelligence.trajectories import VALID_ANCHORS


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Unified configuration controlling all stages of the event_intelligence pipeline."""

    model_name: str = "yolo11m.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    tracker_config: str = "bytetrack.yaml"
    device: str | None = None
    precomputed_detections_csv: str | None = None
    precomputed_tracks_csv: str | None = None
    scene_config_path: str = (
        "cv_projects/05_zone_line_reasoning/configs/scene_config.yaml"
    )
    anchor: str = "bottom_center"
    smoothing_window: int = 3
    max_trail_length: int = 60
    min_dwell_seconds: float = 1.2
    min_entry_frames: int = 1
    grace_frames: int = 5
    line_cooldown_frames: int = 5
    include_line_crossings: bool = True
    per_zone_min_dwell_seconds: Mapping[str, float] = field(default_factory=dict)
    best_frame_strategy: str = "highest_confidence"
    crop_padding_ratio: float = 0.10
    min_padding_px: int = 8
    jpeg_quality: int = 92
    grid_bin_size_px: int = 16
    gaussian_sigma_px: float = 24.0
    normalization_mode: str = "log"
    max_points_per_track_cell: int = 6
    colormap: str = "jet"
    reference_frame_index: int = 35
    inspection_sample_count: int = 12
    save_annotated_video: bool = True
    max_frames: int | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in (0.0, 1.0]")
        if not (0.0 < self.iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in (0.0, 1.0]")
        if self.anchor not in VALID_ANCHORS:
            raise ValueError(
                f"anchor must be one of {VALID_ANCHORS}, got {self.anchor!r}"
            )
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        if self.min_dwell_seconds < 0.0:
            raise ValueError("min_dwell_seconds must be >= 0.0")
        if self.best_frame_strategy not in VALID_BEST_FRAME_STRATEGIES:
            raise ValueError(
                f"best_frame_strategy must be one of {VALID_BEST_FRAME_STRATEGIES}"
            )
        if self.grid_bin_size_px < 1:
            raise ValueError("grid_bin_size_px must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 when specified")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "tracker_config": self.tracker_config,
            "device": self.device,
            "precomputed_detections_csv": self.precomputed_detections_csv,
            "precomputed_tracks_csv": self.precomputed_tracks_csv,
            "scene_config_path": self.scene_config_path,
            "anchor": self.anchor,
            "smoothing_window": self.smoothing_window,
            "max_trail_length": self.max_trail_length,
            "min_dwell_seconds": self.min_dwell_seconds,
            "min_entry_frames": self.min_entry_frames,
            "grace_frames": self.grace_frames,
            "line_cooldown_frames": self.line_cooldown_frames,
            "include_line_crossings": self.include_line_crossings,
            "per_zone_min_dwell_seconds": dict(self.per_zone_min_dwell_seconds),
            "best_frame_strategy": self.best_frame_strategy,
            "crop_padding_ratio": self.crop_padding_ratio,
            "min_padding_px": self.min_padding_px,
            "jpeg_quality": self.jpeg_quality,
            "grid_bin_size_px": self.grid_bin_size_px,
            "gaussian_sigma_px": self.gaussian_sigma_px,
            "normalization_mode": self.normalization_mode,
            "max_points_per_track_cell": self.max_points_per_track_cell,
            "colormap": self.colormap,
            "reference_frame_index": self.reference_frame_index,
            "inspection_sample_count": self.inspection_sample_count,
            "save_annotated_video": self.save_annotated_video,
            "max_frames": self.max_frames,
        }


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    """Summary of an end-to-end event_intelligence pipeline run."""

    run_id: str
    run_dir: str
    source_path: str
    processed_frames: int
    detections_count: int
    tracks_count: int
    trajectories_count: int
    zone_line_signals_count: int
    events_count: int
    zones_count: int
    lines_count: int
    db_path: str
    run_config_path: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "source_path": self.source_path,
            "processed_frames": self.processed_frames,
            "detections_count": self.detections_count,
            "tracks_count": self.tracks_count,
            "trajectories_count": self.trajectories_count,
            "zone_line_signals_count": self.zone_line_signals_count,
            "events_count": self.events_count,
            "zones_count": self.zones_count,
            "lines_count": self.lines_count,
            "db_path": self.db_path,
            "run_config_path": self.run_config_path,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


def load_pipeline_config(path: Path | str | None = None) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file, or return defaults if None."""
    if path is None:
        return PipelineConfig()
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        return PipelineConfig()
    if not isinstance(raw, dict):
        raise TypeError("Pipeline config YAML root must be a mapping")

    defaults = PipelineConfig()
    per_zone_raw = raw.get("per_zone_min_dwell_seconds") or {}
    if not isinstance(per_zone_raw, dict):
        raise TypeError("per_zone_min_dwell_seconds must be a mapping")

    max_frames_val = raw.get("max_frames", defaults.max_frames)
    det_csv = raw.get("precomputed_detections_csv", defaults.precomputed_detections_csv)
    trk_csv = raw.get("precomputed_tracks_csv", defaults.precomputed_tracks_csv)
    dev_val = raw.get("device", defaults.device)

    return PipelineConfig(
        model_name=str(raw.get("model_name", defaults.model_name)),
        confidence_threshold=float(
            raw.get("confidence_threshold", defaults.confidence_threshold)
        ),
        iou_threshold=float(raw.get("iou_threshold", defaults.iou_threshold)),
        tracker_config=str(raw.get("tracker_config", defaults.tracker_config)),
        device=str(dev_val) if dev_val is not None else None,
        precomputed_detections_csv=str(det_csv) if det_csv is not None else None,
        precomputed_tracks_csv=str(trk_csv) if trk_csv is not None else None,
        scene_config_path=str(raw.get("scene_config_path", defaults.scene_config_path)),
        anchor=str(raw.get("anchor", defaults.anchor)),
        smoothing_window=int(raw.get("smoothing_window", defaults.smoothing_window)),
        max_trail_length=int(raw.get("max_trail_length", defaults.max_trail_length)),
        min_dwell_seconds=float(
            raw.get("min_dwell_seconds", defaults.min_dwell_seconds)
        ),
        min_entry_frames=int(raw.get("min_entry_frames", defaults.min_entry_frames)),
        grace_frames=int(raw.get("grace_frames", defaults.grace_frames)),
        line_cooldown_frames=int(
            raw.get("line_cooldown_frames", defaults.line_cooldown_frames)
        ),
        include_line_crossings=bool(
            raw.get("include_line_crossings", defaults.include_line_crossings)
        ),
        per_zone_min_dwell_seconds={str(k): float(v) for k, v in per_zone_raw.items()},
        best_frame_strategy=str(
            raw.get("best_frame_strategy", defaults.best_frame_strategy)
        ),
        crop_padding_ratio=float(
            raw.get("crop_padding_ratio", defaults.crop_padding_ratio)
        ),
        min_padding_px=int(raw.get("min_padding_px", defaults.min_padding_px)),
        jpeg_quality=int(raw.get("jpeg_quality", defaults.jpeg_quality)),
        grid_bin_size_px=int(raw.get("grid_bin_size_px", defaults.grid_bin_size_px)),
        gaussian_sigma_px=float(
            raw.get("gaussian_sigma_px", defaults.gaussian_sigma_px)
        ),
        normalization_mode=str(
            raw.get("normalization_mode", defaults.normalization_mode)
        ),
        max_points_per_track_cell=int(
            raw.get("max_points_per_track_cell", defaults.max_points_per_track_cell)
        ),
        colormap=str(raw.get("colormap", defaults.colormap)),
        reference_frame_index=int(
            raw.get("reference_frame_index", defaults.reference_frame_index)
        ),
        inspection_sample_count=int(
            raw.get("inspection_sample_count", defaults.inspection_sample_count)
        ),
        save_annotated_video=bool(
            raw.get("save_annotated_video", defaults.save_annotated_video)
        ),
        max_frames=int(max_frames_val) if max_frames_val is not None else None,
    )


def save_pipeline_config(config: PipelineConfig, path: Path | str) -> Path:
    """Serialize a PipelineConfig to YAML (e.g. frozen run_config.yaml)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return out_path
