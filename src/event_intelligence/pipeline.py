"""Phase 11 — CV Pipeline Integration: unified end-to-end CV pipeline runner."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import yaml

from event_intelligence.analytics import (
    accumulate_trajectory_grid,
    compute_line_counts,
    compute_track_duration_stats,
    compute_zone_counts,
    render_heatmap_overlay,
    smooth_and_normalize_heatmap,
    write_line_counts_csv,
    write_zone_counts_csv,
)
from event_intelligence.detection import (
    Detection,
    FrameDetections,
    PersonDetector,
    create_video_writer,
    read_detections_csv,
    write_detections_csv,
)
from event_intelligence.events import (
    EventConfig,
    EventRecord,
    draw_dwell_events_overlay,
    evaluate_dwell_events,
    write_events_csv,
)
from event_intelligence.evidence import (
    VALID_BEST_FRAME_STRATEGIES,
    extract_event_storyboard,
    extract_event_video_clip,
    run_visual_evidence,
)
from event_intelligence.insights import (
    InsightConfig,
    InsightRecord,
    detect_cross_camera_transitions,
    generate_all_insights,
    generate_intelligence_report_markdown,
    write_insights_csv,
    write_insights_json,
    write_intelligence_report,
)
from event_intelligence.spatial import (
    SceneConfig,
    evaluate_scene_signals,
    load_scene_config,
    write_zone_line_events_csv,
)
from event_intelligence.storage import (
    CameraMetadata,
    SearchConfig,
    ingest_intelligence_db,
    ingest_multi_camera_intelligence_db,
    insert_insights,
    query_camera_summary,
    query_events,
    query_insights,
)
from event_intelligence.tracking import (
    FrameTracks,
    PersonTracker,
    TrackDetection,
    compute_track_summaries,
    read_tracks_csv,
    write_track_summary_csv,
    write_tracks_csv,
)
from event_intelligence.trajectories import (
    VALID_ANCHORS,
    compute_trajectory_summaries,
    smooth_trajectories,
    tracks_to_trajectories,
    write_trajectories_csv,
    write_trajectory_summary_csv,
)
from event_intelligence.video import (
    VideoSource,
    sample_frame_indices,
    source_name,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    insight_config: InsightConfig | None = None
    extract_video_clips: bool = False
    clip_pre_roll_seconds: float = 1.0
    clip_post_roll_seconds: float = 1.0
    render_clip_telemetry: bool = True
    extract_storyboards: bool = False

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
        if self.clip_pre_roll_seconds < 0.0:
            raise ValueError("clip_pre_roll_seconds must be >= 0.0")
        if self.clip_post_roll_seconds < 0.0:
            raise ValueError("clip_post_roll_seconds must be >= 0.0")

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
            "extract_video_clips": self.extract_video_clips,
            "clip_pre_roll_seconds": self.clip_pre_roll_seconds,
            "clip_post_roll_seconds": self.clip_post_roll_seconds,
            "render_clip_telemetry": self.render_clip_telemetry,
            "extract_storyboards": self.extract_storyboards,
            "insight_config": (
                self.insight_config.to_dict() if self.insight_config else None
            ),
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
    insights_count: int = 0

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
            "insights_count": self.insights_count,
            "db_path": self.db_path,
            "run_config_path": self.run_config_path,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


@dataclass(frozen=True, slots=True)
class MultiCameraPipelineRunResult:
    """Summary of a multi-camera event_intelligence pipeline run."""

    run_id: str
    run_dir: str
    cameras: list[str]
    camera_results: list[PipelineRunResult]
    db_path: str
    cameras_count: int
    total_detections: int
    total_tracks: int
    total_events: int
    elapsed_seconds: float
    total_insights: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "cameras": list(self.cameras),
            "cameras_count": self.cameras_count,
            "total_detections": self.total_detections,
            "total_tracks": self.total_tracks,
            "total_events": self.total_events,
            "total_insights": self.total_insights,
            "camera_results": [r.to_dict() for r in self.camera_results],
            "db_path": self.db_path,
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

    ins_raw = raw.get("insights")
    ins_cfg: InsightConfig | None = None
    if isinstance(ins_raw, dict):
        ins_cfg = InsightConfig(
            loitering_factor=float(ins_raw.get("loitering_factor", 2.5)),
            min_loitering_seconds=float(ins_raw.get("min_loitering_seconds", 15.0)),
            per_zone_loitering_seconds={
                str(k): float(v)
                for k, v in (ins_raw.get("per_zone_loitering_seconds") or {}).items()
            },
            speed_anomaly_threshold_px_s=float(
                ins_raw.get("speed_anomaly_threshold_px_s", 180.0)
            ),
            min_speed_anomaly_frames=int(ins_raw.get("min_speed_anomaly_frames", 5)),
            stationary_speed_threshold_px_s=float(
                ins_raw.get("stationary_speed_threshold_px_s", 5.0)
            ),
            stationary_min_seconds=float(ins_raw.get("stationary_min_seconds", 10.0)),
            restricted_zones=tuple(str(z) for z in ins_raw.get("restricted_zones", ())),
            allowed_line_directions={
                str(k): str(v)
                for k, v in (ins_raw.get("allowed_line_directions") or {}).items()
            },
            disallowed_line_crossings=tuple(
                str(d) for d in ins_raw.get("disallowed_line_crossings", ())
            ),
            cross_camera_min_seconds=float(
                ins_raw.get("cross_camera_min_seconds", 0.5)
            ),
            cross_camera_max_seconds=float(
                ins_raw.get("cross_camera_max_seconds", 60.0)
            ),
        )

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
        insight_config=ins_cfg,
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


def _resolve_optional_path(path_str: str | None, project_root: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (project_root / p)


def _detections_from_tracks(frame_tracks: list[FrameTracks]) -> list[FrameDetections]:
    return [
        FrameDetections(
            frame_id=ft.frame_id,
            detections=[
                Detection(
                    x1=trk.x1,
                    y1=trk.y1,
                    x2=trk.x2,
                    y2=trk.y2,
                    confidence=trk.confidence,
                    class_name="person",
                )
                for trk in ft.tracks
            ],
        )
        for ft in frame_tracks
    ]


def _load_detections_as_frame_records(
    csv_path: Path, max_frames: int | None
) -> list[FrameDetections]:
    rows = read_detections_csv(csv_path)
    by_frame: dict[int, FrameDetections] = {}
    for row in rows:
        fid = int(row["frame_id"])
        if max_frames is not None and fid >= max_frames:
            continue
        rec = by_frame.setdefault(fid, FrameDetections(frame_id=fid))
        if row["x1"] is not None:
            rec.detections.append(
                Detection(
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    confidence=float(row["confidence"]),
                    class_name=str(row["class"] or "person"),
                )
            )
    return [by_frame[k] for k in sorted(by_frame)]


def validate_cached_artifact_source(
    csv_path: Path,
    expected_source_path: Path,
    *,
    strict: bool = False,
) -> bool:
    """Validate whether a precomputed detections/tracks CSV belongs to expected_source_path."""
    if not csv_path.is_file():
        return False

    expected_name = expected_source_path.name.replace("_", "-").lower()
    expected_stem = expected_source_path.stem.replace("_", "-").lower()

    for summary_name in (
        "detection_summary.json",
        "tracking_summary.json",
        "video_info.json",
    ):
        summary_file = csv_path.parent / summary_name
        if summary_file.is_file():
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    data = data[0]
                if isinstance(data, dict):
                    cached_src = data.get("source_path") or data.get("name")
                    if cached_src:
                        cached_name = Path(cached_src).name.replace("_", "-").lower()
                        cached_stem = Path(cached_src).stem.replace("_", "-").lower()
                        if (
                            cached_name != expected_name
                            and cached_stem != expected_stem
                        ):
                            if strict:
                                raise ValueError(
                                    f"Cached CSV {csv_path} was generated for source {cached_src!r}, "
                                    f"not for requested video {expected_source_path.name!r}."
                                )
                            return False
            except (json.JSONDecodeError, OSError):
                pass
    return True


def run_pipeline(
    video_path: Path | str,
    config: PipelineConfig | None = None,
    *,
    output_root: Path | str | None = None,
    run_id: str | None = None,
    project_root: Path | None = None,
    detector: PersonDetector | None = None,
    tracker: PersonTracker | None = None,
) -> PipelineRunResult:
    """Execute the full Phase 1..10 CV pipeline into outputs/runs/<run_id>/."""
    t0 = time.perf_counter()
    cfg = config if config is not None else PipelineConfig()
    root = project_root if project_root is not None else _PROJECT_ROOT

    resolved_run_id = (
        run_id.strip()
        if run_id and run_id.strip()
        else datetime.now(UTC).strftime("run_%Y%m%dT%H%M%S_") + uuid4().hex[:6]
    )
    runs_base = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / "outputs" / "runs").resolve()
    )
    run_dir = (runs_base / resolved_run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = save_pipeline_config(cfg, run_dir / "run_config.yaml")

    src_path = Path(video_path)
    if not src_path.is_absolute() and not src_path.exists():
        candidate = root / src_path
        if candidate.exists():
            src_path = candidate
    src_path = src_path.resolve()

    with VideoSource(src_path) as vs:
        meta = vs.metadata
        fps = float(meta.fps) if meta.fps > 0 else 30.0
        width = int(meta.width)
        height = int(meta.height)
        total_frames = (
            min(meta.frame_count, cfg.max_frames)
            if cfg.max_frames is not None
            else meta.frame_count
        )
        sampled_indices = sample_frame_indices(
            meta.frame_count, cfg.inspection_sample_count
        )
        ref_idx = min(cfg.reference_frame_index, max(0, total_frames - 1))
        reference_frame = vs.read_frame(ref_idx)

    video_info_entry = {
        "name": source_name(src_path),
        **asdict(meta),
        "sampled_frame_indices": sampled_indices,
        "sampled_timestamps_seconds": [idx / fps for idx in sampled_indices],
    }
    (run_dir / "video_info.json").write_text(
        json.dumps([video_info_entry], indent=2) + "\n",
        encoding="utf-8",
    )

    # Resolve tracks & detections (either precomputed CSV or live inference)
    trk_csv_in = _resolve_optional_path(cfg.precomputed_tracks_csv, root)
    det_csv_in = _resolve_optional_path(cfg.precomputed_detections_csv, root)

    use_cached_tracks = (
        trk_csv_in is not None
        and trk_csv_in.is_file()
        and validate_cached_artifact_source(trk_csv_in, src_path)
    )
    if use_cached_tracks:
        frame_tracks = read_tracks_csv(trk_csv_in)
        if cfg.max_frames is not None:
            frame_tracks = [ft for ft in frame_tracks if ft.frame_id < cfg.max_frames]
    else:
        active_tracker = (
            tracker
            if tracker is not None
            else PersonTracker(
                model_name=cfg.model_name,
                tracker_config=cfg.tracker_config,
                confidence_threshold=cfg.confidence_threshold,
                iou_threshold=cfg.iou_threshold,
                device=cfg.device,
            )
        )
        frame_tracks = []
        with VideoSource(src_path) as vs:
            for fid, frame in vs.iter_frames(stop=total_frames):
                trks = active_tracker.track(frame, frame_id=fid)
                frame_tracks.append(FrameTracks(frame_id=fid, tracks=trks))

    use_cached_detections = (
        det_csv_in is not None
        and det_csv_in.is_file()
        and validate_cached_artifact_source(det_csv_in, src_path)
    )
    if use_cached_detections:
        frame_detections = _load_detections_as_frame_records(det_csv_in, cfg.max_frames)
    elif detector is not None:
        frame_detections = []
        with VideoSource(src_path) as vs:
            for fid, frame in vs.iter_frames(stop=total_frames):
                dets = detector.detect(frame)
                frame_detections.append(FrameDetections(frame_id=fid, detections=dets))
    else:
        frame_detections = _detections_from_tracks(frame_tracks)

    detections_csv_path = run_dir / "detections.csv"
    detections_count = write_detections_csv(frame_detections, detections_csv_path)

    tracks_csv_path = write_tracks_csv(run_dir / "tracks.csv", frame_tracks)
    track_summaries = compute_track_summaries(frame_tracks, fps=fps)
    track_summary_csv_path = write_track_summary_csv(
        run_dir / "track_summary.csv", track_summaries
    )

    # Phase 4: Trajectory extraction & smoothing
    raw_trajs = tracks_to_trajectories(frame_tracks, fps=fps, anchor=cfg.anchor)
    trajectories = smooth_trajectories(raw_trajs, window=cfg.smoothing_window)
    trajectories_csv_path = write_trajectories_csv(
        run_dir / "trajectories.csv", trajectories
    )
    traj_summaries = compute_trajectory_summaries(trajectories, fps=fps)
    trajectory_summary_csv_path = write_trajectory_summary_csv(
        run_dir / "trajectory_summary.csv", traj_summaries
    )

    # Phase 5: Zone & line spatial reasoning
    scene_cfg_file = _resolve_optional_path(cfg.scene_config_path, root)
    scene_cfg = (
        load_scene_config(scene_cfg_file)
        if scene_cfg_file is not None and scene_cfg_file.is_file()
        else SceneConfig()
    )
    signals = evaluate_scene_signals(trajectories, scene_cfg)
    write_zone_line_events_csv(run_dir / "zone_line_events.csv", signals)

    # Phase 6: Dwell-time state machine events
    ev_cfg = EventConfig(
        min_dwell_seconds=cfg.min_dwell_seconds,
        min_entry_frames=cfg.min_entry_frames,
        grace_frames=cfg.grace_frames,
        line_cooldown_frames=cfg.line_cooldown_frames,
        include_line_crossings=cfg.include_line_crossings,
        per_zone_min_dwell_seconds=cfg.per_zone_min_dwell_seconds,
    )
    raw_events, traces = evaluate_dwell_events(trajectories, scene_cfg, ev_cfg)
    raw_events_csv = write_events_csv(run_dir / "events.csv", raw_events)

    # Phase 7 & 11e: Visual evidence (snapshots, crops, clips, storyboards, enriched events.csv)
    run_visual_evidence(
        source_path=src_path,
        events_csv_path=raw_events_csv,
        tracks_csv_path=tracks_csv_path,
        output_dir=run_dir,
        scene_config_path=scene_cfg_file,
        camera_id=src_path.stem,
        best_frame_strategy_override=cfg.best_frame_strategy,
        crop_padding_ratio_override=cfg.crop_padding_ratio,
        min_padding_px_override=cfg.min_padding_px,
        jpeg_quality_override=cfg.jpeg_quality,
        extract_video_clips_override=cfg.extract_video_clips,
        clip_pre_roll_seconds_override=cfg.clip_pre_roll_seconds,
        clip_post_roll_seconds_override=cfg.clip_post_roll_seconds,
        render_clip_telemetry_override=cfg.render_clip_telemetry,
        extract_storyboards_override=cfg.extract_storyboards,
    )

    # Phase 8: Heatmaps & zone/line analytics
    flat_points = [pt for tid in sorted(trajectories) for pt in trajectories[tid]]
    raw_grid, full_res = accumulate_trajectory_grid(
        flat_points,
        frame_width=width,
        frame_height=height,
        grid_bin_size_px=cfg.grid_bin_size_px,
        max_points_per_track_cell=cfg.max_points_per_track_cell,
    )
    norm_heat = smooth_and_normalize_heatmap(
        full_res,
        gaussian_sigma_px=cfg.gaussian_sigma_px,
        normalization_mode=cfg.normalization_mode,
    )
    heatmap_img = render_heatmap_overlay(
        reference_frame,
        norm_heat,
        colormap=cfg.colormap,
        scene_config=scene_cfg,
    )
    heatmap_path = run_dir / "heatmap.png"
    cv2.imwrite(str(heatmap_path), heatmap_img)

    zone_counts = compute_zone_counts(
        raw_events,
        trajectory_points=flat_points,
        scene_config=scene_cfg,
    )
    line_counts = compute_line_counts(raw_events, scene_config=scene_cfg)
    duration_stats = compute_track_duration_stats(flat_points)

    zone_counts_csv_path = write_zone_counts_csv(
        zone_counts, run_dir / "zone_counts.csv"
    )
    line_counts_csv_path = write_line_counts_csv(
        line_counts, run_dir / "line_counts.csv"
    )

    analytics_summary = {
        "source_path": str(src_path),
        "video_metadata": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": total_frames,
        },
        "heatmap_stats": {
            "grid_shape": [int(raw_grid.shape[0]), int(raw_grid.shape[1])],
            "total_trajectory_points": len(flat_points),
            "active_grid_cells": int(np.count_nonzero(raw_grid > 0)),
        },
        "zone_counts": [
            {
                "zone_id": z.zone_id,
                "entry_count": z.entry_count,
                "exit_count": z.exit_count,
                "total_dwell_seconds": z.total_dwell_seconds,
                "peak_occupancy": z.peak_occupancy,
            }
            for z in zone_counts
        ],
        "line_counts": [
            {
                "line_id": l.line_id,
                "crossings_ab": l.crossings_ab,
                "crossings_ba": l.crossings_ba,
                "total": l.total,
            }
            for l in line_counts
        ],
        "track_duration_distribution": duration_stats.to_dict(),
        "total_path_length_px": round(sum(s.path_length_px for s in traj_summaries), 2),
    }
    (run_dir / "analytics_summary.json").write_text(
        json.dumps(analytics_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # Render unified annotated_video.mp4
    if cfg.save_annotated_video and total_frames > 0:
        writer = create_video_writer(
            run_dir / "annotated_video.mp4",
            fps=fps,
            width=width,
            height=height,
        )
        try:
            with VideoSource(src_path) as vs:
                for fid, frame in vs.iter_frames(stop=total_frames):
                    rendered = draw_dwell_events_overlay(
                        frame,
                        scene_cfg,
                        trajectories,
                        current_frame_id=fid,
                        current_timestamp=fid / fps,
                        traces=traces,
                        events=raw_events,
                        max_trail_length=cfg.max_trail_length,
                    )
                    writer.write(rendered)
        finally:
            writer.release()

    # Phase 11d: Generate behavioral insights & anomaly intelligence
    ins_cfg = cfg.insight_config if cfg.insight_config is not None else InsightConfig()
    insights = generate_all_insights(
        events=raw_events,
        trajectories=trajectories,
        config=ins_cfg,
        traces=traces,
        camera_id=src_path.stem,
    )

    if (cfg.extract_video_clips or cfg.extract_storyboards) and insights:
        enriched_insights = []
        for ins in insights:
            ins_track_id = ins.track_id if ins.track_id is not None else -1
            ins_track_dets = (
                {
                    pt.frame_id: TrackDetection(
                        frame_id=pt.frame_id,
                        track_id=ins_track_id,
                        x1=pt.x - 15.0,
                        y1=pt.y - 30.0,
                        x2=pt.x + 15.0,
                        y2=pt.y,
                        confidence=0.9,
                    )
                    for pt in trajectories.get(ins_track_id, [])
                }
                if ins_track_id in trajectories
                else {}
            )
            ins_traj_pts = (
                {pt.frame_id: (pt.x, pt.y) for pt in trajectories.get(ins_track_id, [])}
                if ins_track_id in trajectories
                else {}
            )
            ins_ev = EventRecord(
                event_id=ins.insight_id,
                event_type=ins.insight_type,
                track_id=max(0, ins_track_id),
                zone_id=ins.zone_id or "scene",
                start_time=ins.start_time,
                end_time=ins.end_time,
                duration_seconds=ins.duration,
            )

            rel_clip = ""
            if cfg.extract_video_clips and total_frames > 0:
                rel_clip = f"clips/insight_{ins.insight_id}.mp4"
                extract_event_video_clip(
                    video_source=src_path,
                    output_clip_path=run_dir / rel_clip,
                    event=ins_ev,
                    fps=fps,
                    total_source_frames=total_frames,
                    pre_roll_seconds=cfg.clip_pre_roll_seconds,
                    post_roll_seconds=cfg.clip_post_roll_seconds,
                    scene_config=scene_cfg,
                    track_detections_by_frame=ins_track_dets,
                    trajectory_points_by_frame=ins_traj_pts,
                    camera_id=src_path.stem,
                    alert_text=f"{ins.severity.upper()}: {ins.insight_type}",
                    render_telemetry=cfg.render_clip_telemetry,
                )

            rel_sb = ""
            if cfg.extract_storyboards and total_frames > 0:
                rel_sb = f"storyboards/insight_{ins.insight_id}_storyboard.jpg"
                extract_event_storyboard(
                    video_source=src_path,
                    output_storyboard_path=run_dir / rel_sb,
                    event=ins_ev,
                    fps=fps,
                    total_source_frames=total_frames,
                    scene_config=scene_cfg,
                    track_detections_by_frame=ins_track_dets,
                    trajectory_points_by_frame=ins_traj_pts,
                    camera_id=src_path.stem,
                )

            enriched_insights.append(
                InsightRecord(
                    insight_id=ins.insight_id,
                    camera_id=ins.camera_id,
                    insight_type=ins.insight_type,
                    severity=ins.severity,
                    track_id=ins.track_id,
                    zone_id=ins.zone_id,
                    start_time=ins.start_time,
                    end_time=ins.end_time,
                    duration=ins.duration,
                    description=ins.description,
                    evidence_snapshot_path=ins.evidence_snapshot_path,
                    evidence_crop_path=ins.evidence_crop_path,
                    video_clip_path=rel_clip if rel_clip else None,
                    storyboard_path=rel_sb if rel_sb else None,
                    details=ins.details,
                )
            )
        insights = enriched_insights

    insights_json_path = write_insights_json(insights, run_dir / "insights.json")
    write_insights_csv(insights, run_dir / "insights.csv")

    report_md = generate_intelligence_report_markdown(
        insights,
        run_id=resolved_run_id,
        source_info={
            "source_path": str(src_path),
            "total_frames": total_frames,
            "elapsed_seconds": time.perf_counter() - t0,
        },
        zone_counts=zone_counts,
        line_counts=line_counts,
    )
    write_intelligence_report(report_md, run_dir / "intelligence_report.md")

    # Phase 10: Ingest into SQLite intelligence.db
    search_cfg = SearchConfig(
        source_name=src_path.name,
        detections_csv=str(detections_csv_path.resolve()),
        tracks_csv=str(tracks_csv_path.resolve()),
        tracking_summary_csv=str(track_summary_csv_path.resolve()),
        trajectories_csv=str(trajectories_csv_path.resolve()),
        trajectory_summary_csv=str(trajectory_summary_csv_path.resolve()),
        events_csv=str((run_dir / "events.csv").resolve()),
        evidence_root=str(run_dir.resolve()),
        zone_counts_csv=str(zone_counts_csv_path.resolve()),
        line_counts_csv=str(line_counts_csv_path.resolve()),
        insights_json=str(insights_json_path.resolve()),
        write_subproject_docs=False,
    )
    db_path = (run_dir / "intelligence.db").resolve()
    db_summary = ingest_intelligence_db(db_path, search_cfg, project_root=root)

    elapsed = time.perf_counter() - t0
    return PipelineRunResult(
        run_id=resolved_run_id,
        run_dir=str(run_dir),
        source_path=str(src_path),
        processed_frames=total_frames,
        detections_count=detections_count,
        tracks_count=db_summary.tracks_count,
        trajectories_count=db_summary.trajectories_count,
        zone_line_signals_count=len(signals),
        events_count=db_summary.events_count,
        zones_count=db_summary.zone_counts_rows,
        lines_count=db_summary.line_counts_rows,
        insights_count=len(insights),
        db_path=str(db_path),
        run_config_path=str(run_config_path),
        elapsed_seconds=elapsed,
    )


def run_multi_camera_pipeline(
    video_sources: list[Path | str],
    config: PipelineConfig | None = None,
    *,
    camera_ids: list[str] | None = None,
    output_root: Path | str | None = None,
    run_id: str | None = None,
    project_root: Path | None = None,
    detector: PersonDetector | None = None,
    tracker: PersonTracker | None = None,
    scene_configs_by_camera: Mapping[str, Path | str] | None = None,
    precomputed_tracks_by_camera: Mapping[str, Path | str] | None = None,
    precomputed_detections_by_camera: Mapping[str, Path | str] | None = None,
) -> MultiCameraPipelineRunResult:
    """Execute the CV pipeline across multiple camera streams into a unified intelligence.db."""
    if not video_sources:
        raise ValueError("video_sources must be non-empty")

    t0 = time.perf_counter()
    cfg = config if config is not None else PipelineConfig()
    root = project_root if project_root is not None else _PROJECT_ROOT

    resolved_run_id = (
        run_id.strip()
        if run_id and run_id.strip()
        else datetime.now(UTC).strftime("multicam_%Y%m%dT%H%M%S_") + uuid4().hex[:6]
    )
    runs_base = (
        Path(output_root).resolve()
        if output_root is not None
        else (root / "outputs" / "runs").resolve()
    )
    master_run_dir = (runs_base / resolved_run_id).resolve()
    master_run_dir.mkdir(parents=True, exist_ok=True)
    save_pipeline_config(cfg, master_run_dir / "run_config.yaml")

    cams_dir = (master_run_dir / "cameras").resolve()
    cams_dir.mkdir(parents=True, exist_ok=True)

    results: list[PipelineRunResult] = []
    search_configs: list[SearchConfig] = []
    cameras_meta: list[CameraMetadata] = []
    cam_names: list[str] = []

    for i, vsrc in enumerate(video_sources):
        src_p = Path(vsrc)
        cam_id = (
            camera_ids[i].strip()
            if camera_ids and i < len(camera_ids) and camera_ids[i].strip()
            else src_p.stem
        )
        cam_names.append(cam_id)

        cam_scene_cfg = (
            scene_configs_by_camera.get(cam_id)
            if scene_configs_by_camera and cam_id in scene_configs_by_camera
            else cfg.scene_config_path
        )
        cam_tracks = (
            precomputed_tracks_by_camera.get(cam_id)
            if precomputed_tracks_by_camera and cam_id in precomputed_tracks_by_camera
            else cfg.precomputed_tracks_csv
        )
        cam_dets = (
            precomputed_detections_by_camera.get(cam_id)
            if precomputed_detections_by_camera
            and cam_id in precomputed_detections_by_camera
            else cfg.precomputed_detections_csv
        )
        cam_pipeline_cfg = PipelineConfig(
            model_name=cfg.model_name,
            confidence_threshold=cfg.confidence_threshold,
            iou_threshold=cfg.iou_threshold,
            tracker_config=cfg.tracker_config,
            device=cfg.device,
            precomputed_detections_csv=str(cam_dets) if cam_dets else None,
            precomputed_tracks_csv=str(cam_tracks) if cam_tracks else None,
            scene_config_path=str(cam_scene_cfg),
            anchor=cfg.anchor,
            smoothing_window=cfg.smoothing_window,
            max_trail_length=cfg.max_trail_length,
            min_dwell_seconds=cfg.min_dwell_seconds,
            min_entry_frames=cfg.min_entry_frames,
            grace_frames=cfg.grace_frames,
            line_cooldown_frames=cfg.line_cooldown_frames,
            include_line_crossings=cfg.include_line_crossings,
            per_zone_min_dwell_seconds=cfg.per_zone_min_dwell_seconds,
            best_frame_strategy=cfg.best_frame_strategy,
            crop_padding_ratio=cfg.crop_padding_ratio,
            min_padding_px=cfg.min_padding_px,
            jpeg_quality=cfg.jpeg_quality,
            grid_bin_size_px=cfg.grid_bin_size_px,
            gaussian_sigma_px=cfg.gaussian_sigma_px,
            normalization_mode=cfg.normalization_mode,
            max_points_per_track_cell=cfg.max_points_per_track_cell,
            colormap=cfg.colormap,
            reference_frame_index=cfg.reference_frame_index,
            inspection_sample_count=cfg.inspection_sample_count,
            save_annotated_video=cfg.save_annotated_video,
            max_frames=cfg.max_frames,
            insight_config=cfg.insight_config,
        )

        res = run_pipeline(
            vsrc,
            cam_pipeline_cfg,
            output_root=cams_dir,
            run_id=cam_id,
            project_root=root,
            detector=detector,
            tracker=tracker,
        )
        results.append(res)

        cam_run_path = Path(res.run_dir)
        vinfo_path = cam_run_path / "video_info.json"
        fps, w, h, frames = 30.0, 1920, 1080, res.processed_frames
        if vinfo_path.is_file():
            try:
                vinfo = json.loads(vinfo_path.read_text(encoding="utf-8"))[0]
                fps = float(vinfo.get("fps", fps))
                w = int(vinfo.get("width", w))
                h = int(vinfo.get("height", h))
                frames = int(vinfo.get("frame_count", frames))
            except (json.JSONDecodeError, KeyError, IndexError, ValueError, OSError):
                pass

        cameras_meta.append(
            CameraMetadata(
                camera_id=cam_id,
                source_name=src_p.name,
                fps=fps,
                width=w,
                height=h,
                total_frames=frames,
            )
        )
        search_configs.append(
            SearchConfig(
                source_name=src_p.name,
                camera_id=cam_id,
                detections_csv=str((cam_run_path / "detections.csv").resolve()),
                tracks_csv=str((cam_run_path / "tracks.csv").resolve()),
                tracking_summary_csv=str(
                    (cam_run_path / "track_summary.csv").resolve()
                ),
                trajectories_csv=str((cam_run_path / "trajectories.csv").resolve()),
                trajectory_summary_csv=str(
                    (cam_run_path / "trajectory_summary.csv").resolve()
                ),
                events_csv=str((cam_run_path / "events.csv").resolve()),
                evidence_root=str(cam_run_path.resolve()),
                zone_counts_csv=str((cam_run_path / "zone_counts.csv").resolve()),
                line_counts_csv=str((cam_run_path / "line_counts.csv").resolve()),
                insights_json=(
                    str((cam_run_path / "insights.json").resolve())
                    if (cam_run_path / "insights.json").is_file()
                    else None
                ),
                write_subproject_docs=False,
            )
        )

    master_db = (master_run_dir / "intelligence.db").resolve()
    db_summary = ingest_multi_camera_intelligence_db(
        master_db,
        search_configs,
        project_root=root,
        clear_existing=True,
        cameras_metadata=cameras_meta,
    )

    events_by_cam: dict[str, list[Any]] = {}
    for c_name in cam_names:
        events_by_cam[c_name] = query_events(master_db, camera_id=c_name, limit=5000)

    ins_cfg = cfg.insight_config if cfg.insight_config is not None else InsightConfig()
    if len(events_by_cam) > 1:
        cross_cam_insights = detect_cross_camera_transitions(events_by_cam, ins_cfg)
        if cross_cam_insights:
            insert_insights(master_db, cross_cam_insights)

    all_master_insights = query_insights(master_db, limit=1000)
    master_report_md = generate_intelligence_report_markdown(
        [
            InsightRecord(
                insight_id=i.insight_id,
                camera_id=i.camera_id,
                insight_type=i.insight_type,
                severity=i.severity,
                track_id=i.track_id,
                zone_id=i.zone_id,
                start_time=i.start_time,
                end_time=i.end_time,
                duration=i.duration,
                description=i.description,
                evidence_snapshot_path=i.evidence_snapshot_path,
                evidence_crop_path=i.evidence_crop_path,
                video_clip_path=i.video_clip_path,
                storyboard_path=i.storyboard_path,
                details=i.details,
            )
            for i in all_master_insights
        ],
        run_id=resolved_run_id,
        source_info={
            "source_path": f"{len(cam_names)} multi-camera streams ({', '.join(cam_names)})",
            "total_frames": sum(r.processed_frames for r in results),
            "elapsed_seconds": time.perf_counter() - t0,
        },
        camera_summaries=query_camera_summary(master_db),
    )
    write_intelligence_report(
        master_report_md, master_run_dir / "intelligence_report.md"
    )
    write_insights_json(
        [
            InsightRecord(
                insight_id=i.insight_id,
                camera_id=i.camera_id,
                insight_type=i.insight_type,
                severity=i.severity,
                track_id=i.track_id,
                zone_id=i.zone_id,
                start_time=i.start_time,
                end_time=i.end_time,
                duration=i.duration,
                description=i.description,
                evidence_snapshot_path=i.evidence_snapshot_path,
                evidence_crop_path=i.evidence_crop_path,
                video_clip_path=i.video_clip_path,
                storyboard_path=i.storyboard_path,
                details=i.details,
            )
            for i in all_master_insights
        ],
        master_run_dir / "insights.json",
    )

    elapsed = time.perf_counter() - t0
    total_dets = sum(r.detections_count for r in results)
    total_trks = db_summary.tracks_count
    total_evs = db_summary.events_count

    summary_payload = {
        "run_id": resolved_run_id,
        "run_dir": str(master_run_dir),
        "cameras": cam_names,
        "cameras_count": len(cam_names),
        "total_detections": total_dets,
        "total_tracks": total_trks,
        "total_events": total_evs,
        "total_insights": len(all_master_insights),
        "db_path": str(master_db),
        "elapsed_seconds": round(elapsed, 4),
    }
    (master_run_dir / "multicam_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )

    return MultiCameraPipelineRunResult(
        run_id=resolved_run_id,
        run_dir=str(master_run_dir),
        cameras=cam_names,
        camera_results=results,
        db_path=str(master_db),
        cameras_count=len(cam_names),
        total_detections=total_dets,
        total_tracks=total_trks,
        total_events=total_evs,
        total_insights=len(all_master_insights),
        elapsed_seconds=elapsed,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for python -m event_intelligence.pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Searchable Multi-Camera Event Intelligence CV pipeline "
            "end-to-end into outputs/runs/<run_id>/."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to the input video file or MOT17 frame-sequence directory.",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        nargs="+",
        default=None,
        help="Paths to multiple video files or directories for multi-camera processing.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Optional camera identifiers matching --videos (e.g. entrance concourse).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Path to pipeline YAML configuration (default: configs/pipeline.yaml).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for run outputs (default: outputs/runs).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run identifier (writes into <output-root>/<run-id>/).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames to process.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing annotated_video.mp4.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print PipelineRunResult as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for python -m event_intelligence.pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    base_cfg = (
        load_pipeline_config(args.config)
        if args.config is not None and args.config.is_file()
        else PipelineConfig()
    )
    cfg = PipelineConfig(
        model_name=base_cfg.model_name,
        confidence_threshold=base_cfg.confidence_threshold,
        iou_threshold=base_cfg.iou_threshold,
        tracker_config=base_cfg.tracker_config,
        device=base_cfg.device,
        precomputed_detections_csv=base_cfg.precomputed_detections_csv,
        precomputed_tracks_csv=base_cfg.precomputed_tracks_csv,
        scene_config_path=base_cfg.scene_config_path,
        anchor=base_cfg.anchor,
        smoothing_window=base_cfg.smoothing_window,
        max_trail_length=base_cfg.max_trail_length,
        min_dwell_seconds=base_cfg.min_dwell_seconds,
        min_entry_frames=base_cfg.min_entry_frames,
        grace_frames=base_cfg.grace_frames,
        line_cooldown_frames=base_cfg.line_cooldown_frames,
        include_line_crossings=base_cfg.include_line_crossings,
        per_zone_min_dwell_seconds=base_cfg.per_zone_min_dwell_seconds,
        best_frame_strategy=base_cfg.best_frame_strategy,
        crop_padding_ratio=base_cfg.crop_padding_ratio,
        min_padding_px=base_cfg.min_padding_px,
        jpeg_quality=base_cfg.jpeg_quality,
        grid_bin_size_px=base_cfg.grid_bin_size_px,
        gaussian_sigma_px=base_cfg.gaussian_sigma_px,
        normalization_mode=base_cfg.normalization_mode,
        max_points_per_track_cell=base_cfg.max_points_per_track_cell,
        colormap=base_cfg.colormap,
        reference_frame_index=base_cfg.reference_frame_index,
        inspection_sample_count=base_cfg.inspection_sample_count,
        save_annotated_video=(
            False if args.no_video else base_cfg.save_annotated_video
        ),
        max_frames=(
            args.max_frames if args.max_frames is not None else base_cfg.max_frames
        ),
    )

    videos: list[Path] = []
    if args.videos:
        videos = [Path(v) for v in args.videos]
    elif args.video is not None:
        videos = [Path(args.video)]
    else:
        videos = [Path("data/fallback_clips/people-detection.mp4")]

    if len(videos) > 1:
        res_multi = run_multi_camera_pipeline(
            videos,
            cfg,
            camera_ids=args.cameras,
            output_root=args.output_root,
            run_id=args.run_id,
        )
        if args.json:
            print(json.dumps(res_multi.to_dict(), indent=2))
        else:
            print(
                f"Completed multi-camera pipeline run '{res_multi.run_id}' -> {res_multi.run_dir} "
                f"({res_multi.cameras_count} cameras: {', '.join(res_multi.cameras)}, "
                f"detections={res_multi.total_detections}, tracks={res_multi.total_tracks}, "
                f"events={res_multi.total_events}, elapsed={res_multi.elapsed_seconds:.2f}s)"
            )
        return 0

    result = run_pipeline(
        videos[0],
        cfg,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(
            f"Completed pipeline run '{result.run_id}' -> {result.run_dir} "
            f"(frames={result.processed_frames}, detections={result.detections_count}, "
            f"tracks={result.tracks_count}, events={result.events_count}, "
            f"elapsed={result.elapsed_seconds:.2f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
