"""Phase 7: Visual Evidence best-frame selector, padded person cropper,
annotated snapshot renderer, enriched events.csv exporter, path verifier,
and single-pass sequential video CLI entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from event_intelligence.events import (
    VALID_EVENT_TYPES,
    EventRecord,
    read_events_csv,
)
from event_intelligence.spatial import (
    SceneConfig,
    draw_zones_and_lines,
    load_scene_config,
)
from event_intelligence.tracking import FrameTracks, TrackDetection, read_tracks_csv
from event_intelligence.video import VideoSource

EVIDENCE_EVENTS_CSV_HEADER: list[str] = [
    "event_id",
    "event_type",
    "track_id",
    "zone_id",
    "start_time",
    "end_time",
    "duration_seconds",
    "snapshot_path",
    "crop_path",
]

VALID_BEST_FRAME_STRATEGIES: tuple[str, ...] = (
    "highest_confidence",
    "midpoint",
    "start_frame",
    "largest_area",
)


@dataclass(frozen=True)
class EvidenceConfig:
    """Configuration for Phase 7/11e best-frame selection, snapshot/crop, video clip, and storyboard extraction."""

    best_frame_strategy: str = "highest_confidence"
    crop_padding_ratio: float = 0.10
    min_padding_px: int = 8
    jpeg_quality: int = 92
    overlay_scene_geometry: bool = True
    contact_sheet_columns: int = 4
    contact_sheet_max_items: int = 12
    extract_video_clips: bool = False
    clip_pre_roll_seconds: float = 1.0
    clip_post_roll_seconds: float = 1.0
    render_clip_telemetry: bool = True
    extract_storyboards: bool = False
    storyboard_stages: tuple[str, ...] = ("start", "peak", "end")

    def __post_init__(self) -> None:
        if self.best_frame_strategy not in VALID_BEST_FRAME_STRATEGIES:
            raise ValueError(
                f"best_frame_strategy must be one of {VALID_BEST_FRAME_STRATEGIES}, "
                f"got {self.best_frame_strategy!r}"
            )
        if self.crop_padding_ratio < 0.0:
            raise ValueError(
                f"crop_padding_ratio must be >= 0.0, got {self.crop_padding_ratio}"
            )
        if self.min_padding_px < 0:
            raise ValueError(f"min_padding_px must be >= 0, got {self.min_padding_px}")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(
                f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}"
            )
        if self.contact_sheet_columns < 1:
            raise ValueError(
                f"contact_sheet_columns must be >= 1, got {self.contact_sheet_columns}"
            )
        if self.contact_sheet_max_items < 1:
            raise ValueError(
                f"contact_sheet_max_items must be >= 1, got {self.contact_sheet_max_items}"
            )
        if self.clip_pre_roll_seconds < 0.0:
            raise ValueError(
                f"clip_pre_roll_seconds must be >= 0.0, got {self.clip_pre_roll_seconds}"
            )
        if self.clip_post_roll_seconds < 0.0:
            raise ValueError(
                f"clip_post_roll_seconds must be >= 0.0, got {self.clip_post_roll_seconds}"
            )


@dataclass(frozen=True)
class BestFrameSelection:
    """Selected best frame and bounding box for a single structured event."""

    event_id: str
    track_id: int
    frame_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    strategy_used: str
    used_nearest_fallback: bool = False

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class EvidenceEventRecord:
    """An EventRecord enriched with relative/portable snapshot_path and crop_path."""

    event_id: str
    event_type: str
    track_id: int
    zone_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    snapshot_path: str
    crop_path: str
    best_frame_id: int = 0
    best_confidence: float = 0.0
    clipped_at_frame_edge: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"event_type must be one of {VALID_EVENT_TYPES}, got {self.event_type!r}"
            )
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if not self.zone_id or not self.zone_id.strip():
            raise ValueError("zone_id must be a non-empty string")
        if self.start_time < 0.0:
            raise ValueError(f"start_time must be >= 0.0, got {self.start_time}")
        if self.end_time < self.start_time - 1e-9:
            raise ValueError(
                f"end_time ({self.end_time}) must be >= start_time ({self.start_time})"
            )
        if self.duration_seconds < 0.0:
            raise ValueError(
                f"duration_seconds must be >= 0.0, got {self.duration_seconds}"
            )
        if not self.snapshot_path or not self.snapshot_path.strip():
            raise ValueError("snapshot_path must be a non-empty string")
        if not self.crop_path or not self.crop_path.strip():
            raise ValueError("crop_path must be a non-empty string")

    @classmethod
    def from_event(
        cls,
        event: EventRecord,
        *,
        snapshot_path: str,
        crop_path: str,
        best_frame_id: int = 0,
        best_confidence: float = 0.0,
        clipped_at_frame_edge: bool = False,
    ) -> EvidenceEventRecord:
        return cls(
            event_id=event.event_id,
            event_type=event.event_type,
            track_id=event.track_id,
            zone_id=event.zone_id,
            start_time=event.start_time,
            end_time=event.end_time,
            duration_seconds=event.duration_seconds,
            snapshot_path=snapshot_path,
            crop_path=crop_path,
            best_frame_id=best_frame_id,
            best_confidence=best_confidence,
            clipped_at_frame_edge=clipped_at_frame_edge,
        )

    def to_row(self) -> list[str]:
        return [
            self.event_id,
            self.event_type,
            str(self.track_id),
            self.zone_id,
            f"{self.start_time:.4f}",
            f"{self.end_time:.4f}",
            f"{self.duration_seconds:.4f}",
            self.snapshot_path,
            self.crop_path,
        ]


def load_evidence_config(config_path: Path | str) -> EvidenceConfig:
    """Load an EvidenceConfig from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evidence config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)

    if raw is None:
        return EvidenceConfig()
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")

    defaults = EvidenceConfig()
    return EvidenceConfig(
        best_frame_strategy=str(
            raw.get("best_frame_strategy", defaults.best_frame_strategy)
        ),
        crop_padding_ratio=float(
            raw.get("crop_padding_ratio", defaults.crop_padding_ratio)
        ),
        min_padding_px=int(raw.get("min_padding_px", defaults.min_padding_px)),
        jpeg_quality=int(raw.get("jpeg_quality", defaults.jpeg_quality)),
        overlay_scene_geometry=bool(
            raw.get("overlay_scene_geometry", defaults.overlay_scene_geometry)
        ),
        contact_sheet_columns=int(
            raw.get("contact_sheet_columns", defaults.contact_sheet_columns)
        ),
        contact_sheet_max_items=int(
            raw.get("contact_sheet_max_items", defaults.contact_sheet_max_items)
        ),
        extract_video_clips=bool(
            raw.get("extract_video_clips", defaults.extract_video_clips)
        ),
        clip_pre_roll_seconds=float(
            raw.get("clip_pre_roll_seconds", defaults.clip_pre_roll_seconds)
        ),
        clip_post_roll_seconds=float(
            raw.get("clip_post_roll_seconds", defaults.clip_post_roll_seconds)
        ),
        render_clip_telemetry=bool(
            raw.get("render_clip_telemetry", defaults.render_clip_telemetry)
        ),
        extract_storyboards=bool(
            raw.get("extract_storyboards", defaults.extract_storyboards)
        ),
        storyboard_stages=tuple(
            str(s) for s in raw.get("storyboard_stages", defaults.storyboard_stages)
        ),
    )


def save_evidence_config(config: EvidenceConfig, config_path: Path | str) -> Path:
    """Save an EvidenceConfig to a YAML file."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
    return path


def write_evidence_events_csv(
    records: Sequence[EvidenceEventRecord], csv_path: Path | str
) -> Path:
    """Write enriched EvidenceEventRecord rows to the 9-column Phase 7 events.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVIDENCE_EVENTS_CSV_HEADER)
        for rec in records:
            writer.writerow(rec.to_row())
    return path


def read_evidence_events_csv(csv_path: Path | str) -> list[EvidenceEventRecord]:
    """Read enriched EvidenceEventRecord rows from the 9-column Phase 7 events.csv."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Enriched events.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EVIDENCE_EVENTS_CSV_HEADER:
            raise ValueError(
                f"Unexpected events.csv header in {path}: got {header}, "
                f"expected {EVIDENCE_EVENTS_CSV_HEADER}"
            )

        records: list[EvidenceEventRecord] = []
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(EVIDENCE_EVENTS_CSV_HEADER):
                raise ValueError(
                    f"Malformed row in {path}: expected {len(EVIDENCE_EVENTS_CSV_HEADER)} "
                    f"columns, got {len(row)}: {row}"
                )
            records.append(
                EvidenceEventRecord(
                    event_id=row[0].strip(),
                    event_type=row[1].strip(),
                    track_id=int(row[2].strip()),
                    zone_id=row[3].strip(),
                    start_time=float(row[4].strip()),
                    end_time=float(row[5].strip()),
                    duration_seconds=float(row[6].strip()),
                    snapshot_path=row[7].strip(),
                    crop_path=row[8].strip(),
                )
            )
    return records


def index_track_detections(
    frame_tracks: Sequence[FrameTracks],
) -> dict[int, dict[int, TrackDetection]]:
    """Index person detections by track_id -> {frame_id: TrackDetection}."""
    by_track: dict[int, dict[int, TrackDetection]] = {}
    for ft in frame_tracks:
        for det in ft.tracks:
            track_map = by_track.setdefault(det.track_id, {})
            existing = track_map.get(det.frame_id)
            if existing is None or det.confidence > existing.confidence:
                track_map[det.frame_id] = det
    return by_track


def select_best_frame_for_event(
    event: EventRecord,
    track_detections: Mapping[int, Mapping[int, TrackDetection]],
    *,
    fps: float,
    strategy: str = "highest_confidence",
) -> BestFrameSelection:
    """Select the representative best frame and bounding box for an EventRecord.

    Rule:
    1. Convert the event time window ``[start_time, end_time]`` into a frame window
       ``[min_frame, max_frame]`` via ``round(time * fps)``.
    2. Collect all tracked observations of ``event.track_id`` in ``[min_frame, max_frame]``.
    3. If non-empty, rank candidates according to ``strategy``:
       - ``"highest_confidence"``: highest detector confidence, tie-breaking by
         largest bounding-box area ``(x2 - x1) * (y2 - y1)``, then earliest ``frame_id``.
       - ``"largest_area"``: largest bounding-box area, tie-breaking by highest
         confidence, then earliest ``frame_id``.
       - ``"midpoint"``: frame closest to ``(min_frame + max_frame) / 2.0``,
         tie-breaking by highest confidence, then earliest ``frame_id``.
       - ``"start_frame"``: earliest ``frame_id`` in the event window.
    4. If ``[min_frame, max_frame]`` contains no observation for ``event.track_id``
       (e.g. an interpolated gap), fall back to the temporally closest tracked
       frame of ``event.track_id`` and mark ``used_nearest_fallback=True``.
    """
    if fps <= 0.0:
        raise ValueError(f"fps must be > 0.0, got {fps}")
    if strategy not in VALID_BEST_FRAME_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {VALID_BEST_FRAME_STRATEGIES}, got {strategy!r}"
        )

    per_frame = track_detections.get(event.track_id)
    if not per_frame:
        raise KeyError(
            f"No tracked detections found for track_id={event.track_id} "
            f"(event_id={event.event_id!r})"
        )

    min_frame = max(0, round(event.start_time * fps))
    max_frame = max(min_frame, round(event.end_time * fps))

    candidates: list[tuple[int, TrackDetection]] = [
        (fid, det)
        for fid, det in sorted(per_frame.items())
        if min_frame <= fid <= max_frame
    ]

    used_fallback = False
    if not candidates:
        used_fallback = True
        mid_frame = (min_frame + max_frame) / 2.0
        chosen_fid, chosen_det = min(
            per_frame.items(),
            key=lambda item: (abs(item[0] - mid_frame), -item[1].confidence, item[0]),
        )
    elif strategy == "highest_confidence":
        chosen_fid, chosen_det = max(
            candidates,
            key=lambda item: (
                item[1].confidence,
                (item[1].x2 - item[1].x1) * (item[1].y2 - item[1].y1),
                -item[0],
            ),
        )
    elif strategy == "largest_area":
        chosen_fid, chosen_det = max(
            candidates,
            key=lambda item: (
                (item[1].x2 - item[1].x1) * (item[1].y2 - item[1].y1),
                item[1].confidence,
                -item[0],
            ),
        )
    elif strategy == "midpoint":
        mid_frame = (min_frame + max_frame) / 2.0
        chosen_fid, chosen_det = min(
            candidates,
            key=lambda item: (
                abs(item[0] - mid_frame),
                -item[1].confidence,
                item[0],
            ),
        )
    else:  # "start_frame"
        chosen_fid, chosen_det = min(candidates, key=lambda item: item[0])

    return BestFrameSelection(
        event_id=event.event_id,
        track_id=event.track_id,
        frame_id=chosen_fid,
        x1=chosen_det.x1,
        y1=chosen_det.y1,
        x2=chosen_det.x2,
        y2=chosen_det.y2,
        confidence=chosen_det.confidence,
        strategy_used=strategy,
        used_nearest_fallback=used_fallback,
    )


def extract_person_crop(
    frame: np.ndarray,
    selection: BestFrameSelection,
    *,
    padding_ratio: float = 0.10,
    min_padding_px: int = 8,
) -> tuple[np.ndarray, bool]:
    """Extract a padded person crop from ``frame`` clamped to frame boundaries.

    Returns ``(crop_bgr, clipped_at_frame_edge)``.
    """
    if frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"Invalid frame shape for cropping: {frame.shape}")

    height, width = int(frame.shape[0]), int(frame.shape[1])
    box_w = max(1.0, selection.x2 - selection.x1)
    box_h = max(1.0, selection.y2 - selection.y1)

    pad_x = max(min_padding_px, round(box_w * padding_ratio))
    pad_y = max(min_padding_px, round(box_h * padding_ratio))

    raw_x1 = round(selection.x1) - pad_x
    raw_y1 = round(selection.y1) - pad_y
    raw_x2 = round(selection.x2) + pad_x
    raw_y2 = round(selection.y2) + pad_y

    clipped_at_frame_edge = (
        raw_x1 < 0 or raw_y1 < 0 or raw_x2 > width or raw_y2 > height
    )

    crop_x1 = max(0, min(width - 1, raw_x1))
    crop_y1 = max(0, min(height - 1, raw_y1))
    crop_x2 = max(crop_x1 + 1, min(width, raw_x2))
    crop_y2 = max(crop_y1 + 1, min(height, raw_y2))

    return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), clipped_at_frame_edge


def _event_type_color(event_type: str) -> tuple[int, int, int]:
    """Return a distinct BGR highlight color per event_type."""
    palette: dict[str, tuple[int, int, int]] = {
        "dwell_time_event": (0, 215, 255),  # gold
        "zone_entry": (80, 220, 80),  # green
        "zone_exit": (60, 120, 255),  # orange-red
        "line_crossing": (255, 200, 50),  # cyan-blue
    }
    return palette.get(event_type, (0, 255, 255))


def render_event_snapshot(
    frame: np.ndarray,
    event: EventRecord,
    selection: BestFrameSelection,
    *,
    scene_config: SceneConfig | None = None,
) -> np.ndarray:
    """Render an annotated full-frame snapshot proving ``event`` at ``selection.frame_id``."""
    canvas = (
        draw_zones_and_lines(frame, scene_config)
        if scene_config is not None
        else frame.copy()
    )

    color = _event_type_color(event.event_type)
    pt1 = (round(selection.x1), round(selection.y1))
    pt2 = (round(selection.x2), round(selection.y2))
    cv2.rectangle(canvas, pt1, pt2, color, 2, cv2.LINE_AA)

    anchor = (round((selection.x1 + selection.x2) / 2.0), round(selection.y2))
    cv2.circle(canvas, anchor, 5, color, -1, cv2.LINE_AA)

    label = (
        f"{event.event_id} | {event.event_type} | T{event.track_id} "
        f"| {event.zone_id} | f={selection.frame_id} ({selection.confidence:.2f})"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    bx1 = max(4, min(canvas.shape[1] - tw - 12, pt1[0]))
    by2 = max(th + baseline + 10, pt1[1] - 4)
    by1 = max(2, by2 - th - baseline - 8)
    bx2 = min(canvas.shape[1] - 2, bx1 + tw + 10)

    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (bx1 + 5, by2 - baseline - 4),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def render_evidence_contact_sheet(
    items: Sequence[tuple[np.ndarray, str]],
    *,
    columns: int = 4,
    thumb_size: tuple[int, int] = (200, 240),
) -> np.ndarray:
    """Render a grid contact sheet of person crops and event captions."""
    cols = max(1, columns)
    tw, th = thumb_size
    header_h = 36

    if not items:
        blank = np.full((th + header_h, tw * cols, 3), 28, dtype=np.uint8)
        cv2.putText(
            blank,
            "No visual evidence items",
            (16, (th + header_h) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return blank

    rows = (len(items) + cols - 1) // cols
    cell_h = th + header_h
    sheet = np.full((rows * cell_h, cols * tw, 3), 24, dtype=np.uint8)

    for idx, (crop, caption) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = c * tw
        y0 = r * cell_h

        cell = np.full((cell_h, tw, 3), 32, dtype=np.uint8)
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            scale = min((tw - 12) / max(1, cw), (th - 12) / max(1, ch))
            nw = max(1, round(cw * scale))
            nh = max(1, round(ch * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = (tw - nw) // 2
            oy = header_h + (th - nh) // 2
            cell[oy : oy + nh, ox : ox + nw] = resized

        cv2.rectangle(cell, (0, 0), (tw - 1, header_h), (48, 48, 48), -1)
        cv2.rectangle(cell, (0, 0), (tw - 1, cell_h - 1), (80, 80, 80), 1)
        cv2.putText(
            cell,
            caption[:28],
            (6, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        sheet[y0 : y0 + cell_h, x0 : x0 + tw] = cell

    return sheet


def verify_evidence_records(
    records: Sequence[EvidenceEventRecord],
    *,
    base_dir: Path | str,
) -> dict[str, Any]:
    """Verify that all ``snapshot_path`` and ``crop_path`` entries resolve to non-empty files."""
    root = Path(base_dir)
    missing_snapshots: list[str] = []
    missing_crops: list[str] = []
    resolved_snapshots = 0
    resolved_crops = 0

    for rec in records:
        snap_file = root / rec.snapshot_path
        if snap_file.is_file() and snap_file.stat().st_size > 0:
            resolved_snapshots += 1
        else:
            missing_snapshots.append(rec.snapshot_path)

        crop_file = root / rec.crop_path
        if crop_file.is_file() and crop_file.stat().st_size > 0:
            resolved_crops += 1
        else:
            missing_crops.append(rec.crop_path)

    return {
        "total_records": len(records),
        "resolved_snapshots": resolved_snapshots,
        "resolved_crops": resolved_crops,
        "missing_or_empty_snapshots": missing_snapshots,
        "missing_or_empty_crops": missing_crops,
        "all_paths_valid": not missing_snapshots and not missing_crops,
    }


def run_visual_evidence(
    *,
    source_path: Path,
    events_csv_path: Path,
    tracks_csv_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
    scene_config_path: Path | None = None,
    best_frame_strategy_override: str | None = None,
    crop_padding_ratio_override: float | None = None,
    min_padding_px_override: int | None = None,
    jpeg_quality_override: int | None = None,
) -> dict[str, Any]:
    """Execute the complete Phase 7 visual evidence extraction pipeline."""
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    crops_dir = output_dir / "crops"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = (
        load_evidence_config(config_path)
        if config_path is not None and config_path.is_file()
        else EvidenceConfig()
    )
    cfg = EvidenceConfig(
        best_frame_strategy=(
            best_frame_strategy_override
            if best_frame_strategy_override is not None
            else base_cfg.best_frame_strategy
        ),
        crop_padding_ratio=(
            crop_padding_ratio_override
            if crop_padding_ratio_override is not None
            else base_cfg.crop_padding_ratio
        ),
        min_padding_px=(
            min_padding_px_override
            if min_padding_px_override is not None
            else base_cfg.min_padding_px
        ),
        jpeg_quality=(
            jpeg_quality_override
            if jpeg_quality_override is not None
            else base_cfg.jpeg_quality
        ),
        overlay_scene_geometry=base_cfg.overlay_scene_geometry,
        contact_sheet_columns=base_cfg.contact_sheet_columns,
        contact_sheet_max_items=base_cfg.contact_sheet_max_items,
    )

    scene_cfg: SceneConfig | None = None
    if (
        cfg.overlay_scene_geometry
        and scene_config_path is not None
        and scene_config_path.is_file()
    ):
        scene_cfg = load_scene_config(scene_config_path)

    events = read_events_csv(events_csv_path)
    frame_tracks = read_tracks_csv(tracks_csv_path)
    indexed_tracks = index_track_detections(frame_tracks)

    with VideoSource(source_path) as vs:
        meta = vs.metadata
        fps = float(meta.fps) if meta.fps > 0 else 12.0

        selections: list[BestFrameSelection] = [
            select_best_frame_for_event(
                ev,
                indexed_tracks,
                fps=fps,
                strategy=cfg.best_frame_strategy,
            )
            for ev in events
        ]

        by_frame: dict[int, list[tuple[int, EventRecord, BestFrameSelection]]] = {}
        for idx, (ev, sel) in enumerate(zip(events, selections, strict=True)):
            by_frame.setdefault(sel.frame_id, []).append((idx, ev, sel))

        max_needed_frame = max(by_frame.keys()) if by_frame else -1
        jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]

        enriched_by_idx: dict[int, EvidenceEventRecord] = {}
        contact_items_by_idx: dict[int, tuple[np.ndarray, str]] = {}

        for frame_id, frame in vs.iter_frames():
            if frame_id in by_frame:
                for idx, ev, sel in by_frame[frame_id]:
                    rel_snap = f"snapshots/{ev.event_id}_{ev.track_id}.jpg"
                    rel_crop = f"crops/{ev.event_id}_{ev.track_id}.jpg"

                    snapshot_bgr = render_event_snapshot(
                        frame,
                        ev,
                        sel,
                        scene_config=scene_cfg,
                    )
                    crop_bgr, clipped = extract_person_crop(
                        frame,
                        sel,
                        padding_ratio=cfg.crop_padding_ratio,
                        min_padding_px=cfg.min_padding_px,
                    )

                    cv2.imwrite(
                        str(output_dir / rel_snap),
                        snapshot_bgr,
                        jpeg_params,
                    )
                    cv2.imwrite(
                        str(output_dir / rel_crop),
                        crop_bgr,
                        jpeg_params,
                    )

                    enriched_by_idx[idx] = EvidenceEventRecord.from_event(
                        ev,
                        snapshot_path=rel_snap,
                        crop_path=rel_crop,
                        best_frame_id=sel.frame_id,
                        best_confidence=sel.confidence,
                        clipped_at_frame_edge=clipped,
                    )
                    caption = f"{ev.event_id} {ev.event_type[:10]} T{ev.track_id}"
                    contact_items_by_idx[idx] = (crop_bgr, caption)

            if frame_id >= max_needed_frame:
                break

    if len(enriched_by_idx) != len(events):
        missing_events = [
            events[i].event_id for i in range(len(events)) if i not in enriched_by_idx
        ]
        raise RuntimeError(
            f"Video ended before extracting evidence for events: {missing_events}"
        )

    enriched_records = [enriched_by_idx[i] for i in range(len(events))]
    verification = verify_evidence_records(enriched_records, base_dir=output_dir)
    if not verification["all_paths_valid"]:
        raise RuntimeError(f"Evidence path verification failed: {verification}")

    events_out_path = write_evidence_events_csv(
        enriched_records, output_dir / "events.csv"
    )
    config_out_path = save_evidence_config(cfg, output_dir / "evidence.yaml")

    contact_items = [
        contact_items_by_idx[i]
        for i in range(min(len(events), cfg.contact_sheet_max_items))
    ]
    contact_sheet = render_evidence_contact_sheet(
        contact_items,
        columns=cfg.contact_sheet_columns,
    )
    contact_sheet_path = output_dir / "evidence_contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet_path), contact_sheet, jpeg_params)

    elapsed_s = time.perf_counter() - t0
    clipped_count = sum(1 for r in enriched_records if r.clipped_at_frame_edge)
    fallback_count = sum(1 for s in selections if s.used_nearest_fallback)
    mean_conf = (
        float(np.mean([s.confidence for s in selections])) if selections else 0.0
    )

    event_counts_by_type: dict[str, int] = {}
    for rec in enriched_records:
        event_counts_by_type[rec.event_type] = (
            event_counts_by_type.get(rec.event_type, 0) + 1
        )

    summary: dict[str, Any] = {
        "source_path": str(source_path),
        "events_csv_input": str(events_csv_path),
        "tracks_csv_input": str(tracks_csv_path),
        "events_csv_output": str(events_out_path),
        "evidence_config_output": str(config_out_path),
        "contact_sheet_output": str(contact_sheet_path),
        "best_frame_strategy": cfg.best_frame_strategy,
        "crop_padding_ratio": cfg.crop_padding_ratio,
        "min_padding_px": cfg.min_padding_px,
        "jpeg_quality": cfg.jpeg_quality,
        "total_events": len(enriched_records),
        "unique_frames_decoded": len(by_frame),
        "event_counts_by_type": event_counts_by_type,
        "mean_best_confidence": round(mean_conf, 4),
        "clipped_crop_count": clipped_count,
        "nearest_fallback_count": fallback_count,
        "verification": verification,
        "elapsed_seconds": round(elapsed_s, 4),
    }
    summary_path = output_dir / "visual_evidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for Phase 7 visual evidence extraction."""
    parser = argparse.ArgumentParser(
        description="Phase 7: Extract best-frame snapshots and padded person crops per event."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        required=True,
        help="Path to the Phase 6 events.csv input file.",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to the Phase 3 tracks.csv input file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "evidence.yaml",
        help="Path to evidence.yaml configuration file.",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "05_zone_line_reasoning"
            / "configs"
            / "scene_config.yaml"
        ),
        help="Optional path to Phase 5 scene_config.yaml for zone/line overlays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "default_run",
        help="Directory to write snapshots/, crops/, enriched events.csv, and summary.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(VALID_BEST_FRAME_STRATEGIES),
        default=None,
        help="Optional override for best_frame_strategy.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=None,
        help="Optional override for crop_padding_ratio.",
    )
    parser.add_argument(
        "--min-padding-px",
        type=int,
        default=None,
        help="Optional override for min_padding_px.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="Optional override for jpeg_quality (1-100).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 7 visual evidence extraction."""
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_visual_evidence(
        source_path=args.source,
        events_csv_path=args.events,
        tracks_csv_path=args.tracks,
        output_dir=args.output_dir,
        config_path=args.config,
        scene_config_path=args.scene_config,
        best_frame_strategy_override=args.strategy,
        crop_padding_ratio_override=args.padding_ratio,
        min_padding_px_override=args.min_padding_px,
        jpeg_quality_override=args.jpeg_quality,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
