"""Phase 7: Visual Evidence best-frame selector, padded person cropper,
annotated snapshot renderer, enriched events.csv exporter, path verifier,
and single-pass sequential video CLI entrypoint.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from track_people import FrameTracks, TrackDetection
except ImportError:
    _subproject_3_scripts = (
        Path(__file__).resolve().parents[2] / "03_person_tracking" / "scripts"
    )
    if str(_subproject_3_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_3_scripts))
    from track_people import FrameTracks, TrackDetection

try:
    from generate_dwell_events import VALID_EVENT_TYPES, EventRecord
except ImportError:
    _subproject_6_scripts = (
        Path(__file__).resolve().parents[2] / "06_dwell_time_events" / "scripts"
    )
    if str(_subproject_6_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_6_scripts))
    from generate_dwell_events import VALID_EVENT_TYPES, EventRecord

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
    """Configuration for Phase 7 best-frame selection and snapshot/crop extraction."""

    best_frame_strategy: str = "highest_confidence"
    crop_padding_ratio: float = 0.10
    min_padding_px: int = 8
    jpeg_quality: int = 92
    overlay_scene_geometry: bool = True
    contact_sheet_columns: int = 4
    contact_sheet_max_items: int = 12

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
