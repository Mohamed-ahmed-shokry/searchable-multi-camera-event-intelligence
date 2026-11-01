# Subproject 07 — Visual Evidence (`cv_projects/07_visual_evidence/`)

## Overview

Phase 7 attaches **auditable visual proof** to every structured behavioral event (`zone_entry`, `zone_exit`, `dwell_time_event`, `line_crossing`) produced in Phase 6:
1. **Best-Frame Selection (`select_best_frame_for_event`)**:
   - Maps each event's `[start_time, end_time]` window to `[min_frame, max_frame]` using `round(time * fps)`.
   - Evaluates all tracked person boxes (`TrackDetection`) for `event.track_id` inside `[min_frame, max_frame]`.
   - Supports 4 configurable strategies in `configs/evidence.yaml` (`best_frame_strategy`):
     - `highest_confidence` (*default*): highest detector confidence, tie-breaking by largest bounding-box area `(x2 - x1) * (y2 - y1)`, then earliest `frame_id`.
     - `largest_area`: largest bounding-box area, tie-breaking by highest confidence, then earliest `frame_id`.
     - `midpoint`: frame closest to the temporal midpoint `(min_frame + max_frame) / 2.0`, tie-breaking by highest confidence.
     - `start_frame`: earliest frame in the event window.
   - Falls back to the temporally closest tracked observation (`used_nearest_fallback=True`) if an event falls inside a short interpolated gap.
2. **Annotated Full-Frame Snapshot (`snapshots/<event_id>_<track_id>.jpg`)**:
   - Overlays Phase 5 polygons and lines (`scene_config.yaml`), highlights the target person's bounding box and bottom-center anchor point in an event-type-specific color, and draws a crisp HUD badge (`<event_id> | <event_type> | T<track_id> | <zone_id> | f=<frame_id> (<conf>)`).
3. **Padded Person Crop (`crops/<event_id>_<track_id>.jpg`)**:
   - Extracts a tight person crop with configurable context padding (`crop_padding_ratio = 0.10`, `min_padding_px = 8`) safely clamped to frame boundaries.
4. **Enriched 9-Column `events.csv` & Path Verification (`verify_evidence_records`)**:
   - Exports `event_id, event_type, track_id, zone_id, start_time, end_time, duration_seconds, snapshot_path, crop_path` and verifies that 100% of `snapshot_path` and `crop_path` entries resolve to non-empty image files on disk.

---

## Directory Structure

```text
cv_projects/07_visual_evidence/
├── README.md
├── notes.md
├── configs/
│   └── evidence.yaml
├── notebooks/
│   └── 01_explore_visual_evidence.ipynb
├── scripts/
│   └── extract_visual_evidence.py
└── outputs/
```

---

## Reproducible CLI Usage

```bash
uv run --locked python cv_projects/07_visual_evidence/scripts/extract_visual_evidence.py \
  --source data/fallback_clips/people-detection.mp4 \
  --events cv_projects/06_dwell_time_events/outputs/people_detection/events.csv \
  --tracks cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv \
  --config cv_projects/07_visual_evidence/configs/evidence.yaml \
  --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
  --output-dir cv_projects/07_visual_evidence/outputs/people_detection
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--source` | Path to input video file or MOT sequence directory | *Required* |
| `--events` | Path to Phase 6 `events.csv` | *Required* |
| `--tracks` | Path to Phase 3 `tracks.csv` | *Required* |
| `--config` | Path to `evidence.yaml` | `configs/evidence.yaml` |
| `--scene-config` | Optional path to Phase 5 `scene_config.yaml` | `../05_zone_line_reasoning/configs/scene_config.yaml` |
| `--output-dir` | Output directory for `snapshots/`, `crops/`, `events.csv`, and summary | `outputs/default_run` |
| `--strategy` | Override `best_frame_strategy` (`highest_confidence`, `midpoint`, `start_frame`, `largest_area`) | `None` |
| `--padding-ratio` | Override `crop_padding_ratio` | `None` |
| `--min-padding-px` | Override `min_padding_px` | `None` |
| `--jpeg-quality` | Override JPEG quality (`1-100`) | `None` |
