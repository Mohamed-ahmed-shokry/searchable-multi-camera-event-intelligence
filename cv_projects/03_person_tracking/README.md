# 03 Person Tracking

Keep consistent `track_id` values across frames using ByteTrack (`bytetrack.yaml`) and BoT-SORT (`botsort.yaml`) powered by Ultralytics YOLO11 detections, export schema-compliant `tracks.csv` and `track_summary.csv` files, render annotated tracking videos (`annotated_tracking_video.mp4`), detect fragmentation/handoff anomalies, and evaluate CLEAR-MOT (`MOTA`, `MOTP`, `IDSW`) and `IDF1` against MOT17 ground truth (`checkpoint_metrics.json`).

## Usage

### 1. Run Person Tracking (`track_people.py`)

```bash
# Install CUDA detection/tracking dependencies
uv sync --locked --extra detection-cuda

# Track people in a video clip or MOT17 sequence directory
uv run python cv_projects/03_person_tracking/scripts/track_people.py \
    data/mot17/MOT17-11-FRCNN \
    --config cv_projects/03_person_tracking/configs/tracking.yaml \
    --output-dir cv_projects/03_person_tracking/outputs/mot17_11_bytetrack
```

### 2. Side-by-Side Tracker Comparison & Quantitative MOT17 Checkpoint (`compare_trackers.py`)

```bash
uv run python cv_projects/03_person_tracking/scripts/compare_trackers.py \
    data/mot17/MOT17-11-FRCNN \
    --output-dir cv_projects/03_person_tracking/outputs/mot17_11_comparison \
    --save-videos
```

## Output Schemas

- `outputs/<run>/tracks.csv`:
  - Columns: `frame_id, track_id, x1, y1, x2, y2, confidence`
  - Every input frame `0 .. N-1` is accounted for (frames with zero active tracks emit `frame_id,,,,,,`).
- `outputs/<run>/track_summary.csv`:
  - Columns: `track_id, start_frame, end_frame, duration_frames, duration_seconds, observed_frames`
- `outputs/<run>/annotated_tracking_video.mp4`:
  - Annotated video with deterministic per-track colors and `ID <track_id> (<conf>)` labels.
- `outputs/<run>/checkpoint_metrics.json`:
  - Quantitative comparison across `bytetrack` and `botsort` (`mota`, `motp`, `idf1`, `id_switches`, `precision`, `recall`, `fps`, `num_unique_tracks`, `fragmentation_count`, and detailed `id_switch_events` / `anomalies`).
