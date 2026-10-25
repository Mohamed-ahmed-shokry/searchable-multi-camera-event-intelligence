# 04 — Trajectory Extraction

Convert Phase 3 person tracks (`tracks.csv`) into ordered, timestamped 2D ground-plane trajectories (`trajectories.csv`), per-track path-length summaries (`track_summary.csv`), static full-scene trajectory maps (`trajectory_map.png`), and trailing motion overlay videos (`trajectory_overlay_video.mp4`).

---

## Layout

```text
cv_projects/04_trajectory_extraction/
  README.md
  notes.md
  configs/
    trajectory.yaml
  notebooks/
    01_explore_trajectories.ipynb
  scripts/
    extract_trajectories.py
    compare_smoothing.py
  outputs/                # gitignored generated artifacts
```

---

## Usage

### 1. Extract Trajectories & Render Overlay Video / Map

```bash
uv run --locked python cv_projects/04_trajectory_extraction/scripts/extract_trajectories.py \
  data/fallback_clips/people-detection.mp4 \
  --tracks-csv cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv \
  -o cv_projects/04_trajectory_extraction/outputs/people_detection
```

Run on a MOT17 frame sequence:

```bash
uv run --locked python cv_projects/04_trajectory_extraction/scripts/extract_trajectories.py \
  data/mot17/MOT17-11-FRCNN \
  --tracks-csv cv_projects/03_person_tracking/outputs/mot17_11_comparison/bytetrack/tracks.csv \
  --max-frames 300 \
  -o cv_projects/04_trajectory_extraction/outputs/mot17_11
```

### 2. Compare Anchor Strategies & Smoothing Windows

```bash
uv run --locked python cv_projects/04_trajectory_extraction/scripts/compare_smoothing.py \
  cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv \
  --fps 12.0 \
  --windows 1,3,5 \
  -o cv_projects/04_trajectory_extraction/outputs/people_detection_smoothing
```

---

## Output Schemas

### `trajectories.csv`

| Column | Type | Description |
| --- | --- | --- |
| `track_id` | `int` | Persistent person track ID from Phase 3 |
| `frame_id` | `int` | 0-based video frame index |
| `timestamp` | `float` | Video timestamp in seconds (`frame_id / fps`, 4 decimal places) |
| `x` | `float` | 2D anchor horizontal coordinate in pixels (2 decimal places) |
| `y` | `float` | 2D anchor vertical coordinate in pixels (2 decimal places) |

### `track_summary.csv`

| Column | Type | Description |
| --- | --- | --- |
| `track_id` | `int` | Persistent person track ID |
| `path_length_px` | `float` | Cumulative Euclidean path length in pixels across ordered points |
| `duration_seconds` | `float` | Track span duration in seconds (`(end_frame - start_frame + 1) / fps`) |
| `frame_count` | `int` | Number of frames in which this track was observed |

### Visual & JSON Artifacts

- `trajectory_map.png` — Dimmed reference frame overlaid with complete per-track polylines, start circles, end dots, and `ID <id>` labels.
- `trajectory_overlay_video.mp4` — Annotated video showing trailing polylines (`max_trail_length` points), current anchor points, and bounding boxes.
- `trajectory_extraction_summary.json` — Structured summary containing resolution, FPS, path length totals, and raw vs. smoothed coordinate jitter metrics.
