# Subproject 08 — Heatmaps and Analytics (`cv_projects/08_heatmaps_analytics/`)

## Overview

Phase 8 summarizes spatial movement and behavioral event activity into visual and tabular artifacts:
1. **2D Trajectory Grid & Stationary-Person Mitigation (`accumulate_trajectory_grid`)**:
   - Bins bottom-center trajectory coordinates into a `grid_bin_size_px` (`16 px`) 2D grid with a configurable per-track-per-cell cap (`max_points_per_track_cell = 6`) so a stationary person cannot wash out walking corridors.
   - Upsamples the grid to exact `(frame_height, frame_width)` resolution before smoothing so the overlay never distorts scene geometry.
2. **Gaussian Smoothing & Normalization (`smooth_and_normalize_heatmap`)**:
   - Applies `cv2.GaussianBlur` (`gaussian_sigma_px = 24.0`) and normalizes into `[0.0, 1.0]` using `log` (default), `percentile`, or `linear` scaling.
3. **Reference-Frame Heatmap Overlay (`render_heatmap_overlay` -> `heatmap.png`)**:
   - Blends an OpenCV colormap (`jet`, `inferno`, `turbo`, or `hot`) onto the scene reference frame with Phase 5 polygon/line overlays and a Low–High colorbar legend.
4. **Per-Zone & Per-Line Activity Tables (`zone_counts.csv`, `line_counts.csv`)**:
   - `zone_counts.csv`: `zone_id, entry_count, exit_count, total_dwell_seconds, peak_occupancy`
   - `line_counts.csv`: `line_id, crossings_ab, crossings_ba, total`
5. **Track Duration Distribution & Summary (`analytics_summary.json`)**:
   - Computes `min`, `max`, `mean`, `median`, `p90` durations, path lengths, and duration histogram buckets.

---

## Directory Structure

```text
cv_projects/08_heatmaps_analytics/
├── README.md
├── notes.md
├── configs/
│   └── analytics.yaml
├── notebooks/
│   └── 01_explore_heatmaps_analytics.ipynb
├── scripts/
│   └── generate_heatmaps_analytics.py
└── outputs/
```

---

## Reproducible CLI Usage

```bash
uv run --locked python cv_projects/08_heatmaps_analytics/scripts/generate_heatmaps_analytics.py \
  --source data/fallback_clips/people-detection.mp4 \
  --trajectories cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv \
  --events cv_projects/07_visual_evidence/outputs/people_detection/events.csv \
  --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
  --config cv_projects/08_heatmaps_analytics/configs/analytics.yaml \
  --output-dir cv_projects/08_heatmaps_analytics/outputs/people_detection
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--source` | Path to input video file or MOT sequence directory | *Required* |
| `--trajectories` | Path to Phase 4 `trajectories.csv` | *Required* |
| `--events` | Path to Phase 6 or Phase 7 `events.csv` | *Required* |
| `--config` | Path to `analytics.yaml` | `configs/analytics.yaml` |
| `--scene-config` | Optional path to Phase 5 `scene_config.yaml` | `../05_zone_line_reasoning/configs/scene_config.yaml` |
| `--output-dir` | Output directory for `heatmap.png`, `zone_counts.csv`, `line_counts.csv`, and `analytics_summary.json` | `outputs/default_run` |
| `--grid-bin-size` | Override `grid_bin_size_px` | `None` |
| `--sigma` | Override `gaussian_sigma_px` | `None` |
| `--normalization` | Override `normalization_mode` (`log`, `percentile`, `linear`) | `None` |
| `--max-points-per-track-cell` | Override `max_points_per_track_cell` | `None` |
| `--colormap` | Override `colormap` (`jet`, `inferno`, `turbo`, `hot`) | `None` |
