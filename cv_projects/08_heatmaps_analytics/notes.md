# Phase 8 — Heatmaps and Analytics: Engineering & Empirical Notes

## 1. Spatial Heatmap Design & Risk Mitigations

### 1.1 Mitigating Stationary-Person Dominance (`max_points_per_track_cell` + `log` Normalization)
A naive 2D histogram over trajectory points suffers when a single person stands still for several seconds: one grid cell accumulates dozens of points, pushing every walking corridor into the bottom `5%` of the colormap.
We mitigate this in two complementary ways in `accumulate_trajectory_grid` and `smooth_and_normalize_heatmap`:
1. **Per-Track-Per-Cell Cap (`max_points_per_track_cell = 6`)**: Each unique `(track_id, row_idx, col_idx)` tuple can contribute at most `6` observations (`0.5s` @ `12 fps`) to a single `16×16 px` grid cell. A cell only reaches high counts when **multiple distinct tracks** traverse that cell or when a track moves across adjacent cells.
2. **Logarithmic Normalization (`normalization_mode = "log"`)**: After Gaussian smoothing (`sigma = 24.0 px`), density values are normalized via `log1p(density) / log1p(max_density)`, preserving contrast across both high-traffic entryways and moderate-traffic aisles.

### 1.2 Mitigating Grid-vs-Frame Resolution Distortion
To prevent aspect-ratio or margin distortion between the 2D accumulator grid (`48×27` bins at `grid_bin_size_px = 16` for `768×432`) and the video frame:
- `accumulate_trajectory_grid` immediately upsamples the `(27, 48)` grid to exact `(frame_height, frame_width) = (432, 768)` using nearest-neighbor interpolation (`cv2.INTER_NEAREST`) before applying `cv2.GaussianBlur` in pixel space (`gaussian_sigma_px = 24.0`).
- `render_heatmap_overlay` blends the colormap directly onto the `(432, 768, 3)` reference frame (`frame 35`) using a soft alpha mask `alpha * (norm ** 0.65)` where `norm > 0.015`, keeping zero-activity walls/ceilings un-tinted while highlighting active floor regions.

---

## 2. Empirical Results & Manual Cross-Check on `people-detection.mp4`

Command executed:
```bash
uv run --locked python cv_projects/08_heatmaps_analytics/scripts/generate_heatmaps_analytics.py \
  --source data/fallback_clips/people-detection.mp4 \
  --trajectories cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv \
  --events cv_projects/07_visual_evidence/outputs/people_detection/events.csv \
  --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
  --config cv_projects/08_heatmaps_analytics/configs/analytics.yaml \
  --output-dir cv_projects/08_heatmaps_analytics/outputs/people_detection
```

### 2.1 Visual Hot-Spot Verification (`heatmap.png`)
Inspecting `cv_projects/08_heatmaps_analytics/outputs/people_detection/heatmap.png` confirms three primary hot spots that correspond directly to visible foot traffic in `people-detection.mp4`:
1. **Bottom Entrance Corridor (`entrance_area`, `x ≈ 240..540, y ≈ 360..432`)**: All `7` tracked people enter the scene from the bottom edge of the camera view and walk upward across `entrance_area` and `doorway_line`.
2. **Left Doorway Exit (`x ≈ 40, y ≈ 248`, peak cell `row=15, col=2`, `capped_count=11.0`)**: Tracks `2, 3, 4, 5` converge at the left corridor exit after crossing `center_concourse`.
3. **Center Concourse to Right Aisle (`x ≈ 450..680, y ≈ 250..360`)**: Tracks `1, 6, 7` veer right across `aisle_split_line` and walk through `right_aisle`.

### 2.2 Per-Zone Counts Cross-Check (`zone_counts.csv` vs `events.csv`)

| `zone_id` | `entry_count` | `exit_count` | `total_dwell_seconds` | `peak_occupancy` | Manual Cross-Check Against `events.csv` (48 rows) |
|---|---|---|---|---|---|
| `entrance_area` | `7` | `7` | `9.0001` | `3` | Exactly `7 zone_entry`, `7 zone_exit`, and `5 dwell_time_event` rows (`1.3334 + 1.9167 + 1.8333 + 1.9167 + 2.0000 = 9.0001s`); frames `185..198` have Tracks `2, 3, 4` simultaneously inside `entrance_area` (`peak_occupancy = 3`). |
| `center_concourse` | `6` | `6` | `3.2500` | `2` | Exactly `6 zone_entry`, `6 zone_exit`, and `2 dwell_time_event` rows (`1.5833 + 1.6667 = 3.2500s`); simultaneous occupancy reaches `2`. |
| `right_aisle` | `3` | `0` | `8.0000` | `2` | Exactly `3 zone_entry`, `0 zone_exit` (Tracks `1, 6, 7` terminate at the right frame edge inside `right_aisle` with `track_lost_in_zone=True`), and `3 dwell_time_event` rows (`2.3333 + 2.4167 + 3.2500 = 8.0000s`). |
| **Total** | **`16`** | **`13`** | **`20.2501`** | — | Matches `16 zone_entry`, `13 zone_exit`, and `10 dwell_time_event` in `events.csv` `100%`. |

### 2.3 Per-Line Counts Cross-Check (`line_counts.csv` vs `events.csv`)

| `line_id` | `crossings_ab` | `crossings_ba` | `total` | Manual Cross-Check Against `events.csv` |
|---|---|---|---|---|
| `doorway_line` | `0` | `7` | `7` | All `7` tracks cross `doorway_line` moving upward (`B->A`: `evt_0003, evt_0017, evt_0019, evt_0025, evt_0032, evt_0040, evt_0044`). |
| `aisle_split_line` | `1` | `1` | `2` | `evt_0015` (`Track 3`, `A->B`) and `evt_0007` (`Track 1`, `B->A`). |
| **Total** | **`1`** | **`8`** | **`9`** | Matches all `9 line_crossing` rows in `events.csv` `100%`. |

### 2.4 Track Duration Distribution (`analytics_summary.json`)
Across all `7` tracks (`409` trajectory points):
- `min_duration_seconds`: `3.0000s`
- `max_duration_seconds`: `5.7500s`
- `mean_duration_seconds`: `4.7857s`
- `median_duration_seconds`: `5.0000s`
- `p90_duration_seconds`: `5.4000s`
- `mean_path_length_px`: `461.46 px` (`max_path_length_px = 553.84 px`)
- `duration_buckets`: `{"0.0-2.0s": 0, "2.0-4.0s": 1, "4.0-6.0s": 6, "6.0-10.0s": 0, ">=10.0s": 0}`
