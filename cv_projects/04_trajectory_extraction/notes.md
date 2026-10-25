# Phase 4 — Trajectory Extraction Notes

## 1. Anchor-Point Choice (`bottom_center` vs `centroid`)

Phase 4 converts per-frame bounding boxes `(x1, y1, x2, y2)` from Phase 3 (`tracks.csv`) into ordered 2D trajectory coordinates `(x, y)` with video-derived timestamps `timestamp = frame_id / fps`:

- **`bottom_center` (default)**: `(x, y) = (0.5 * (x1 + x2), y2)`.
  - Represents the pedestrian's feet on the ground plane.
  - Because Phase 5 (Zone & Line Reasoning) and Phase 6 (Dwell-Time Events) evaluate polygon containment and line crossings drawn on the floor plane, `bottom_center` avoids perspective parallax errors where a tall person's torso centroid projects into an adjacent floor zone while their feet are still outside.
  - When a person enters/exits near the bottom border of a retail camera (`people-detection.mp4`, `y2 ≈ 431` on a `768×432` frame), `y2` is briefly clipped by the bottom image edge while their torso rises into the frame (`frames 19–26`), whereas in full-body views (`frames 27–75` and `MOT17-11-FRCNN`) `bottom_center` tracks the exact foot contact line.
- **`centroid` (supported via `--anchor centroid`)**: `(x, y) = (0.5 * (x1 + x2), 0.5 * (y1 + y2))`.
  - Useful when feet are persistently occluded by low barriers or shelves, at the cost of a vertical offset proportional to half the bounding-box height (`0.5 * (y2 - y1)`).

---

## 2. Raw vs. Rolling-Mean Smoothing Comparison (`compare_smoothing.py`)

We evaluated causal rolling-mean smoothing across windows `w ∈ {1, 3, 5}` (where `w = 1` is unsmoothed raw coordinates) and both anchors on two real footage sequences:

### 2.1 `data/fallback_clips/people-detection.mp4` (596 frames, 768×432 @ 12.0 fps, 7 tracks, 409 points)

| Anchor | Window `w` | Total Path Length (px) | Mean Step (px) | Max Step (px) | Mean Jitter (px) | Jitter Reduction (%) | Teleport Jumps (`>150 px`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bottom_center` | `1` (raw) | `3327.40` | `8.277` | `54.285` | `4.084` | `0.00%` | `0` |
| `bottom_center` | `3` (default) | `3230.20` | `8.035` | `28.133` | `2.455` | **`39.89%`** | `0` |
| `bottom_center` | `5` | `3155.28` | `7.849` | `18.476` | `1.493` | **`63.44%`** | `0` |
| `centroid` | `1` (raw) | `3439.11` | `8.555` | `54.426` | `3.567` | `0.00%` | `0` |
| `centroid` | `3` | `3342.83` | `8.316` | `28.133` | `2.194` | **`38.49%`** | `0` |
| `centroid` | `5` | `3269.58` | `8.133` | `18.392` | `1.323` | **`62.91%`** | `0` |

### 2.2 `data/mot17/MOT17-11-FRCNN` (300 frames, 1920×1080 @ 30.0 fps, 43 tracks, 2,435 points)

| Anchor | Window `w` | Total Path Length (px) | Mean Step (px) | Max Step (px) | Mean Jitter (px) | Jitter Reduction (%) | Teleport Jumps (`>150 px`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bottom_center` | `1` (raw) | `16669.72` | `6.969` | `233.484` | `3.294` | `0.00%` | `1` |
| `bottom_center` | `3` (default) | `15369.41` | `6.425` | `116.742` | `1.750` | **`46.87%`** | `0` |
| `bottom_center` | `5` | `14567.65` | `6.090` | `116.742` | `1.279` | **`61.17%`** | `0` |
| `centroid` | `1` (raw) | `14930.90` | `6.242` | `139.472` | `2.663` | `0.00%` | `0` |
| `centroid` | `3` | `13930.32` | `5.824` | `69.736` | `1.463` | **`45.06%`** | `0` |
| `centroid` | `5` | `13207.21` | `5.521` | `69.736` | `1.114` | **`58.17%`** | `0` |

**Why `smoothing_window = 3` is the locked default:**
- A 3-frame causal rolling mean cuts second-order bounding-box jitter (`||(p_{t+1}-p_t)-(p_t-p_{t-1})||`) by **39.89%** on `people-detection.mp4` and **46.87%** on `MOT17-11-FRCNN` while introducing only a 1-frame (`33–83 ms`) lag, preserving sharp turns for Phase 5 line crossings.
- Larger windows (`w = 5`) reduce jitter further (`~61–63%`) but round sharp corners and shorten total path length by `5.2%` (`3327.40 px → 3155.28 px`).

---

## 3. Per-Track Summary (`people-detection.mp4`, `bottom_center`, `w = 3`)

| `track_id` | `path_length_px` | `duration_seconds` | `frame_count` |
| ---: | ---: | ---: | ---: |
| `1` | `553.84` | `4.750` | `57` |
| `2` | `356.26` | `5.250` | `63` |
| `3` | `538.99` | `5.833` | `70` |
| `5` | `454.76` | `5.250` | `63` |
| `6` | `419.16` | `5.083` | `61` |
| `7` | `436.56` | `4.833` | `58` |
| `9` | `470.64` | `3.083` | `37` |

---

## 4. Manual Cross-Check of Extracted Trajectories Against Video & `tracks.csv`

1. **Exact coordinate & smoothing verification (`people-detection.mp4`, `track_id = 1`, frames `19–21`)**:
   - Frame `19` (`t = 19 / 12 = 1.5833 s`): box `(194.01, 319.26, 306.73, 430.96)` → raw `bottom_center = (250.37, 430.96)` → `w=3` output `(250.37, 430.96)`.
   - Frame `20` (`t = 20 / 12 = 1.6667 s`): box `(192.43, 290.36, 334.15, 430.99)` → raw `bottom_center = (263.29, 430.99)` → `w=3` mean over frames `19..20` = `((250.37 + 263.29)/2, (430.96 + 430.99)/2) = (256.83, 430.98)`.
   - Frame `21` (`t = 21 / 12 = 1.7500 s`): box `(190.73, 264.90, 356.84, 430.64)` → raw `bottom_center = (273.785, 430.64)` → `w=3` mean over frames `19..21` = `(262.48, 430.86)`.
   - Over its full 57-frame lifespan (`frames 19–75`, `duration = 4.750 s`), `track_id = 1` walks diagonally from the bottom-left entrance `(250.37, 430.96)` toward the upper-right aisle `(552.17, 178.60)` covering `553.84 px` with zero fragmentation gaps.
2. **Anomaly inspection (`MOT17-11-FRCNN`, `track_id = 67`)**:
   - `analyze_trajectory_quality` flagged a single raw step exceeding `150 px`: `track_id = 67` jumped `233.48 px` between `frame 257` and `frame 272`.
   - Cross-checking the video confirms `track_id = 67` was occluded for 14 frames (`258–271`) behind a foreground pedestrian under camera panning and re-linked by ByteTrack's 30-frame track buffer at `frame 272`.
