# Phase 5 Observations — Zone and Line Reasoning (`05_zone_line_reasoning`)

## 1. Visual Scene Geometry Layout (`people-detection.mp4`, `768×432 @ 12.0 fps`)

Using `scripts/define_scene_geometry.py` on frame `35` (`outputs/people_detection/reference_frame_grid.png` and `outputs/people_detection/scene_layout_preview.png`), we defined 3 floor-plane polygon zones and 2 directed tripwire lines in `configs/scene_config.yaml`:

### Polygon Zones (`ZoneSpec`)
| Zone ID | Polygon Vertices `(x, y)` | Floor-Plane Role | Area (`px²`) |
| --- | --- | --- | ---: |
| `entrance_area` | `(140, 350) → (520, 350) → (560, 431) → (110, 431)` | Lower foreground doorway threshold where shoppers enter the camera FOV | `33,615.0` |
| `center_concourse` | `(180, 250) → (560, 250) → (540, 350) → (160, 350)` | Central wooden-floor walking corridor immediately beyond the entrance threshold | `38,000.0` |
| `right_aisle` | `(460, 200) → (750, 200) → (750, 360) → (460, 360)` | Right-hand display aisle where shoppers turn and exit toward the right | `46,400.0` |

### Directed Tripwire Lines (`LineSpec`)
| Line ID | Endpoint `A (x, y)` | Endpoint `B (x, y)` | Length (`px`) | Orientation & Direction Semantics |
| --- | --- | --- | ---: | --- |
| `doorway_line` | `(100, 360)` | `(620, 360)` | `520.0` | Horizontal threshold tripwire across the lower entrance (`B->A` = walking upward from `y > 360` into the store; `A->B` = walking downward toward the exit) |
| `aisle_split_line` | `(370, 180)` | `(370, 420)` | `240.0` | Vertical concourse divider (`B->A` = moving left-to-right from `x < 370` to `x > 370`; `A->B` = moving right-to-left from `x > 370` to `x < 370`) |

---

## 2. Spatial Reasoning Results on Smoothed Trajectories (`bottom_center`, `w=3`)

Running `scripts/reason_zones_lines.py` on `cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv` (`7` tracks, `409` points across `596` frames) completed in **`3.82 s`** (`~156 fps` including MP4 rendering) and emitted **`38` spatial signals** in `zone_line_events.csv`:

### Signal Counts by Zone
| Zone ID | `zone_enter` | `zone_exit` | Net Active at Track End | Notes |
| --- | ---: | ---: | ---: | --- |
| `entrance_area` | `7` | `7` | `0` | All 7 tracked shoppers enter at the bottom threshold and exit northward into the store |
| `center_concourse` | `6` | `6` | `0` | 6 of 7 tracks traverse the central concourse (`Track 6` enters on the far right and steps directly from `entrance_area` into `right_aisle` at frame `372`) |
| `right_aisle` | `3` | `0` | `3` | Tracks `1`, `6`, and `7` enter the right-hand aisle and remain inside until their tracks terminate at the right edge of the FOV |
| **Total** | **`16`** | **`13`** | **`3`** | |

### Signal Counts by Line and Direction
| Line ID | `A->B` Crossings | `B->A` Crossings | Total Crossings | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `doorway_line` | `0` | `7` | `7` | Every tracked shopper (`Tracks 1, 2, 3, 5, 6, 7, 9`) crosses `doorway_line` in the `B->A` (inbound/upward) direction; zero outbound crossings occur in this clip |
| `aisle_split_line` | `1` | `1` | `2` | `Track 1` crosses left-to-right (`B->A`) at frame `37` (`t=3.0833s`) en route to `right_aisle`; `Track 3` briefly angles right-to-left (`A->B`) at frame `198` (`t=16.5000s`) |
| **Total** | **`1`** | **`8`** | **`9`** | |

---

## 3. Boundary Flicker Analysis: Raw (`w=1`) vs Smoothed (`w=3`, `w=5`) & Anchor Choice

We evaluated `analyze_boundary_flicker(signals, max_flicker_frames=5)` across both anchor strategies (`bottom_center` vs `centroid`) and rolling-mean smoothing windows (`w=1, 3, 5`):

| Anchor Strategy | Smoothing Window (`w`) | Total Signals | `zone_enter` | `zone_exit` | `line_cross` | Zone Flicker Cases (`<= 5` frames) | Line Jitter Cases (`<= 5` frames) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bottom_center` | `1` (raw) | `40` | `17` | `14` | `9` | **`1`** | `0` |
| **`bottom_center` (default)** | **`3`** | **`38`** | **`16`** | **`13`** | **`9`** | **`0`** | **`0`** |
| `bottom_center` | `5` | `38` | `16` | `13` | `9` | `0` | `0` |
| `centroid` | `1` (raw) | `36` | `14` | `14` | `8` | **`5`** | `0` |
| `centroid` | `3` | `36` | `14` | `14` | `8` | **`2`** | `0` |
| `centroid` | `5` | `36` | `14` | `14` | `8` | **`2`** | `0` |

### Key Findings on Boundary Flicker and False Crossings
1. **Single-frame bbox jitter on raw (`w=1`) `bottom_center` causes false zone exit/re-entry:**
   - Without trajectory smoothing (`w=1`), **`Track 3`** exhibits a 1-frame boundary oscillation on `entrance_area`: it triggers `zone_exit` at frame `194` (`y` dips just outside the left/top edge) and immediately triggers `zone_enter` at frame `195` (`gap_frames = 1`) before genuinely exiting at frame `205`.
   - Applying Phase 4's causal 3-frame rolling mean (`w=3`) **eliminates 100% of boundary flicker (`zone_flicker_count: 0`, `line_jitter_count: 0`)** while preserving exact transition frames (`frame 36`, `205`, `206`, `364`, `365`, `537`).
2. **Why `centroid` fails on floor-plane zones (`5` flicker cases on `w=1`):**
   - Because the bounding-box centroid `(0.5*(x1+x2), 0.5*(y1+y2))` sits `~75–95 px` higher in image space than the person's feet (`y2`), a shopper first detected near the bottom of the frame (`y2 ≈ 420`) already has their torso centroid near `y ≈ 350` (the exact top border of `entrance_area`).
   - Consequently, `centroid` trajectories barely clip `entrance_area` for `2–5` frames (`Tracks 1, 2, 3, 5`) and miss `Track 6`'s `doorway_line` crossing altogether (`8` crossings vs `9`).
3. **Implications for Phase 6 (Dwell-Time Events):**
   - Raw `zone_enter` and `zone_exit` signals in `zone_line_events.csv` are deliberately emitted without minimum-duration filtering so Phase 5 remains a pure geometric signal layer.
   - Phase 6 (`06_dwell_time_events`) will pair `zone_enter` and `zone_exit` per `(track_id, zone_id)` and apply configurable dwell-time thresholds (`min_dwell_seconds`) and short-gap tolerances.
