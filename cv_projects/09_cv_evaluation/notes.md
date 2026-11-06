# Phase 9 Observations — CV Evaluation (`09_cv_evaluation`)

## 1. Quantitative Evaluation (`MOT17-11-FRCNN` Pedestrian Ground Truth)

Evaluated on `data/mot17/MOT17-11-FRCNN` (`1920×1080 @ 30 fps`, first `300` frames, `3,559` active
pedestrian ground-truth boxes across `33` unique GT identities in frames `0..299`, `iou_threshold = 0.50`,
`yolo11m.pt` @ `conf >= 0.40`):

### 1.1 Detection Precision & Recall (`evaluate_detection_precision_recall`)

| Metric | Value | Notes |
| --- | ---: | --- |
| **Precision** | **`0.8287` (`82.87%`)** | `2,018` true positives out of `2,435` predicted boxes (`417` FP) |
| **Recall** | **`0.5670` (`56.70%`)** | `2,018` true positives out of `3,559` active GT boxes (`1,541` FN) |
| **F1 Score** | **`0.6733` (`67.33%`)** | Harmonic mean of precision and recall |
| **Mean Matched IoU** | **`0.8587` (`85.87%`)** | Tight bounding-box localization on matched detections |

### 1.2 Multi-Object Tracking (`ByteTrack` vs `BoT-SORT`, `evaluate_tracking_metrics`)

| Tracker | MOTA | MOTP | IDF1 | Precision | Recall | ID Switches | Fragmentations | Mostly Tracked (`>=80%`) | Partially Tracked (`20-80%`) | Mostly Lost (`<20%`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`bytetrack`** | **`0.4484`** | `0.8587` | `0.6026` | **`0.8287`** | `0.5670` | `5` | **`35`** | `9/33` (`27.3%`) | `10/33` (`30.3%`) | `14/33` (`42.4%`) |
| **`botsort`** | `0.4470` | **`0.8656`** | **`0.6076`** | `0.8250` | **`0.5684`** | **`3`** | `45` | `9/33` (`27.3%`) | `10/33` (`30.3%`) | `14/33` (`42.4%`) |

**Honest Interpretation:**
- **Why Precision (`82.87%`) & MOTP (`85.87%`) are high while Recall (`56.70%`) & MOTA (`44.84%`) are moderate:**
  Official `MOT17-11-FRCNN` ground truth includes all pedestrians down to `0.0` visibility (heavily occluded
  behind foreground pedestrians or glass Mall railings) and tiny far-depth figures (`< 25 px` tall). Using
  an off-the-shelf COCO `yolo11m.pt` detector with a conservative confidence threshold (`0.40`) deliberately
  prioritizes precision over recall so downstream spatial zones and tripwires do not trigger on false-positive
  ghost boxes. As a result, `14/33` (`42.4%`) GT identities—almost all small background or heavily occluded
  shoppers—are mostly lost (`< 20%` lifespan tracked), whereas foreground/midground shoppers are tracked with
  only `5` ID switches across `3,559` GT boxes.

---

## 2. Qualitative Surveillance Evaluation (`people-detection.mp4`, `596` frames @ `12.0 fps`)

Cross-checked `cv_projects/07_visual_evidence/outputs/people_detection/events.csv` (`48` events),
`cv_projects/08_heatmaps_analytics/outputs/people_detection/zone_counts.csv`, `line_counts.csv`, and
`cv_projects/06_dwell_time_events/outputs/people_detection/event_debug_log/stay_traces.json` (`16` stays)
against manual frame-by-frame playback verification:

| Audit Category | Manual Playback Ground Truth | Automated Pipeline Output | Accuracy / Error |
| --- | --- | --- | --- |
| **Zone Entries (`16` total)** | `entrance_area=7`, `center_concourse=6`, `right_aisle=3` | `entrance_area=7`, `center_concourse=6`, `right_aisle=3` | **`100.0%` (`3/3` zones exact)** |
| **Zone Exits (`13` total)** | `entrance_area=7`, `center_concourse=6`, `right_aisle=0` | `entrance_area=7`, `center_concourse=6`, `right_aisle=0` | **`100.0%` (`3/3` zones exact)** |
| **Line Crossings (`9` total)** | `doorway_line:B->A=7`, `doorway_line:A->B=0`, `aisle_split_line:A->B=1`, `aisle_split_line:B->A=1` | `doorway_line:B->A=7`, `doorway_line:A->B=0`, `aisle_split_line:A->B=1`, `aisle_split_line:B->A=1` | **`100.0%` (`4/4` directions exact)** |
| **Dwell-Time Events (`10` total)** | `10` triggered stays (`6` in `entrance_area`, `1` in `center_concourse`, `3` in `right_aisle`) | `10` `dwell_time_event` rows (`20.2501s` total dwell) | **`100.0%` count match, `0.0000s` max duration error** |
| **Track-Loss-in-Zone Resolutions** | `3` shoppers (`Tracks 1, 6, 7`) vanish behind right wall in `right_aisle` | `3` stays resolved with `track_lost_in_zone=true` and `0` fabricated `zone_exit` rows | **`100.0%` match** |

---

## 3. Per-Stage Processing FPS & Latency Benchmark (`596` frames, RTX 3060 Laptop GPU + CPU)

| Stage ID | Stage Name | Device | Frames | Elapsed (s) | Throughput (FPS) | Latency (ms/frame) | Share of Pipeline (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `01_video_inspection` | Video Inspection & Sampling | `cpu (OpenCV)` | `596` | `0.4200` | **`1419.05`** | `0.70` | `0.52%` |
| `02_person_detection` | Person Detection (`YOLO11m`) | `cuda:0` | `596` | `32.4460` | **`18.37`** | `54.44` | `39.97%` |
| `03_person_tracking` | Person Tracking (`ByteTrack + YOLO11m`) | `cuda:0` | `596` | `34.9400` | **`17.06`** | `58.62` | `43.05%` |
| `04_trajectory_extraction` | Trajectory Extraction & Smoothing | `cpu (NumPy/OpenCV)` | `596` | `2.4380` | **`244.46`** | `4.09` | `3.00%` |
| `05_zone_line_reasoning` | Zone & Line Spatial Reasoning | `cpu (OpenCV)` | `596` | `4.1850` | **`142.41`** | `7.02` | `5.16%` |
| `06_dwell_time_events` | Dwell-Time Event State Machine | `cpu (OpenCV)` | `596` | `5.3110` | **`112.22`** | `8.91` | `6.54%` |
| `07_visual_evidence` | Visual Evidence Snapshot & Crop Extraction | `cpu (OpenCV)` | `596` | `1.1089` | **`537.47`** | `1.86` | `1.37%` |
| `08_heatmaps_analytics` | Heatmaps & Activity Analytics | `cpu (NumPy/OpenCV)` | `596` | `0.3176` | **`1876.57`** | `0.53` | `0.39%` |

- **Bottleneck Analysis:** GPU neural inference (`02_person_detection` and `03_person_tracking`, which both
  run `YOLO11m` forward passes plus full MP4 video encoding in standalone experiment mode) accounts for
  **`83.02%`** of cumulative standalone stage runtime. In an integrated pipeline (Phase 11), `03_person_tracking`
  subsumes `02_person_detection` in a single YOLO11m pass, reducing end-to-end processing time for all 8 stages
  from `81.17s` to **`48.72s` (`12.23 FPS` end-to-end, faster than real-time `12.0 FPS` source playback)**.

---

## 4. Concrete Failure Examples & Honest Limitations (`outputs/full_evaluation/failure_examples/`)

1. **`failure_01_mot17_crowded_id_switch.jpg` (`MOT17-11-FRCNN` Frame `23`):**
   - **Symptom:** Ground-truth pedestrian `#28` is tracked as predicted ID `7` up to frame `18`, is fully
     occluded for `4` frames (`19..22`) by a closer pedestrian walking in the opposite direction, and is
     re-bound to predicted ID `12` at frame `23` (`IoU = 0.8053`).
   - **Root Cause:** Linear Kalman box prediction drifts under simultaneous camera forward translation and
     full occlusion; without deep appearance ReID features, ByteTrack assigns a new track identity.
2. **`failure_02_mot17_small_occluded_misses.jpg` (`MOT17-11-FRCNN` Frame `155`):**
   - **Symptom:** Multiple background pedestrians and partially occluded walkers (e.g., `GT #13`, `GT #77`,
     `GT #80` behind the glass railing) are missed (`FN = 1,541` boxes over `300` frames).
   - **Root Cause:** Off-the-shelf COCO `yolo11m.pt` at `conf >= 0.40` misses pedestrians whose visible height
     is `< 25 px` or who are `> 70%` occluded by crowd members or glass reflections.
3. **`failure_03_boundary_truncated_crop.jpg` (`people-detection.mp4` Frame `19`):**
   - **Symptom:** Shopper `Track 1` enters from the bottom edge of the camera frame (`x1=194.01, y1=319.26,
     x2=306.73, y2=430.96` on a `768×432` frame), truncating the lower body at the bottom border and causing
     `7/48` instantaneous entry/crossing evidence crops in Phase 7 to be clamped (`clipped_at_frame_edge = true`).
   - **Root Cause:** The physical camera mount cuts across the bottom doorway floor threshold at `y = 432`.
4. **`failure_04_track_loss_in_right_aisle.jpg` (`people-detection.mp4` Frame `75`):**
   - **Symptom:** All `3` shoppers who enter `right_aisle` (`Track 1` at frame `75`, `Track 6` at frame `401`,
     `Track 7` at frame `410`) disappear inside the zone without an exit (`3` entries, `0` exits).
   - **Root Cause:** An opaque right-hand architectural wall occludes the exit corridor outside the camera's
     field of view. Phase 6 handles this honestly by resolving those `3` stays with `track_lost_in_zone = true`
     at `last_inside_time` rather than fabricating synthetic `zone_exit` events.
