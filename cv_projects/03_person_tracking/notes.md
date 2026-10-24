# Phase 3 Notes — Person Tracking

Observations, side-by-side `ByteTrack` vs `BoT-SORT` evaluation, early quantitative MOT17 checkpoint (`checkpoint_metrics.json`), and concrete frame-level ID-switch and fragmentation analyses.

## 1. Side-by-Side Tracker Comparison & Quantitative MOT17 Checkpoint

Evaluated on `data/mot17/MOT17-11-FRCNN` (1920×1080 @ 30 fps, first 300 frames, `yolo11m.pt`, `conf >= 0.4`, `iou_threshold = 0.5` against official `MOT17-11-FRCNN/gt/gt.txt` pedestrian ground truth, 3,559 active pedestrian GT boxes):

| Metric | ByteTrack (`bytetrack.yaml`) | BoT-SORT (`botsort.yaml`) | Notes |
|--------|------------------------------|---------------------------|-------|
| **MOTA** | **0.4484 (44.84%)** | 0.4470 (44.70%) | ByteTrack slightly edges BoT-SORT due to fewer false positives (417 vs 429) |
| **MOTP** | 0.8587 (85.87%) | **0.8656 (86.56%)** | High bounding-box localization precision for both trackers |
| **IDF1** | 0.6026 (60.26%) | **0.6076 (60.76%)** | BoT-SORT's camera-motion compensation slightly improves global identity F1 (+0.50%) |
| **ID Switches (IDSW)** | 5 | **3** | Both keep ID switches very low over 3,559 GT boxes |
| **Precision / Recall** | **82.87%** / 56.70% | 82.50% / **56.84%** |Recall reflects single-threshold (`0.4`) COCO detector vs. heavily occluded MOT17 GT |
| **Throughput (1080p + Video Render)** | **8.42 FPS** (35.6s) | 5.84 FPS (51.3s) | ByteTrack is **44.2% faster** than BoT-SORT on 1920×1080 frames |
| **Unique Track IDs** | 43 | 36 | Ground truth has 75 total sequence identities |
| **Fragmentation Gaps** | **35** | 45 | ByteTrack exhibits fewer mid-track gaps |

### Static Surveillance Validation (`data/fallback_clips/people-detection.mp4`, 768×432 @ 12 fps, 596 frames)

- **Tracker**: `bytetrack.yaml` (`yolo11m.pt`, `conf >= 0.4`, `cuda:0`)
- **Throughput**: **17.06 FPS** (34.94s total including `annotated_tracking_video.mp4` encoding)
- **Unique Tracks**: **7** (`track_id` 1, 2, 3, 5, 6, 7, 9; 409 total bounding boxes)
- **Continuity**: **0 fragmentations** and **0 spatial handoffs** — every single tracked person has `duration_frames == observed_frames` (100% continuous track persistence across all 596 frames).

## 2. Concrete ID-Switch & Fragmentation Log (`MOT17-11-FRCNN`)

Below are 5 concrete, frame-indexed ID-switch and fragmentation cases observed and logged in `outputs/mot17_11_comparison/checkpoint_metrics.json`:

1. **Frame 23 — Occlusion Re-entry Switch on GT ID `28` (ByteTrack)**:
   - `track_id=12` disappears after frame `18` and reappears at frame `23` (`gap_frames=4`) when a foreground pedestrian crosses in front of GT pedestrian `28`. At frame `23`, GT ID `28` switches from `old_pred_id=7` to `new_pred_id=12` (`IoU=0.8053`).
2. **Frame 55 — Multi-Person Walkway Crossing on GT ID `25` (ByteTrack & BoT-SORT)**:
   - During a dense pedestrian crossing near the center of the walkway combined with forward camera translation, GT pedestrian `25` switches from `old_pred_id=6` to `new_pred_id=22` (`IoU=0.5681`) in ByteTrack and from `old_pred_id=6` to `new_pred_id=19` (`IoU=0.5736`) in BoT-SORT.
3. **Frames 146–155 — Prolonged Occlusion Handoff on GT ID `6` (ByteTrack & BoT-SORT)**:
   - GT pedestrian `6` (initially matched to `pred_id=8`) is occluded by oncoming pedestrians. BoT-SORT reassigns `new_pred_id=39` at frame `146` (`IoU=0.5005`), whereas ByteTrack bridges part of the gap (`lost_after_frame=149`, `reappeared_at_frame=155`, `gap_frames=5`) before switching to `new_pred_id=51` at frame `155` (`IoU=0.7811`).
4. **Frames 193–198 — Spatial Handoff & 19-Frame Gap on GT ID `55` (ByteTrack)**:
   - `old_track_id=22` ends at frame `193` and `new_track_id=59` initializes 38.77 px away at frame `198`, triggering an ID switch on GT ID `55` (`old_pred_id=53 -> new_pred_id=59`, `IoU=0.6000`), followed by a 19-frame re-identification gap (`lost_after_frame=198`, `reappeared_at_frame=218`).
5. **Frames 245–246 — Secondary Crossing Switch on GT ID `25` (ByteTrack & BoT-SORT)**:
   - A second occlusion on GT ID `25` triggers a switch from `old_pred_id=22` to `new_pred_id=66` at frame `246` (`IoU=0.5031`) in ByteTrack and from `old_pred_id=19` to `new_pred_id=51` at frame `245` (`IoU=0.5121`) in BoT-SORT.

## 3. Tracker Decision (`configs/tracking.yaml`)

- **Chosen Default Tracker**: **ByteTrack (`bytetrack.yaml`)** with `yolo11m.pt` (`confidence_threshold: 0.4`, `iou_threshold: 0.45`).
- **Justification**:
  1. On static surveillance cameras (our primary target domain for zone, line, and dwell-time reasoning in Phases 5–6), ByteTrack achieves **0 fragmentations** and **17.06 FPS**.
  2. On moving-camera 1080p MOT17 footage (`MOT17-11-FRCNN`), ByteTrack achieves **44.2% higher throughput** than BoT-SORT (`8.42 FPS` vs `5.84 FPS`), slightly higher `MOTA` (`0.4484` vs `0.4470`), and fewer track fragmentations (`35` vs `45`), while trailing by only `0.50%` in `IDF1` (`0.6026` vs `0.6076`).
  3. `botsort.yaml` remains supported via `--tracker botsort.yaml` or `comparison_tracker` in `configs/tracking.yaml` for moving-camera sequences.
