# Phase 7 — Visual Evidence: Engineering & Empirical Notes

## 1. Best-Frame Selection Rule & Strategy Comparison

For every structured event (`EventRecord`) emitted in Phase 6 over a time window `[start_time, end_time]`:
1. **Frame Window Conversion**:
   - `min_frame = max(0, round(start_time * fps))`
   - `max_frame = max(min_frame, round(end_time * fps))`
2. **Candidate Filtering**:
   - Filter all tracked detections (`TrackDetection`) of `event.track_id` from Phase 3 `tracks.csv` where `min_frame <= frame_id <= max_frame`.
3. **Candidate Ranking (`best_frame_strategy`)**:
   - **`highest_confidence` (default)**: Selects the frame in `[min_frame, max_frame]` with the highest detector confidence `confidence`, tie-breaking by largest bounding-box area `(x2 - x1) * (y2 - y1)`, then earliest `frame_id`.
   - **`largest_area`**: Selects the frame with the largest bounding-box area `(x2 - x1) * (y2 - y1)`, tie-breaking by highest confidence, then earliest `frame_id`.
   - **`midpoint`**: Selects the frame closest to `(min_frame + max_frame) / 2.0`, tie-breaking by highest confidence, then earliest `frame_id`.
   - **`start_frame`**: Selects the earliest `frame_id` in `[min_frame, max_frame]`.
4. **Nearest-Frame Fallback**:
   - If `[min_frame, max_frame]` contains no raw tracked bounding box for `event.track_id` (for instance, if a line crossing occurred on a short interpolated trajectory gap), the selector falls back to the temporally closest tracked frame for `event.track_id` and records `used_nearest_fallback = True`.

### Why `highest_confidence` Is Default
On `people-detection.mp4` (`768×432 @ 12.0 fps`, `596` frames), instantaneous `zone_entry` events at `entrance_area` occur as the person steps through the bottom boundary of the camera view (`y2 ≈ 431`), where the body is partially truncated. Over multi-second `dwell_time_event` windows (`1.33s` to `2.33s`), `highest_confidence` naturally selects the frame where the person's full silhouette is sharpest and least occluded (`mean_best_confidence = 0.8912`), producing clean full-body crops (`crops/<event_id>_<track_id>.jpg`).

---

## 2. Empirical Results on `data/fallback_clips/people-detection.mp4`

Command executed:
```bash
uv run --locked python cv_projects/07_visual_evidence/scripts/extract_visual_evidence.py \
  --source data/fallback_clips/people-detection.mp4 \
  --events cv_projects/06_dwell_time_events/outputs/people_detection/events.csv \
  --tracks cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv \
  --output-dir cv_projects/07_visual_evidence/outputs/people_detection
```

| Metric | Value |
|---|---|
| Total input events (`events.csv`) | `48` (`16 zone_entry`, `13 zone_exit`, `10 dwell_time_event`, `9 line_crossing`) |
| Unique best frames decoded (single pass) | `34` (out of `596` video frames) |
| Mean selected detector confidence | `0.8912` |
| Nearest-frame fallback count | `0` (`100%` of events had tracked boxes inside `[min_frame, max_frame]`) |
| Crops clamped at frame boundary (`clipped_crop_count`) | `7` / `48` (all at `entrance_area` bottom boundary `y2 -> 432`) |
| Resolved snapshots (`snapshots/<event_id>_<track_id>.jpg`) | `48` / `48` (`all_paths_valid = true`) |
| Resolved crops (`crops/<event_id>_<track_id>.jpg`) | `48` / `48` (`all_paths_valid = true`) |
| Pipeline runtime | `1.11s` |

---

## 3. Sample Visual Audit (`snapshots/` & `crops/`)

- **`evt_0001` (`zone_entry`, Track `1`, `entrance_area`, `t = 1.5833s`, `f = 19`, `conf = 0.80`)**:
  - Track `1` enters the bottom of the frame (`entrance_area`); crop is clamped cleanly at `y2 = 432` (`clipped_at_frame_edge = True`).
- **`evt_0002` (`dwell_time_event`, Track `1`, `entrance_area`, `1.5833s -> 2.9167s`, `f = 35`, `conf = 0.91`)**:
  - `highest_confidence` selects frame `35` when Track `1` is fully visible standing in `entrance_area`.
- **`evt_0006` (`dwell_time_event`, Track `1`, `center_concourse`, `3.0000s -> 4.5833s`, `f = 44`, `conf = 0.92`)**:
  - Snapshot `snapshots/evt_0006_1.jpg` clearly shows the gold bounding box, foot-point anchor dot inside `center_concourse`, and HUD label `evt_0006 | dwell_time_event | T1 | center_concourse | f=44 (0.92)`.
- **`evt_0012` (`dwell_time_event`, Track `3`, `entrance_area`, `15.0833s -> 17.0000s`, `f = 198`, `conf = 0.91`)**:
  - Full-body crop `crops/evt_0012_3.jpg` captures Track `3` walking forward into the concourse with 10% context padding.
