# 03 Person Tracking

Keep consistent `track_id` values across frames using ByteTrack and BoT-SORT, produce schema-compliant `tracks.csv` and `track_summary.csv` exports, render annotated tracking videos, and evaluate MOTA/IDF1/ID-switches on MOT17 ground truth.

## Planned Outputs

- `outputs/<run>/tracks.csv` — `frame_id, track_id, x1, y1, x2, y2, confidence`
- `outputs/<run>/track_summary.csv` — `track_id, start_frame, end_frame, duration_frames, duration_seconds, observed_frames`
- `outputs/<run>/annotated_tracking_video.mp4` — per-track colored bounding boxes and `ID <id>` labels
- `outputs/<run>/checkpoint_metrics.json` — early quantitative checkpoint (`MOTA`, `IDF1`, `id_switches`, `fps`, `num_tracks`)
- `notes.md` — ByteTrack vs BoT-SORT comparison and observed ID-switch/fragmentation cases
