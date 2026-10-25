# 04 Trajectory Extraction

Convert multi-object person tracks (`tracks.csv`) into timestamped ground-plane movement paths (`trajectories.csv`) using the bottom-center bounding-box anchor point, evaluate raw vs rolling-mean smoothed trajectories, compute per-track path length and duration (`track_summary.csv`), and render color-coded trajectory trail overlays (`trajectory_overlay_video.mp4` and `trajectory_map.png`).

## Planned Outputs

- `outputs/<run>/trajectories.csv` — `track_id, frame_id, timestamp, x, y`
- `outputs/<run>/track_summary.csv` — `track_id, path_length_px, duration_seconds, frame_count`
- `outputs/<run>/trajectory_overlay_video.mp4` — video with temporal movement trails and foot anchor points
- `outputs/<run>/trajectory_map.png` — full-scene reference frame with complete trajectory paths (`O` start, `X` end)
- `outputs/<run>/smoothing_comparison.json` — raw vs rolling-mean path length and jitter metrics
- `notes.md` — anchor-point rationale, smoothing comparison, and frame-by-frame track cross-check
