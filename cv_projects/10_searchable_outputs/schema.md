# Phase 10 SQLite Schema — `intelligence.db`

Source video: **`people-detection.mp4`**

## 1. Table Row Count Summary

| Table Name | Primary Key | Secondary Indexes | Row Count | Description |
| --- | --- | --- | ---: | --- |
| `detections` | `detection_id` | `idx_detections_frame_id (frame_id)` | `418` | Phase 2 person detections (`x1, y1, x2, y2, confidence, class_name`) |
| `track_observations` | `observation_id` | `idx_track_obs_track_frame (track_id, frame_id)` | `409` | Phase 3 per-frame tracked bounding boxes |
| `tracks` | `track_id` | `idx_tracks_duration (duration_seconds DESC)` | `7` | Per-identity lifespan and ground-plane `path_length_px` |
| `trajectories` | `(track_id, frame_id)` | `idx_trajectories_timestamp (timestamp)` | `409` | Phase 4 smoothed `bottom_center` foot-anchor coordinates |
| `events` | `event_id` | `idx_events_type_zone`, `idx_events_track_id`, `idx_events_time_range` | `48` | Phase 6/7 structured events with `snapshot_path` and `crop_path` |
| `zone_counts` | `zone_id` | — | `3` | Phase 8 per-zone entries, exits, total dwell, and peak occupancy |
| `line_counts` | `line_id` | — | `2` | Phase 8 per-line directional crossings (`A->B`, `B->A`, `total`) |

---

## 2. SQLite DDL (`SCHEMA_DDL`)

```sql
CREATE TABLE IF NOT EXISTS detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL,
    class_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON detections (frame_id);

CREATE TABLE IF NOT EXISTS track_observations (
    observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    confidence REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_track_obs_track_frame ON track_observations (track_id, frame_id);

CREATE TABLE IF NOT EXISTS tracks (
    track_id INTEGER PRIMARY KEY,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    duration_frames INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    observed_frames INTEGER NOT NULL,
    path_length_px REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tracks_duration ON tracks (duration_seconds DESC);

CREATE TABLE IF NOT EXISTS trajectories (
    track_id INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    PRIMARY KEY (track_id, frame_id)
);
CREATE INDEX IF NOT EXISTS idx_trajectories_timestamp ON trajectories (timestamp);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    zone_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    snapshot_path TEXT NOT NULL,
    crop_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type_zone ON events (event_type, zone_id);
CREATE INDEX IF NOT EXISTS idx_events_track_id ON events (track_id);
CREATE INDEX IF NOT EXISTS idx_events_time_range ON events (start_time, end_time);

CREATE TABLE IF NOT EXISTS zone_counts (
    zone_id TEXT PRIMARY KEY,
    entry_count INTEGER NOT NULL,
    exit_count INTEGER NOT NULL,
    total_dwell_seconds REAL NOT NULL,
    peak_occupancy INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS line_counts (
    line_id TEXT PRIMARY KEY,
    crossings_ab INTEGER NOT NULL,
    crossings_ba INTEGER NOT NULL,
    total INTEGER NOT NULL
);
```
