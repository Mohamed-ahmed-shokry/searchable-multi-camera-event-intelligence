# Phase 10 Notes — Searchable Outputs (`intelligence.db`)

## 1. Ingestion Summary (`people-detection.mp4`)

Command executed:

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --config cv_projects/10_searchable_outputs/configs/search.yaml \
    --output-dir cv_projects/10_searchable_outputs/outputs/people_detection \
    --build
```

Ingested row counts across the 7 SQLite tables in `intelligence.db`:

| Table | Row Count | Upstream Artifact |
| --- | ---: | --- |
| `detections` | `418` | `cv_projects/02_person_detection/outputs/people_detection_run/detections.csv` |
| `track_observations` | `409` | `cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv` |
| `tracks` | `7` | Joined from Phase 3 `track_summary.csv` and Phase 4 `track_summary.csv` |
| `trajectories` | `409` | `cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv` |
| `events` | `48` | `cv_projects/07_visual_evidence/outputs/people_detection/events.csv` |
| `zone_counts` | `3` | `cv_projects/08_heatmaps_analytics/outputs/people_detection/zone_counts.csv` |
| `line_counts` | `2` | `cv_projects/08_heatmaps_analytics/outputs/people_detection/line_counts.csv` |

All `48 / 48` event rows in `intelligence.db` were verified against `cv_projects/07_visual_evidence/outputs/people_detection/` (`valid_evidence_events = 48`, `all_evidence_valid = true`).

## 2. Key Technical Findings

1. **Dual `track_summary.csv` Schema Unification:**
   - Phase 3 (`03_person_tracking`) records frame lifespan (`start_frame`, `end_frame`, `duration_frames`, `duration_seconds`, `observed_frames`), while Phase 4 (`04_trajectory_extraction`) records ground-plane displacement (`path_length_px`, `duration_seconds`, `frame_count`).
   - `ingest_intelligence_db` joins both summaries by `track_id` (and computes a fallback from `trajectories.csv` if either summary is omitted) so `query_tracks` can filter by both duration (`[min_duration, max_duration]`) and `min_path_length_px` in a single query.
2. **Directed Line-Crossing Prefix Matching:**
   - Phase 5/6/7 store directed line crossings in `events.zone_id` as `<line_id>:A->B` or `<line_id>:B->A` (e.g., `doorway_line:B->A`).
   - `query_events` uses `(zone_id = ? OR zone_id LIKE ?)` (`f"{zone_id}:%"`) when the caller passes an undirected line name (`--zone doorway_line`), while still supporting exact directional filters (`--zone "doorway_line:B->A"`).
3. **Windows SQLite Connection Lifecycle:**
   - Python's `with sqlite3.connect(db_path) as conn:` context manager commits/rolls back transactions on exit without closing the file handle, causing `PermissionError: [WinError 32]` on Windows when temporary directories are cleaned up.
   - Wrapping every `sqlite3.connect(...)` call in `contextlib.closing(sqlite3.connect(...))` guarantees deterministic handle release on Windows and Linux.
