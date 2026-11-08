# Phase 10 — Searchable Outputs (`cv_projects/10_searchable_outputs/`)

Phase 10 loads all structured outputs from Phases 2–8 (`detections`, `track_observations`, `tracks`, `trajectories`, `events`, `zone_counts`, and `line_counts`) into a normalized, indexed SQLite database (`intelligence.db`) and provides a CLI and programmatic query engine (`scripts/search_events.py`) for events, tracks, and visual evidence.

---

## 1. Quick Start

### Build `intelligence.db`, `schema.md`, and `search_examples.md`

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --config cv_projects/10_searchable_outputs/configs/search.yaml \
    --output-dir cv_projects/10_searchable_outputs/outputs/people_detection \
    --build
```

### Query Events by Type and Zone

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --event-type dwell_time_event --zone right_aisle
```

### Query Directed Line Crossings in a Time Window

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --event-type line_crossing --zone doorway_line --min-time 15.0 --max-time 20.0
```

### Query Tracks by Duration and Path Length

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --tracks --min-duration 5.0 --min-path-length 350.0
```

### Look Up and Verify Visual Evidence by `event_id`

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --evidence-event-id evt_0002 --json
```

---

## 2. Directory Layout

```text
cv_projects/10_searchable_outputs/
  README.md
  schema.md
  search_examples.md
  notes.md
  configs/
    search.yaml
  notebooks/
    01_explore_searchable_outputs.ipynb
  scripts/
    search_events.py
  outputs/
    people_detection/
      intelligence.db
      schema.md
      search_examples.md
      search_summary.json
      search.yaml
```
