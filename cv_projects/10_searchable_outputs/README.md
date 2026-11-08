# Phase 10 — Searchable Outputs (`10_searchable_outputs`)

## Goal

Query every structured CV output (`events`, `tracks`, `trajectories`, `detections`, `zone_counts`,
`line_counts`, and visual evidence `snapshots/` + `crops/`) without a product or dashboard by
loading artifacts into a documented SQLite database (`intelligence.db`) and querying via
`scripts/search_events.py`.

## Directory Structure

```text
cv_projects/10_searchable_outputs/
  README.md
  notes.md
  schema.md
  search_examples.md
  configs/
    search.yaml
  notebooks/
    01_explore_searchable_outputs.ipynb
  scripts/
    search_events.py
  outputs/
    people_detection/
      intelligence.db
      search.yaml
      search_summary.json
      schema.md
      search_examples.md
```

## Quick Start

```bash
# Build intelligence.db and generate schema.md + search_examples.md
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --config cv_projects/10_searchable_outputs/configs/search.yaml \
    --build

# Query dwell events in right_aisle
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --event-type dwell_time_event --zone right_aisle
```
