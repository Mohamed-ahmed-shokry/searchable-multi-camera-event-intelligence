# Searchable Multi-Camera Event Intelligence

Searchable Multi-Camera Event Intelligence is a computer-vision-first project for turning video footage into structured intelligence: detections, tracks, trajectories, events, visual evidence, summaries, analytics, and eventually searchable review workflows.

The long-term vision is a full video intelligence system. The current priority is smaller and more deliberate: build the computer vision foundation first.

```text
video -> vision -> behavior -> events -> search -> insight
```

## Core Idea

This project is not just an object detection demo, and it is not a dashboard project with a detector attached.

The goal is to understand what happens in video by moving through a complete CV reasoning pipeline:

```text
video source
-> frame inspection
-> object detection
-> single-camera tracking
-> trajectory extraction
-> spatial reasoning
-> event generation
-> visual evidence
-> structured outputs
-> searchable review later
```

Early work should prove that the system can understand movement and behavior in video before investing in backend, frontend, deployment, or product features.

## Current Project Phase

This repository may become a full software system later, but the current phase is focused on the computer vision foundation.

The first practical direction is single-camera, person-focused video intelligence:

- inspect one video source
- detect people
- track people over time
- extract trajectories
- reason about zones and lines
- generate meaningful events
- save visual evidence
- produce structured outputs
- evaluate what works and what fails

Productization comes later, after the CV outputs are useful.

## Setup and Checks

Use Python 3.12 (the tested interpreter is 3.12.12) and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The default install
contains NumPy, OpenCV, Shapely, and PyYAML plus dev tooling (`pytest`, `ruff` in the
`dev` dependency group).

```bash
uv sync --locked
uv run --locked pytest -v                # full test suite
uv run --locked ruff check .             # lint
uv run --locked ruff format --check .    # formatting check
uv run --locked python cv_projects/01_video_inspection/scripts/inspect_video.py --help
```

Continuous integration (GitHub Actions, `.github/workflows/ci.yml`) runs exactly
these checks — `uv sync --locked`, both ruff gates, and `pytest` — on every push,
pull request, and manual dispatch, on `windows-latest` (the development OS).

After inspecting real footage, validate the produced `video_info.json` against
the published dataset specs:

```bash
uv run --locked python cv_projects/01_video_inspection/scripts/validate_dataset.py <run_dir>/video_info.json
```

Choose one detection/tracking environment when needed:

```bash
# NVIDIA CUDA 12.8 wheels on Windows/Linux
uv sync --locked --extra detection-cuda

# CPU-only detection environment
uv sync --locked --extra detection-cpu

# Optional notebook tools can be combined with either environment
uv sync --locked --extra detection-cuda --group notebooks

# Run Phase 2 person detection
uv run python cv_projects/02_person_detection/scripts/detect_people.py <source_path>

# Benchmark YOLO11 model variants (n vs s vs m)
uv run python cv_projects/02_person_detection/scripts/compare_models.py <source_path>

# Run Phase 3 person tracking (ByteTrack / BoT-SORT)
uv run python cv_projects/03_person_tracking/scripts/track_people.py <source_path>

# Compare ByteTrack vs BoT-SORT and generate MOT17 checkpoint_metrics.json
uv run python cv_projects/03_person_tracking/scripts/compare_trackers.py <source_path>

# Run Phase 4 trajectory extraction (runs in the default locked environment)
uv run --locked python cv_projects/04_trajectory_extraction/scripts/extract_trajectories.py <source_path> --tracks-csv <tracks.csv>

# Compare anchor strategies and rolling-mean smoothing windows
uv run --locked python cv_projects/04_trajectory_extraction/scripts/compare_smoothing.py <tracks.csv>

# Preview or interactively author Phase 5 polygon zones and tripwire lines
uv run --locked python cv_projects/05_zone_line_reasoning/scripts/define_scene_geometry.py <source_path>

# Run Phase 5 zone and line reasoning (runs in the default locked environment)
uv run --locked python cv_projects/05_zone_line_reasoning/scripts/reason_zones_lines.py <source_path> --trajectories-csv <trajectories.csv>

# Run Phase 6 stateful dwell-time event generation (runs in the default locked environment)
uv run --locked python cv_projects/06_dwell_time_events/scripts/generate_dwell_events.py <source_path> --trajectories-csv <trajectories.csv>

# Run Phase 7 visual evidence extraction (runs in the default locked environment)
uv run --locked python cv_projects/07_visual_evidence/scripts/extract_visual_evidence.py --source <source_path> --events <events.csv> --tracks <tracks.csv>

# Run Phase 8 spatial heatmaps and zone/line analytics (runs in the default locked environment)
uv run --locked python cv_projects/08_heatmaps_analytics/scripts/generate_heatmaps_analytics.py --source <source_path> --trajectories <trajectories.csv> --events <events.csv>

# Run Phase 9 quantitative MOT17 and qualitative surveillance CV evaluation
uv run --locked python cv_projects/09_cv_evaluation/scripts/evaluate_pipeline.py --config cv_projects/09_cv_evaluation/configs/evaluation.yaml --output-dir cv_projects/09_cv_evaluation/outputs/full_evaluation

# Run Phase 10 SQLite ingestion and CLI event/track/evidence search
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py --config cv_projects/10_searchable_outputs/configs/search.yaml --output-dir cv_projects/10_searchable_outputs/outputs/people_detection --build
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db --event-type dwell_time_event --zone right_aisle

# Run Phase 11 unified end-to-end CV pipeline (video -> vision -> behavior -> events -> storage)
uv run --locked python -m event_intelligence.pipeline --video data/fallback_clips/people-detection.mp4 --config configs/pipeline.yaml
uv run --locked python -m event_intelligence.pipeline --video data/mot17/MOT17-11-FRCNN --config configs/pipeline_mot17.yaml

# Query a pipeline run's SQLite database directly via event_intelligence.search
uv run --locked python -m event_intelligence.search --db outputs/runs/people_detection/intelligence.db --event-type dwell_time_event --zone right_aisle --json
uv run --locked python -m event_intelligence.search --db outputs/runs/people_detection/intelligence.db --tracks --min-duration 5.0 --json
```

The two detection extras conflict intentionally; do not use `--all-extras`.
Dependencies are exactly pinned and `uv.lock` is committed. This follows
[uv's explicit PyTorch index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/).
Installing CUDA wheels does not prove that a driver/GPU is available or that
inference has run. A plain `uv sync` can remove optional packages; include the
chosen extra when resynchronizing a detection environment.

## Questions The System Should Eventually Answer

The project should grow toward answering questions such as:

- Who appeared in this video?
- Where did each person move?
- Which zones did they enter or exit?
- Which lines did they cross?
- How long did they stay in an area?
- Which events were important?
- What snapshot, crop, or clip supports each event?
- Can the results be searched after processing?
- Can activity be summarized across videos or cameras later?

## CV-First Development Philosophy

The project should grow in this order:

```text
CV experiment
-> reusable CV module
-> event intelligence pipeline
-> searchable outputs
-> inspection dashboard
-> full product later
```

The early goal is engineering depth, not a polished application shell.

Prefer focused experiments that answer:

- What problem does this CV capability solve?
- What input does it need?
- What output does it produce?
- How can the output be visually checked?
- How can it be evaluated?
- What are its limitations?
- How could it become part of the main system later?

## Main CV Capabilities

The current roadmap focuses on these capabilities:

- video inspection
- person detection
- person tracking
- trajectory extraction
- polygon zone reasoning
- line crossing detection
- dwell-time analysis
- event lifecycle and deduplication
- snapshots and crops as visual evidence
- heatmaps and activity analytics
- CV evaluation
- structured searchable outputs

Advanced features can come later, including multi-camera processing, cross-camera identity association, real-time streams, custom training, and richer product workflows.

## Project Roadmap

Build the project in phases. Each phase should be small enough to test, review, and document on its own.

| Phase | Focus | Goal |
| --- | --- | --- |
| 0 | Project foundation | Define direction, rules, docs, and experiment style. |
| 1 | Video inspection | Read video metadata and inspect sample frames. |
| 2 | Person detection | Detect people and produce annotated visual proof. |
| 3 | Person tracking | Assign stable IDs to people over time. |
| 4 | Trajectory extraction | Convert tracks into timestamped movement paths. |
| 5 | Zone and line reasoning | Detect zone entry, zone exit, and line crossing. |
| 6 | Dwell-time events | Measure time in zones and generate clean events. |
| 7 | Visual evidence | Save snapshots and crops that support events. |
| 8 | Heatmaps and analytics | Summarize movement and activity patterns. |
| 9 | CV evaluation | Measure quality, speed, failures, and limitations. |
| 10 | Searchable outputs | Query structured CV results without a full product. |
| 11 | CV pipeline integration | Consolidate proven experiments into one runnable `src/event_intelligence/` system. |
| 12 | Productization later | Add dashboard, API, database, jobs, auth, and deployment when justified. |

Do not skip directly to productization unless the project direction changes explicitly. See
[docs/project_phases.md](docs/project_phases.md) for the canonical, detailed phase plan
(Definition of Done per phase, locked decisions, dependency schedule).

## CV Subprojects

Use `cv_projects/` for research-style computer vision experiments and milestones.

Recommended structure:

```text
cv_projects/
  01_video_inspection/
  02_person_detection/
  03_person_tracking/
  04_trajectory_extraction/
  05_zone_line_reasoning/
  06_dwell_time_events/
  07_visual_evidence/
  08_heatmaps_analytics/
  09_cv_evaluation/
  10_searchable_outputs/
```

Each subproject should focus on meaningful CV capability, not a whole product feature.

## Notebook-Based Workflow

Each CV subproject should support exploration and reproducibility.

Recommended subproject layout:

```text
cv_projects/<number>_<capability>/
  README.md
  notebooks/
    01_explore_<topic>.ipynb
  scripts/
    run_<capability>.py
  configs/
    <capability>.yaml
  outputs/
  notes.md
```
- repeatable demos

Use shared source modules later for:

- stable reusable CV logic
- cleaned implementations
- code that has moved beyond one experiment

The intended flow is:

```text
notebook exploration -> reproducible script -> reusable module
```

## Expected Output Artifacts

Each processed run may eventually produce artifacts such as:

```text
outputs/runs/<run_id>/
  summary.json
  detections.csv
  tracks.csv
  trajectories.csv
  events.csv
  annotated_video.mp4
  heatmap.png
  snapshots/
  crops/
  evaluation_report.md
```

Not every phase needs every artifact. Early phases should produce only the outputs needed to verify that specific CV capability.

## Event Intelligence Direction

The system should not generate noisy frame-by-frame alerts. Event logic should eventually reason about state over time.

Example event types:

- zone entry
- zone exit
- line crossing
- dwell-time event
- stopped person or object
- occupancy alert
- restricted-zone entry
- heatmap generated

Important event principle:

```text
one meaningful behavior -> one clean event
```

For example, if a person stays in a zone for more than a threshold, the system should generate one dwell-time event, not hundreds of repeated frame alerts.

## Searchable Outputs Direction

Search comes after useful CV outputs exist.

Start with structured search over exported artifacts, such as:

- event type
- video
- time range
- track ID
- zone
- line
- duration
- evidence path

Natural-language search, dashboards, APIs, and product workflows can come later.

## What Is Deferred

Do not start with:

- React dashboard
- backend API
- authentication
- deployment
- cloud infrastructure
- complex database design
- real-time RTSP processing
- custom model training
- multi-camera re-identification
- license plate recognition
- face recognition
- large product architecture

These can be valuable later, but the first priority is a working CV intelligence pipeline.

## Repository Status

The repository has completed **Phase 0 (Foundation)**, **Phase 1 (Video Inspection & Dataset Validation)**, **Phase 2 (Person Detection)**, **Phase 3 (Person Tracking & Early Quantitative MOT17 Checkpoint)**, **Phase 4 (Trajectory Extraction)**, **Phase 5 (Zone and Line Reasoning)**, **Phase 6 (Dwell-Time Events)**, **Phase 7 (Visual Evidence)**, **Phase 8 (Heatmaps and Analytics)**, **Phase 9 (CV Evaluation)**, **Phase 10 (Searchable Outputs)**, and **Phase 11/11c (CV Pipeline Integration & Multi-Camera Search Intelligence)**.

Completed capabilities:

- `cv_projects/01_video_inspection/` — unified `VideoSource` reader, contact sheets, and MOT17 spec validator
- `cv_projects/02_person_detection/` — CUDA-verified YOLO11 person detection, `detections.csv`, and `n`/`s`/`m` model benchmark
- `cv_projects/03_person_tracking/` — ByteTrack and BoT-SORT person tracking, `tracks.csv`, `track_summary.csv`, `annotated_tracking_video.mp4`, and CLEAR-MOT (`MOTA`/`IDF1`/`IDSW`) checkpoint evaluation
- `cv_projects/04_trajectory_extraction/` — ordered ground-plane trajectories (`trajectories.csv`), per-track path-length summaries (`track_summary.csv`), causal rolling-mean smoothing, static `trajectory_map.png`, and `trajectory_overlay_video.mp4`
- `cv_projects/05_zone_line_reasoning/` — named polygon zones and directed tripwire lines (`scene_config.yaml`), visual scene authoring (`define_scene_geometry.py`), `zone_enter`/`zone_exit`/`line_cross` signals (`zone_line_events.csv`), boundary flicker analysis, and `annotated_spatial_video.mp4`
- `cv_projects/06_dwell_time_events/` — stateful `ZoneDwellStateMachine` (`candidate -> active -> triggered -> resolved`), hysteresis grace-period flicker absorption, track-loss-in-zone handling, `events.csv`, `event_debug_log/`, `dwell_examples/`, and `annotated_events_video.mp4`
- `cv_projects/07_visual_evidence/` — configurable best-frame selection (`highest_confidence`, `largest_area`, `midpoint`, `start_frame`), annotated full-frame snapshots (`snapshots/<event_id>_<track_id>.jpg`), padded person crops (`crops/<event_id>_<track_id>.jpg`), enriched 9-column `events.csv`, contact sheet (`evidence_contact_sheet.jpg`), and automated path verification
- `cv_projects/08_heatmaps_analytics/` — 2D spatial trajectory density accumulation with per-track-cell stationary capping, Gaussian smoothing, reference-frame heatmap overlay (`heatmap.png`), per-zone counts (`zone_counts.csv`), per-line directional counts (`line_counts.csv`), track duration distributions, and `analytics_summary.json`
- `cv_projects/09_cv_evaluation/` — quantitative MOT17 evaluation (`Precision`, `Recall`, `F1`, `MOTA`, `IDF1`, `IDSW`), runtime FPS benchmark, qualitative surveillance clip error taxonomy (`false_positive`, `false_negative`, `id_switch`, `boundary_jitter`, `event_rule_edge_case`), annotated failure examples (`failure_examples/`), and structured `evaluation_report.md`
- `cv_projects/10_searchable_outputs/` — normalized 7-table SQLite database (`intelligence.db`), CLI and programmatic query engine (`search_events.py`), schema documentation (`schema.md`), and worked query examples (`search_examples.md`)
- `src/event_intelligence/` — consolidated end-to-end CV package (`video`, `detection`, `tracking`, `trajectories`, `spatial`, `events`, `evidence`, `analytics`, `evaluation`, `storage`, `pipeline`, `search`) runnable end-to-end:
  - **Single camera:** `uv run --locked python -m event_intelligence.pipeline --video data/fallback_clips/people-detection.mp4 --config configs/pipeline.yaml`
  - **Multi-camera:** `uv run --locked python -m event_intelligence.pipeline --videos data/fallback_clips/people-detection.mp4 data/fallback_clips/store-aisle-detection.mp4 --cameras entrance concourse --config configs/pipeline_multicam.yaml`
  - **Cross-camera search:** `uv run --locked python -m event_intelligence.search --db outputs/runs/<run_id>/intelligence.db --camera-summary`
  - **Camera filtering:** `uv run --locked python -m event_intelligence.search --db outputs/runs/<run_id>/intelligence.db --camera entrance --tracks --json`

Next recommended step:

```text
Phase 12 — Productization later (deferred until explicitly requested)
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
