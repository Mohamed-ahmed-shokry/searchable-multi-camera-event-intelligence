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
| 11 | Productization later | Add dashboard, API, database, jobs, auth, and deployment when justified. |

Do not skip directly to productization unless the project direction changes explicitly.

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

Each subproject should focus on one meaningful CV capability, not a whole product feature.

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

Use notebooks for:

- exploration
- visual debugging
- comparing ideas
- reviewing failures
- recording observations

Use scripts for:

- reproducible runs
- consistent outputs
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

The repository is currently in the planning and foundation stage.

Current completed foundation work:

- project direction defined
- agent rules added in `AGENTS.md`
- CV-first roadmap agreed
- notebook-based subproject workflow agreed

Next recommended step:

```text
Create docs/project_phases.md or start cv_projects/01_video_inspection/
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
