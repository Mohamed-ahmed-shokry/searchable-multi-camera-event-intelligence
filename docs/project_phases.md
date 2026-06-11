# Project Phases — Searchable Multi-Camera Event Intelligence

This is the canonical phase plan. Build in order. A phase is done only when its
Definition of Done checklist passes.

```text
video -> vision -> behavior -> events -> search -> insight
```

```text
Track A: CV Experiments      (cv_projects/  — Phases 1–10)
Track B: CV System Build     (src/          — Phase 11)
Track C: Productization      (deferred      — Phase 12)
```

---

## Key Decisions (locked in)

| Decision | Choice | Why |
|----------|--------|-----|
| Test footage | **MOT17 + VIRAT** public datasets | MOT17 has ground-truth annotations → real quantitative evaluation. VIRAT is surveillance-style static-camera footage → realistic zones, lines, dwell scenarios. |
| Footage fallback | Any static-camera surveillance-style clip (own camera, public-domain CCTV-style) | VIRAT downloads (data.kitware.com) are large and access can be flaky. Phases 5–8 need static surveillance-style footage, not VIRAT specifically. |
| Hardware | **RTX 3060 Mobile, 6 GB VRAM (CUDA)** | Inference-only workload. `yolo11m` baseline is comfortable for inference; `yolo11l` possible but compare speed first. No custom training planned, so 6 GB is not a constraint. |
| Detector | `ultralytics` YOLO11 (pretrained), `yolo11m` baseline | Best-supported, built-in tracking, no custom training needed. Compare `n`/`s`/`m` in Phase 2. |
| Tracker | ByteTrack (default), BoT-SORT (comparison) | Both built into Ultralytics; compared quantitatively in Phase 3. |
| Geometry | `shapely` | Standard, simple polygon/line math. |
| Search store | `sqlite3` (stdlib), `duckdb` only if justified | Zero-infrastructure searchability. |
| Package name | `event_intelligence` (under `src/`) | One top-level package; importable, installable, runnable as `python -m event_intelligence.pipeline`. |
| Version pinning | Pin `ultralytics` and `torch` in `pyproject.toml`; record model weights version in configs | Ultralytics changes tracker behavior between releases; reproducibility depends on pinning. |

**Dataset usage strategy:**

- **MOT17 sequences** → Phases 2–4 and Phase 9 (detection, tracking, trajectories, evaluation against ground truth)
- **VIRAT clips (or fallback static footage)** → Phases 5–8 (zones, lines, dwell, evidence, heatmaps)

**License note:** MOT17 is CC BY-NC-SA (non-commercial). Fine for evaluation and
learning; evaluation results do not contaminate a future product. Record dataset
license terms in `data/README.md`. Do not redistribute footage.

---

## Core Principles

1. Build in phase order. Do not skip ahead.
2. A phase is done only when its **Definition of Done** checklist passes.
3. Notebooks for exploration → scripts for reproducibility → `src/` for the integrated system.
4. **One meaningful behavior → one clean event.** Event logic is stateful, never frame-reactive.
5. Code is promoted to `src/` only in Phase 11, after the experiment proved the approach.
   Exception: small shared utilities (e.g. video reader helpers) may be promoted earlier
   if two or more phases need them — with explicit approval.
6. Productization starts only after the integrated pipeline runs end-to-end on one command.

---

## Repository Structure

```text
searchable-multi-camera-event-intelligence/
  cv_projects/                  ← Track A: phase experiments
  src/
    event_intelligence/         ← Track B: the integrated CV system (built in Phase 11)
      video/                    ←   reading, metadata, frame sampling
      detection/                ←   person detection
      tracking/                 ←   ID tracking
      trajectories/             ←   path extraction
      spatial/                  ←   zones, lines, geometry
      events/                   ←   stateful event lifecycle
      evidence/                 ←   snapshots, crops
      analytics/                ←   heatmaps, summaries
      storage/                  ←   searchable output store
      pipeline.py               ←   orchestrates all stages
  configs/                      ← shared configs (model, zones, thresholds, pipeline)
  data/                         ← MOT17 / VIRAT clips (gitignored)
  outputs/                      ← generated run artifacts (gitignored)
  docs/                         ← this plan, experiment template, findings
  tests/                        ← tests for event/geometry/trajectory logic (started in Phase 6)
  AGENTS.md
  CLAUDE.md
  README.md
  pyproject.toml
```

`.gitignore` for `data/`, `outputs/`, and `cv_projects/**/outputs/` is part of the
first commit.

---

## Subproject Pattern (Track A)

```text
cv_projects/<NN>_<capability>/
  README.md           ← goal, approach, outputs of this phase
  notebooks/
    01_explore_*.ipynb
  scripts/
    run_*.py
  configs/
    *.yaml
  outputs/            ← gitignored
  notes.md            ← observations, failures, limitations, decisions
```

---

## Dependency Introduction Schedule

Dependencies are added only when their phase begins.

| Phase | `uv add` |
|-------|----------|
| 1 | `opencv-python`, `numpy`, `jupyter` |
| 2 | `ultralytics` (+ CUDA-enabled `torch` — verify `torch.cuda.is_available()` first; pin versions) |
| 3 | `pandas`, `motmetrics` (early tracking checkpoint — see Phase 3) |
| 5 | `shapely`, `pyyaml` |
| 6 | `pytest` (event state machine tests — see Phase 6) |
| 8 | `matplotlib`, `seaborn` |
| 10 | none (`sqlite3` is stdlib); `duckdb` only if justified |

---

# Track A — CV Experiments (Phases 1–10)

---

## Phase 1 — Video Inspection + Data Acquisition

**Subproject:** `cv_projects/01_video_inspection/`

**Goal:** Acquire test footage and fully understand it before running any models.

**Work:**

- Research and download 2–3 MOT17 sequences and 2–3 VIRAT clips into `data/`
  (documented in `data/README.md`: source URLs, licenses, which clips chosen and why)
- If VIRAT access proves painful, use the fallback: any static-camera
  surveillance-style footage with people moving through definable zones
- Note: MOT17 ships as JPEG frame sequences + `seqinfo.ini`, not video files —
  handle both inputs (video file OR frame directory) from the start
- Read metadata: FPS, resolution, frame count, duration, codec
- Sample frames at regular intervals; build a contact sheet per source
- Assess scene quality: lighting, camera angle, crowd density, occlusion levels

**Outputs:**

```text
video_info.json          (one entry per source)
sample_frames/
contact_sheet_<source>.png
notes.md                 (scene assessment per clip)
data/README.md
```

**Risks / failure modes:**

- MOT17 frame-sequence format breaking assumptions of video-file-only code
- VIRAT files are large — pick short clips (1–3 min)
- VIRAT access friction — switch to fallback footage early rather than stalling

**Definition of Done:**

- [ ] At least one MOT17 sequence and one static-camera clip (VIRAT or fallback) in `data/` with documented provenance and license terms
- [ ] `video_info.json` validated against known dataset specs
- [ ] Contact sheets visually reviewed
- [ ] Reader works for both video files and frame directories

---

## Phase 2 — Person Detection

**Subproject:** `cv_projects/02_person_detection/`

**Goal:** Detect people and produce annotated visual proof, with GPU verified.

**Work:**

- Verify CUDA: `torch.cuda.is_available()` — record GPU model and VRAM in notes
  (expected: RTX 3060 Mobile, 6 GB)
- Run pretrained YOLO11 (`yolo11m` baseline; compare `n`/`s`/`m` speed-accuracy
  in notebook), person class only
- Confidence filtering (start 0.4; tune and record in config)
- Draw boxes + confidence on frames; render annotated video
- Save detections to CSV
- Pin `ultralytics`/`torch` versions and record model weights version in config

**Outputs:**

```text
detections.csv          (frame_id, x1, y1, x2, y2, confidence, class)
annotated_video.mp4
annotated_frames/
configs/detection.yaml  (model, confidence, device, weights version)
notes.md                (model comparison, FPS on GPU, failure observations)
```

**Risks / failure modes:** small/distant people in wide shots, heavy occlusion in
MOT17 crowds, false positives on mannequins/posters.

**Definition of Done:**

- [ ] CUDA confirmed and used (`device=0`)
- [ ] Annotated video reviewed — boxes on people, confidence visible
- [ ] Model size comparison recorded in notes.md
- [ ] Detection FPS measured and recorded
- [ ] Versions pinned in `pyproject.toml`

---

## Phase 3 — Person Tracking (with early quantitative checkpoint)

**Subproject:** `cv_projects/03_person_tracking/`

**Goal:** Stable IDs across frames, verified against ground truth before anything
is built on top of them.

**Work:**

- Run ByteTrack via Ultralytics; run BoT-SORT on the same clip and compare
- Record per-track: start frame, end frame, duration
- Annotated video with ID labels
- Count and log observed ID switches manually
- **Early quantitative checkpoint:** run `motmetrics` against MOT17 ground truth
  for a quick MOTA/IDF1 on at least one sequence. This is not the full Phase 9
  evaluation — it is a cheap gate that protects Phases 4–8 from being built on a
  quantitatively bad tracker.

**Outputs:**

```text
tracks.csv                       (frame_id, track_id, x1, y1, x2, y2, confidence)
track_summary.csv                (track_id, start_frame, end_frame, duration_frames, duration_seconds)
annotated_tracking_video.mp4
configs/tracking.yaml
checkpoint_metrics.json          (quick MOTA/IDF1 per tracker)
notes.md                         (ByteTrack vs BoT-SORT comparison, ID switch log)
```

**Risks / failure modes:** ID switches during occlusion or crossing paths; track
fragmentation; ID re-use after long absence — all directly corrupt downstream
dwell events, so log every observed case.

**Definition of Done:**

- [ ] IDs visibly stable for non-occluded people across the clip
- [ ] ByteTrack vs BoT-SORT compared quantitatively (MOTA/IDF1), one chosen with reason
- [ ] All observed ID switches logged in notes.md
- [ ] Checkpoint metrics recorded — if results are bad, fix tracking before Phase 4

---

## Phase 4 — Trajectory Extraction

**Subproject:** `cv_projects/04_trajectory_extraction/`

**Goal:** Convert tracks into timestamped movement paths.

**Work:**

- Anchor point: **bottom-center of bbox** (foot position — better ground-plane
  proxy than centroid for zone logic later); record this decision
- Compute `(x, y)` + timestamp per frame per track
  (MOT17 frame sequences have no real timestamps — derive from frame_id / FPS)
- Optional light smoothing (rolling mean) — compare raw vs smoothed in notebook
- Draw color-coded trails on reference frame and as overlay video
- Compute path length + duration per track

**Outputs:**

```text
trajectories.csv               (track_id, frame_id, timestamp, x, y)
track_summary.csv              (track_id, path_length_px, duration_seconds, frame_count)
trajectory_overlay_video.mp4
trajectory_map.png
notes.md
```

**Risks / failure modes:** bbox jitter making paths noisy; partial occlusion
shifting the bbox bottom; ID switches producing impossible "teleporting" paths.

**Definition of Done:**

- [ ] Anchor-point decision documented with reasoning
- [ ] Trails look like plausible human movement on visual review
- [ ] One track manually cross-checked against the source video

---

## Phase 5 — Zone and Line Reasoning

**Subproject:** `cv_projects/05_zone_line_reasoning/`

**Goal:** Spatial reasoning — zones and crossing lines.
(Primary footage: VIRAT or fallback static-camera clips.)

**Work:**

- Build a small helper notebook to click polygon points on a reference frame →
  writes `scene_config.yaml` (manual coordinate guessing is painful)
- Point-in-polygon per frame per track (shapely)
- Line crossing via side-of-line sign change between consecutive frames, with
  direction (A→B vs B→A)
- Emit raw entry/exit/crossing signals (deduplication is Phase 6)
- Render zones/lines on annotated video

**Config:**

```yaml
# scene_config.yaml
zones:
  - id: entrance_area
    polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
lines:
  - id: doorway_line
    points: [[x1, y1], [x2, y2]]
```

**Outputs:**

```text
scene_config.yaml
zone_line_events.csv           (frame_id, track_id, signal_type, zone_id, line_id, direction, timestamp)
annotated_spatial_video.mp4
notes.md
```

**Risks / failure modes:** boundary flicker (person hovering on a zone edge fires
rapid enter/exit signals — this is exactly what Phase 6 must absorb); trajectory
jitter causing false line crossings.

**Definition of Done:**

- [ ] Zones defined visually via the helper, not hand-typed coordinates
- [ ] Entry/exit signals match visual observation on review
- [ ] Line crossings include direction
- [ ] Boundary-flicker cases captured and documented for Phase 6

---

## Phase 6 — Dwell-Time Events (with unit tests)

**Subproject:** `cv_projects/06_dwell_time_events/`

**Goal:** Stateful event logic — one behavior, one clean event. This is the
highest-risk pure-logic component in the system, so it gets tests now, not in
Phase 11.

**Work:**

- Event lifecycle state machine per `(track_id, zone_id)`:

  ```text
  candidate → active → triggered → resolved
  ```

- Configurable dwell threshold; exactly one `dwell_time_event` per continuous stay
- Exactly one `zone_entry` / `zone_exit` per transition
- **Hysteresis / grace period**: brief boundary flickers (≤ N frames outside) do
  not reset dwell state — tune N from Phase 5 flicker observations
- Handle track loss inside a zone (resolve event at last seen time, flag it)
- Per-event debug trace for inspection
- **Unit tests (`tests/`)**: state machine transitions, hysteresis behavior,
  track-loss handling — pure logic, no model inference. Phase 11 extends this
  suite rather than starting from scratch.

**Outputs:**

```text
events.csv             (event_id, event_type, track_id, zone_id, start_time, end_time, duration_seconds)
configs/events.yaml    (thresholds, grace period)
event_debug_log/
dwell_examples/
tests/test_event_state_machine.py
notes.md
```

**Risks / failure modes:** ID switches splitting one stay into two events;
over-aggressive grace period merging genuinely separate visits; off-by-one
timing at entry/exit.

**Definition of Done:**

- [ ] Event count is orders of magnitude below frame count
- [ ] One track manually traced — event timeline matches the video
- [ ] Boundary flicker absorbed (verified against Phase 5 flicker cases)
- [ ] Track-loss-in-zone handled and flagged
- [ ] `pytest` passes on the event state machine

---

## Phase 7 — Visual Evidence

**Subproject:** `cv_projects/07_visual_evidence/`

**Goal:** Auditable proof attached to every event.

**Work:**

- Best-frame selection per event (e.g. highest detection confidence within the
  event window — document the rule)
- Full-frame snapshot + tight person crop (small padding) per event
- Deterministic naming: `<event_id>_<track_id>.jpg`
- Add `snapshot_path`, `crop_path` columns to events.csv

**Outputs:**

```text
snapshots/
crops/
events.csv             (with evidence path columns)
notes.md
```

**Risks / failure modes:** best frame having a partially occluded subject; crop
boundaries clipping at frame edges; seek inaccuracy in compressed video.

**Definition of Done:**

- [ ] Random sample of snapshot/crop pairs matches event descriptions
- [ ] All paths in events.csv resolve to real files
- [ ] Best-frame rule documented

---

## Phase 8 — Heatmaps and Analytics

**Subproject:** `cv_projects/08_heatmaps_analytics/`

**Goal:** Visual + statistical summary of activity.

**Work:**

- Accumulate trajectory points into a 2D grid; Gaussian smoothing; render heatmap
  overlaid on a reference frame
- Per-zone stats: entries, exits, total dwell, peak occupancy
- Per-line stats: crossings by direction
- Track duration distribution
- Structured activity summary JSON

**Outputs:**

```text
heatmap.png
analytics_summary.json
zone_counts.csv        (zone_id, entry_count, exit_count, total_dwell_seconds, peak_occupancy)
line_counts.csv        (line_id, crossings_ab, crossings_ba, total)
notes.md
```

**Risks / failure modes:** heatmap dominated by one stationary person; mismatched
grid vs frame resolution distorting the overlay.

**Definition of Done:**

- [ ] Hot spots correspond to visible activity in the footage
- [ ] Zone stats cross-checked against manual counts from events.csv
- [ ] Summary JSON validates and is human-readable

---

## Phase 9 — CV Evaluation

**Subproject:** `cv_projects/09_cv_evaluation/`

**Goal:** Measure quality honestly against ground truth and by manual review.
(The Phase 3 checkpoint was a gate; this is the full evaluation.)

**Work:**

- **Quantitative (MOT17 ground truth + `motmetrics`):** detection
  precision/recall, MOTA, IDF1, ID switch count
- **Qualitative (VIRAT / fallback, no labels):** manual count comparisons,
  zone/line event accuracy spot checks, dwell duration sanity checks against
  playback
- Processing FPS per pipeline stage on the GPU
- Collect failure examples (frames/clips) with explanations

**Outputs:**

```text
evaluation_report.md
metrics.json           (precision, recall, MOTA, IDF1, ID switches, per-stage FPS)
failure_examples/
notes.md
```

**Definition of Done:**

- [ ] MOTA/IDF1 computed on at least one MOT17 sequence
- [ ] Spot-check accuracy recorded for zone and line events
- [ ] Report names concrete failures and limitations — honesty over polish
- [ ] Per-stage FPS table included

---

## Phase 10 — Searchable Outputs

**Subproject:** `cv_projects/10_searchable_outputs/`

**Goal:** Query everything without a product.

**Work:**

- Load events, tracks, trajectories, detections into SQLite (`intelligence.db`)
  with a documented schema
- CLI script supporting: events by type / zone / track / time range; tracks by
  duration; evidence lookup by event ID
- Worked query examples with expected outputs

**Outputs:**

```text
intelligence.db
search_events.py
search_examples.md
schema.md
notes.md
```

**Definition of Done:**

- [ ] Every documented query runs and returns correct results
- [ ] Evidence paths returned by queries resolve to real files
- [ ] Schema documented

---

# Track B — Phase 11: CV Pipeline Integration

**Goal:** Consolidate ten proven experiments into **one coherent CV system** in
`src/event_intelligence/`, runnable end-to-end with a single command. This is
the bridge between experiments and product.

**Work:**

1. **Promote, don't rewrite.** Move proven logic from each `cv_projects/` phase
   into its `src/event_intelligence/` module. Each promotion = one focused
   commit (`refactor(detection): promote detection logic to src`). The
   experiments stay untouched as the historical record.

2. **Module map:**

   | Source experiment | → module |
   |---|---|
   | 01_video_inspection | `src/event_intelligence/video/` |
   | 02_person_detection | `src/event_intelligence/detection/` |
   | 03_person_tracking | `src/event_intelligence/tracking/` |
   | 04_trajectory_extraction | `src/event_intelligence/trajectories/` |
   | 05_zone_line_reasoning | `src/event_intelligence/spatial/` |
   | 06_dwell_time_events | `src/event_intelligence/events/` |
   | 07_visual_evidence | `src/event_intelligence/evidence/` |
   | 08_heatmaps_analytics | `src/event_intelligence/analytics/` |
   | 10_searchable_outputs | `src/event_intelligence/storage/` |

3. **One pipeline entry point:**

   ```bash
   uv run python -m event_intelligence.pipeline --video data/clip.mp4 --config configs/pipeline.yaml
   ```

   produces the complete artifact set:

   ```text
   outputs/runs/<run_id>/
     video_info.json
     detections.csv
     tracks.csv  track_summary.csv
     trajectories.csv
     zone_line_events.csv
     events.csv               (with evidence paths)
     snapshots/  crops/
     heatmap.png  analytics_summary.json
     zone_counts.csv  line_counts.csv
     annotated_video.mp4
     intelligence.db
     run_config.yaml          (frozen copy of the config used)
   ```

4. **Single pipeline config** (`configs/pipeline.yaml`) controlling every stage:
   model, confidence, tracker, scene config path, dwell thresholds, output toggles.

5. **Tests** (`tests/`): extend the Phase 6 suite with geometry
   (point-in-polygon, line-side), trajectory math, and CSV schema tests. Model
   inference itself is not unit-tested; it was evaluated in Phase 9.

6. **Integration verification:** run the full pipeline on one MOT17 sequence and
   one static-camera clip; outputs must match (within reason) what the
   individual experiments produced.

**Definition of Done:**

- [ ] One command produces the full artifact set on both test sources
- [ ] All modules promoted with explanation per AGENTS.md Rule 3
- [ ] `pytest` passes on event state machine + geometry + trajectory logic
- [ ] Frozen `run_config.yaml` saved per run (reproducibility)
- [ ] `cv_projects/` experiments left intact

---

# Track C — Phase 12: Productization (Explicitly Deferred)

Only after Phase 11's pipeline runs end-to-end on one command:

```text
simple review dashboard
FastAPI backend serving intelligence.db
React frontend for event review and search
job scheduling for batch video processing
authentication and access control
multi-camera processing and cross-camera identity
deployment, cloud infrastructure, retention policies
real-time RTSP processing
```

---

## Phase Flow Summary

```text
Phase 0   Foundation                           ✅ done
Phase 1   Video inspection + data acquisition  ← next  (MOT17 + VIRAT/fallback into data/)
Phase 2   Person detection                     (CUDA verified, YOLO11, versions pinned)
Phase 3   Person tracking                      (ByteTrack vs BoT-SORT + MOTA/IDF1 checkpoint)
Phase 4   Trajectory extraction                (bottom-center anchor)
Phase 5   Zone & line reasoning                (static footage, click-to-define zones)
Phase 6   Dwell-time events                    (lifecycle + hysteresis + unit tests)
Phase 7   Visual evidence                      (best-frame snapshots/crops)
Phase 8   Heatmaps & analytics
Phase 9   CV evaluation                        (full MOTA/IDF1 vs MOT17 ground truth)
Phase 10  Searchable outputs                   (SQLite + query CLI)
─────────────────────────────────────────────
Phase 11  CV PIPELINE INTEGRATION              (src/event_intelligence/)
─────────────────────────────────────────────
Phase 12  Productization                       (deferred)
```

---

## What Is Off-Limits Until Phase 12

React/any frontend · FastAPI/any HTTP API · auth · deployment · cloud infra ·
real-time RTSP · custom model training · multi-camera re-ID · face recognition ·
license plate recognition · complex database design beyond SQLite/DuckDB.

---

## Agent Rules (see AGENTS.md, non-negotiable)

1. Think first, code later — wait for explicit approval before any code
2. Small steps only
3. Explain every change
4. Ask before assuming
5. Never delete or overwrite silently
6. Smallest-part commit workflow
7. Conventional Commits: `<type>[scope]: <description>`

---

## Responsible Use

MOT17 and VIRAT contain footage of real people; both are research datasets with
usage terms — record license terms in `data/README.md` and do not redistribute
footage. The project makes no identity claims and includes no face recognition.
Future deployments must consider purpose limits, consent/legal basis, access
controls, audit logs, retention limits, and human review for consequential
decisions.

---

*Apache License 2.0*
