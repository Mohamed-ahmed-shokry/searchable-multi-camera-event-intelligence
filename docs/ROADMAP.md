# Development Roadmap — Living Progress Record

This file tracks executed development phases as one-time records (scope, acceptance
criteria, validation evidence, decisions). The canonical CV phase plan lives in
[project_phases.md](project_phases.md).

---

## Completed Phase: CI-0 — Verification & CI Foundation (2026-10-21)

### Objective

The repository had working tests (46 unittest cases) but **no automated CI pipeline,
no test-runner tooling, and no lint gates**. Nothing verified code quality on push.
This phase established a complete, continuously green verification pipeline.

Demonstrable end-to-end outcome: every push to GitHub triggers a CI workflow that
installs the locked environment, lints, format-checks, and runs the full test suite,
and the latest run on the phase branch is 100% green, with a draft PR documenting it.

### Scope

1. `pytest` as the test runner (dev dependency group) with unittest-compatible suite.
2. `pytest` configuration (`testpaths`) in `pyproject.toml`.
3. `ruff` as the lint + formatter gate (dev dependency group), applied repo-wide.
4. GitHub Actions workflow: `uv`-based sync from the committed `uv.lock`, ruff check,
   ruff format check, full pytest session.
5. Branch/PR delivery: phase branch, pushed incrementally, draft PR with phase summary.
6. Docs: README test/CI instructions, progress record, decision log.

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `pytest` runs the full existing suite in one command; all tests pass | ✅ | `uv run --locked pytest -q` → `46 passed, 49 subtests passed` (exit 0) |
| 2 | `ruff check` passes with the repo-managed strictness (default + line-length config) | ✅ | `uv run --locked ruff check .` → `All checks passed!` (exit 0) |
| 3 | `ruff format --check` passes on all tracked Python files | ✅ | `uv run --locked ruff format --check .` → `17 files already formatted` (exit 0) |
| 4 | GitHub Actions workflow executes `uv sync --locked`, ruff checks, and pytest on `windows-latest` (matching the dev OS) | ✅ | `.github/workflows/ci.yml` committed; remote run 39513973426 succeeded (29 s) |
| 5 | All local CI-parity checks and the remote CI workflow run pass 100% | ✅ | commands above + `gh run list`: `completed / success` |
| 6 | Draft PR open with phase summary tied to this roadmap | ✅ | PR #1 https://github.com/Mohamed-ahmed-shokry/searchable-multi-camera-event-intelligence/pull/1 |
| 7 | README + progress record + decision log updated (separate commits) | ✅ | README test-section commit; ROADMAP criteria + ticks commit; Decision Log addendum commit |

### Validation & CI Plan (exact commands, matching the workflow)

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
```

Remote: `gh run list --branch phase/ci-verification-foundation` → latest run green.
Local parity reference: baseline suite `python -m unittest discover -s tests` → 46 tests OK (Stage A).

### Exclusions (deferred, with reason)

- **Datasets / footage ingestion (Phase 1 DoD remainder):** requires multi-GB dataset
  downloads; not needed by this phase and time-flaky in this environment. Logged as
  next-phase candidate in `project_phases.md`.
- **Ubuntu/multi-OS CI matrix:** OpenCV `VideoWriter` codec behavior differs per OS and
  cannot be verified locally here; smallest-blast-radius choice is `windows-latest`
  (the actual dev OS). Extending the matrix is a follow-up after a Linux runner tackle.
- **Coverage thresholds / type checker:** premature for this codebase stage; each would
  be its own phase with justified thresholds, not boilerplate.

### Micro-Task Breakdown (execution record)

- [x] 1. `docs(roadmap): define CI-0 phase scope and micro-task checklist` (00f4012)
- [x] 2. `build(dev): add pytest runner to a dev dependency group` (eec6ffd)
- [x] 3. `build(dev): configure pytest testpaths in pyproject.toml` (8072f71)
- [x] 4. `test(runner): verify pytest executes the existing 46-test suite` ✅ 46 passed, 49 subtests
- [x] 5. `build(dev): add ruff to the dev dependency group` (4717f47)
- [x] 6. `build(lint): configure ruff line length and target version` — not needed: defaults (88 cols, py312 targets) already match the codebase; no config file added (see Decision Log addendum)
- [x] 7. `style: fix ruff findings per file (small atomic commits)` (18b8a4f, f177242, 7395743, dcbfbef + conftest order-independence fix 0ff2999)
- [x] 8. `style: apply ruff format repo-wide (one file-group per commit)` (e424add, 8c5216e, 020ccb9, 26e070f)
- [x] 9. `test(regression): confirm full pytest suite green after style commits` ✅ 46 passed, 49 subtests passed
- [x] 10. `ci: add GitHub Actions workflow (uv sync, ruff check, ruff format, pytest)` (e07a522)
- [x] 11. `ci: verify local parity of every workflow command` ✅ exit 0 for all
- [x] 12. `docs(readme): document pytest and ruff commands replacing unittest instructions`
- [x] 13. Push branch, create draft PR (PR #1)
- [x] 14. Monitor remote CI; fix any failure per its own micro-commit — run 39513973426 succeeded on first attempt; no fixes needed
- [x] 15. Final: tick acceptance criteria, decision log addendum, CI log (separate docs commits)

### Decision Log

| Decision | Options considered | Choice + repository evidence |
|----------|--------------------|------------------------------|
| Test runner | stdlib unittest (status quo) / pytest dev group | **pytest** as runner — existing tests are `unittest.TestCase`, so pytest executes them without rewriting tests; runner unification is reversible and low-risk. unittest stays the base style per existing code. |
| Linter/formatter | ruff / flake8+black / none | **ruff** — single tool, uv-native era standard, zero-config default behavior closest to repo's existing clean style; flake8+black is two tools and slower. |
| CI runner OS | windows-latest / ubuntu-latest / matrix | **windows-latest only** — dev machine is Windows (`win32`), `helpers.make_video` uses MJPG `VideoWriter` whose Linux behavior can't be verified locally; ubuntu can be added as a separate verified follow-up. |
| CI env install scope | default only / + detection extras / + notebooks | **default only** — tests only need numpy+opencv+pytest+ruff (Stage A evidence); CUDA extras ~2.5 GB on CI and untestable without GPU; README already treats extras as opt-in. |
| pytest rewrite of tests | keep unittest classes / rewrite to pytest style | **keep unittest classes** — pytest runs them natively; no churn, smallest diff. |
| Ruff config file | add `[tool.ruff]` section / use defaults | **use defaults** — `ruff check`+`--fix` and formatting with targets py312 / 88 cols produced zero spurious findings beyond real ones; no config file is simpler and reversible. |
| CI trigger scope | main only / all branches + PR | **all branches + PR** — phase branches need CI feedback before merging to `main`; matches the continuous-push workflow. |
| Roadmap file | extend project_phases.md / new docs/ROADMAP.md | **new `docs/ROADMAP.md`** — phase-brief records stay separate from the canonical locked CV DoD plan; README already links to `project_phases.md` for phases. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| `uv run python -m unittest discover -s tests -v` (Stage A baseline) | ✅ 46 tests OK | pre-change baseline |
| `uv run pytest -q` (first pytest run) | ✅ 46 passed, 49 subtests passed | before any style changes |
| `uv run ruff check .` (Stage A) | ❌ 14 errors (7×SIM117, 3×I001, 2×UP024, 1×UP037, 1×PYI034) | fixed per-file in micro-commits |
| `uv run ruff format --check .` (Stage A) | ❌ 4 files would reformat | fixed per-file in micro-commits |
| `uv sync --locked` (parity) | ✅ exit 0 | — |
| `uv run --locked ruff check .` (parity) | ✅ `All checks passed!` exit 0 | — |
| `uv run --locked ruff format --check .` (parity) | ✅ `17 files already formatted` exit 0 | — |
| `uv run --locked pytest -q` (parity) | ✅ `46 passed, 49 subtests passed` exit 0 | — |
| Remote CI: `gh run list --branch phase/ci-verification-foundation` | ✅ run 39513973426 — completed / success (29 s) | workflow `ci`, HEAD e07a522 |
| Remote CI (PR event): run 39514225001 | ✅ completed / success (29 s) | on PR #1 |
| Remote CI (final docs push): run 39514219892 | ✅ completed / success (27 s) | HEAD f784699 |
| E2E real execution | ✅ | `run_inspection` on a synthetic MJPEG clip + MOT-style sequence → `run.json` complete, 2 metadata entries, 10 sampled frames, 2 contact sheets; CLI `--help` verified |
| Merged | ✅ | PR #1 merged into `main` by the repository owner (merge commit 7b6f379); CI green on `main` (run 39518138728) |

---

## Completed Phase: 1b — Data Acquisition & Inspection Validation (2026-10-22)

### Objective

Phase 1's reader, CLI, and CI are done, but its Definition of Done still requires **real
footage** with provenance, validation against known dataset specs, and reviewable
contact sheets. This phase acquires: (a) 2–3 MOT17 training sequences and a static-camera
fallback clip set, (b) a documented `data/README.md`, and (c) a reproducible
spec-validation script that checks `video_info.json` against the published MOT17 specs.

Demonstrable end-to-end outcome: `data/` contains exactly the documented footage;
one inspection run over all sources produces `video_info.json` whose every entry passes
the spec validator; notes record scene assessment.

### Scope

1. Download official `MOT17.zip` (5.86 GB, background curl) and extract 3 chosen
   training sequences (one FRCNN variant per base sequence): MOT17-04, MOT17-05, MOT17-11.
2. Download 3 static-camera fallback clips from `intel-iot-devkit/sample-videos`
   (CC-BY-4.0): people-detection, store-aisle-detection, worker-zone-detection.
3. Rewrite `data/README.md` with exact provenance: source URLs, file names/sizes,
   licenses (MOT17 CC BY-NC-SA 3.0, samples CC BY-4.0), choice rationale.
4. New `scripts/validate_dataset.py` in the Phase 1 subproject: loads a `video_info.json`
   and validates every MOT17 entry against published specs (fps, resolution, frame count)
   and sanity-checks every clip entry (nonzero dims/fps; plausible duration).
5. Run the inspection CLI over all 6 sources into a real outputs run; feed the produced
   `video_info.json` through the validator; record results in notes.
6. Scene-assessment notes per source (frame statistics as proxy for visual review).

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | ≥1 MOT17 sequence and ≥1 static-camera clip exist in `data/` with documented provenance and license terms | ☐ | — |
| 2 | `video_info.json` from a real inspection run validated against known MOT17 specs by a committed, tested validator | ☐ | — |
| 3 | Contact sheets generated for every source and basic scene sanity verified programmatically; visual review flagged for the human owner | ☐ | — |
| 4 | Reader keeps working for both video files and frame directories on real footage | ☐ | — |
| 5 | All local CI-parity checks and remote CI workflow runs pass 100% | ☐ | — |
| 6 | Draft PR open/updated with phase summary tied to this roadmap | ☐ | — |

### Validation & CI Plan

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
# phase-specific:
uv run python cv_projects/01_video_inspection/scripts/inspect_video.py data/... 
uv run python cv_projects/01_video_inspection/scripts/validate_dataset.py <run>/video_info.json
```

Remote: `gh run list --branch phase/1-data-acquisition` → latest run green on HEAD.

### Exclusions

- **Phase 2 detector runs (YOLO11 on this footage):** needs the `detection-cuda` sync and
  GPU verification; that is Phase 2's own phase against this now-acquired footage.
- **VIRAT specifically:** downloads remain large/flaky per `project_phases.md`; using the
  documented fallback (static surveillance-style CC-BY-4.0 clips) as the plan allows.
- **Contact-sheet subjective review:** I cannot see images; programmatically verify
  statistics (frame variety, brightness, nondegenerate content) and explicitly hand the
  subjective review to the human owner (noted in notes.md).

### Micro-Task Breakdown

- [ ] 1. `docs(roadmap): define Phase 1b data acquisition scope and micro-task checklist`
- [ ] 2. `build(data): download three CC-BY static camera fallback clips`
- [ ] 3. `docs(data): document provenance and licenses for acquired footage`
- [ ] 4. `feat(inspection): add dataset spec validation script skeleton`
- [ ] 5. `test(inspection): add tests for dataset spec validator`
- [ ] 6. `feat(inspection): implement MOT17 spec table and validation logic`
- [ ] 7. Extract MOT17 sequences once background download completes
- [ ] 8. `fix(inspection): ...` any validator/reader issues surfaced by real data
- [ ] 9. Run real inspection over all sources; validate; record notes
- [ ] 10. Tick acceptance criteria, decision log, CI log (docs commits); push; PR; monitor CI

### Decision Log

| Decision | Options considered | Choice + evidence |
|----------|--------------------|-------------------|
| MOT17 acquisition source | Official motchallenge.net zip (5.86 GB, 1.6 MB/s ≈ 60 min) / HF mirrors (per-file CDN, 1050 files ≈ slow per request but parallelizable) / only fallback footage (violates DoD) | **Official zip in background curl** — authoritative source, resume-capable (`-C -`), runs concurrently with code work; verified 200/Content-Length 5860214001 and measured throughput at Stage A. |
| Which MOT17 sequences | 04+05+11 vs 02+05+09 vs full 7 | **04, 05, 11 (FRCNN variant each)** — published spec table gives distinct scene types (dense night street / crossing conflicts with camera moves / simple elevated street); FRCNN is the reference detector variant; roadmap asks 2–3 sequences only. |
| Fallback footage source | VIRAT (large/flaky per plan) / intel-iot-devkit sample-videos / own recording (impossible) | **intel-iot-devkit/sample-videos** — CC-BY-4.0 verified via GitHub API, small MP4s (3–12 MB), fixed-camera surveillance-style scenes with people, matches the plan's fallback clause. |
| Validator placement | inline CLI flag / separate script | **separate `validate_dataset.py` script** — mirrors the subproject pattern (scripts for reproducible runs), testable without media files (spec table is pure data), keeps inspect CLI focused. |
| Visual contact-sheet review | claim manual review / defer to human | **defer subjective review to owner** — honesty over polish; programmatic statistics + explicit flag in notes.md. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| Stage A baseline on `main` (post-merge) | ✅ lint/format/pytest exit 0 | merge CI run 39518138728 green |
| (phase work) | ☐ | filled as executed |

---

## Completed Phase: 2 — Person Detection (2026-10-24)

### Objective

Detect people and produce annotated visual proof with CUDA acceleration verified on the
development hardware (RTX 3060 Mobile / Laptop, 6 GB VRAM). Establish reproducible
person-only detection on video files and MOT17 frame sequences with schema-compliant
CSV export and annotated video rendering.

Demonstrable end-to-end outcome: cv_projects/02_person_detection/scripts/detect_people.py
executes against real video files and image sequences, produces schema-compliant
detections.csv (where every video frame is accounted for and non-person classes are
filtered out), renders aannotated_video.mp4 with bounding boxes and confidence scores,
and logs speed-accuracy benchmark comparison across YOLO11 models (n/s/m).

### Scope

1. CUDA environment verification on hardware (device name, VRAM, PyTorch CUDA build).
2. Detection configuration (configs/detection.yaml) defining model weights, confidence
   threshold (0.4), device selection, and class filtering (person only).
3. Core detection module and CLI (cv_projects/02_person_detection/scripts/detect_people.py):
   - Reads input using Phase 1's VideoSource (supports both video files and frame sequences).
   - Runs pretrained YOLO11 model with device fallback (cuda:0 / cpu).
   - Filters detections strictly to person class (COCO class 0).
   - Applies confidence filtering.
   - Formats detections into detections.csv schema (frame_id, x1, y1, x2, y2, confidence, class)
     ensuring every frame index 0 .. N-1 is present in the output.
   - Annotates frames with bounding boxes and confidence scores and renders aannotated_video.mp4.
   - Optionally dumps sample annotated frames into annotated_frames/.
4. Model comparison script (cv_projects/02_person_detection/scripts/compare_models.py)
   benchmarking yolo11n, yolo11s, and yolo11m on FPS and person detection counts.
5. Exploratory Jupyter notebook notebooks/01_compare_models.ipynb.
6. Unit and integration tests (tests/test_detect_people.py) covering schema conformity,
   person filtering, empty frame accounting, coordinate normalization, and CLI operation.
7. Documentation and notes (cv_projects/02_person_detection/notes.md, README.md).

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | CUDA verified: GPU model and VRAM recorded in notes | ✅ | RTX 3060 Laptop GPU (6.44 GB VRAM), CUDA 12.8, torch 2.11.0+cu128 recorded in notes.md |
| 2 | Model chosen: yolo11m baseline compared against n/s on real footage (speed FPS + detection trade-off recorded) | ✅ | Benchmark on real footage: n (22.84 FPS, 52 dets), s (25.89 FPS, 56 dets), m (22.40 FPS, 58 dets); yolo11m selected for highest recall within VRAM budget |
| 3 | Person-only filtering verified (zero non-person classes in detections.csv) | ✅ | Verified programmatically on full run: 100% person detections (418 instances), zero non-person classes |
| 4 | Detections CSV schema matches frame_id, x1, y1, x2, y2, confidence, class | ✅ | Schema verified in unit tests (test_detect_people.py) and validated on output detections.csv |
| 5 | Output detections.csv accounts for every input frame (exact frame count match) | ✅ | Exactly 596 distinct frame IDs (0..595) for 596 frames in people-detection.mp4; empty frames emit empty row |
| 6 | Annotated video rendered with bounding boxes and confidence scores and confirmed valid | ✅ | aannotated_video.mp4 verified via OpenCV VideoCapture: 768x432, 12 fps, 596 readable frames |
| 7 | Dependencies pinned in pyproject.toml; model weights recorded in configs/detection.yaml | ✅ | torch 2.11.0, torchvision 0.26.0, ultralytics 8.4.132 in pyproject.toml; yolo11m.pt in configs/detection.yaml |
| 8 | Comprehensive tests pass and CI parity checks pass 100% | ✅ | 79 passed, 56 subtests in pytest; ruff check and format check 100% clean |
| 9 | Draft PR open/updated with phase summary tied to this roadmap | ✅ | PR #3: https://github.com/Mohamed-ahmed-shokry/searchable-multi-camera-event-intelligence/pull/3 |

### Validation & CI Plan

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -v
# phase-specific execution:
uv run python cv_projects/02_person_detection/scripts/detect_people.py data/fallback_clips/people-detection.mp4
uv run python cv_projects/02_person_detection/scripts/compare_models.py
```

Remote: `gh run list --branch phase/2-person-detection` → latest run green on HEAD.

### Exclusions

- Multi-object tracking (ByteTrack / BoT-SORT): Track IDs and temporal trajectories
  belong to Phase 3 (Person Tracking) and Phase 4 (Trajectory Extraction).
- Custom model fine-tuning: Pretrained COCO weights are sufficient and locked per
  architecture decisions in docs/project_phases.md.
- Large cloud downloads in CI: Remote CI runner uses locked CPU environment without
  downloading YOLO weights; unit tests use mock/deterministic fixtures. Real weight
  inference runs locally on the GPU and is verified on test footage.

### Micro-Task Breakdown

- [x] 1. docs(roadmap): define Phase 2 person detection scope and micro-task checklist (60bd511)
- [x] 2. feat(detection): define default detection configuration in YAML (a13fe87)
- [x] 3. types(detection): implement Detection and FrameDetections schema with CSV serialization (ec56aff)
- [x] 4. test(detection): add unit tests for Detection schemas and CSV export (a3a889f)
- [x] 5. feat(detection): implement PersonDetector with person class filtering and CUDA auto-device (b11875f)
- [x] 6. feat(detection): implement frame bounding box annotation and video renderer (3e5f7c3)
- [x] 7. feat(detection): implement detect_people CLI with VideoSource input integration (636c2fa)
- [x] 8. test(detection): add tests for PersonDetector, CLI arguments, and empty frame accounting (1c13450)
- [x] 9. feat(detection): implement compare_models benchmark script for yolo11 n/s/m (b84306d)
- [x] 10. feat(detection): add exploratory model comparison notebook (815a07b)
- [x] 11. docs(detection): record CUDA specs and benchmark observations in notes.md (1f85800)
- [x] 12. Run detection and benchmark on real test footage; generate outputs (verified on people-detection.mp4)
- [x] 13. docs(readme): update README and subproject README with detection usage (901b71a, e3fc30f)
- [x] 14. Push to remote, monitor CI, create draft PR, and verify 100% green status

### Decision Log

| Decision | Options considered | Choice + evidence |
|----------|--------------------|-------------------|
| Empty frame accounting in CSV | Skip frames with 0 detections / Emit row with empty bbox fields | Emit row with empty bbox fields — frame_id,,,,,, ensures len(set(df['frame_id'])) == frame_count and every frame is accounted for without ambiguous gaps. |
| Person class ID | Dynamically lookup names['person'] / hardcode 0 | Lookup names mapping with fallback to 0 — standard COCO class 0 is person, but dynamic lookup guarantees compatibility with any custom weights dictionary. |
| Test suite isolation in CI | Download real weights in CI / Mock YOLO results in unit tests | Mock results in unit tests — CI runner (windows-latest) runs uv sync --locked on the default dependency set; tests verify detector logic, frame accounting, CSV serialization, and CLI without network flakiness. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| uv run --locked ruff check . | ✅ All checks passed | repo-wide lint gate clean |
| uv run --locked ruff format --check . | ✅ 22 files already formatted | code style gate clean |
| uv run --locked pytest -v | ✅ 79 passed, 56 subtests | unit + integration test suite passing |
| Real benchmark: compare_models.py | ✅ yolo11 n/s/m evaluated | n: 22.84 FPS, s: 25.89 FPS, m: 22.40 FPS on RTX 3060 Laptop GPU |
| Real detection: detect_people.py | ✅ 596 frames in 32.4s (18.37 FPS) | 418 detections, 100% person class, 596/596 frames accounted for in CSV |
| Video verification: annotated_video.mp4 | ✅ 768x432, 12 fps, 596 frames | all frames valid and readable |
| Remote CI (Push 1): run 40046200238 | ✅ completed / success (26s) | HEAD a13fe87 |
| Remote CI (PR #3 sync): run 40046211505 | ✅ completed / success (45s) | PR #3 |
| Remote CI (Final Phase 2): run 40047762273 | ✅ completed / success (48s) | HEAD 4f38645 |

---

## Current Phase: 3 — Person Tracking & MOT17 Validation (2026-10-24)

### Objective

Complete the prerequisite MOT17-11-FRCNN dataset extraction and Phase 1/2 validation, then
deliver **Phase 3 — Person Tracking (with early quantitative checkpoint)** end to end:
keep consistent `track_id` values across frames using ByteTrack and BoT-SORT, produce
schema-compliant `tracks.csv` and `track_summary.csv` outputs, render
`annotated_tracking_video.mp4` with distinct per-track colors and labels, detect track
fragmentation and ID-switch anomalies, evaluate MOTA/IDF1/ID-switches against MOT17
ground truth (`checkpoint_metrics.json`), and document the chosen tracker.

### Scope

1. Fix the thread-safety race condition in `cv_projects/01_video_inspection/scripts/fetch_mot17_sequence.py`
   so concurrent workers fetch each entry's local ZIP header and compressed payload in a
   single ranged HTTP request without mutating shared `ZipFile` state.
2. Complete extraction of `data/mot17/MOT17-11-FRCNN` (all 900 frames + `gt/gt.txt` +
   `seqinfo.ini`), validate all dataset sources with `inspect_video.py` and
   `validate_dataset.py`, and run Phase 2 detection/benchmark validation on `MOT17-11-FRCNN`.
3. Scaffold `cv_projects/03_person_tracking/` (`README.md`, `notes.md`, `configs/tracking.yaml`,
   `scripts/`, `notebooks/`, `outputs/`).
4. Implement `cv_projects/03_person_tracking/scripts/track_people.py`:
   - `TrackDetection`, `FrameTracks`, and `TrackSummary` dataclasses.
   - CSV serialization/deserialization for `tracks.csv` (`frame_id, track_id, x1, y1, x2, y2, confidence`)
     and `track_summary.csv` (`track_id, start_frame, end_frame, duration_frames, duration_seconds, observed_frames`).
   - Track fragmentation / gap and close-proximity ID-switch heuristic analyzer (`analyze_track_anomalies`).
   - `PersonTracker` wrapper around Ultralytics `model.track(persist=True, tracker=...)` supporting
     `bytetrack.yaml` and `botsort.yaml` with person-class filtering and deterministic per-track colors.
   - Annotated tracking video renderer (`annotated_tracking_video.mp4`) and CLI entrypoint.
5. Implement `cv_projects/03_person_tracking/scripts/evaluate_mot.py`:
   - MOT17 `gt.txt` parser (pedestrian class 1, active mark 1, visibility > 0, 1-based to 0-based frame conversion).
   - Pairwise IoU matrix and greedy/optimal bipartite assignment for CLEAR-MOT (`MOTA`, `MOTP`, `IDSW`) and `IDF1`.
6. Implement `cv_projects/03_person_tracking/scripts/compare_trackers.py`:
   - Side-by-side ByteTrack vs BoT-SORT comparison on real sequences + MOT17 ground truth,
     exporting `checkpoint_metrics.json`.
7. Add exploratory notebook `cv_projects/03_person_tracking/notebooks/01_compare_trackers.ipynb`.
8. Add comprehensive unit and integration tests (`tests/test_fetch_mot17_sequence.py`,
   `tests/test_track_people.py`) and verify 100% green local and remote CI.

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `fetch_mot17_sequence.py` thread-safety bug fixed and tested; `MOT17-11-FRCNN` (900 frames) extracted and validated via `validate_dataset.py` | ☐ | — |
| 2 | ByteTrack and BoT-SORT compared side-by-side on the same sequences | ☐ | — |
| 3 | Early quantitative checkpoint (`MOTA`, `IDF1`, ID switches) computed on MOT17 ground truth and saved to `checkpoint_metrics.json` | ☐ | — |
| 4 | Tracker chosen and configured in `cv_projects/03_person_tracking/configs/tracking.yaml` | ☐ | — |
| 5 | `tracks.csv` matches schema (`frame_id, track_id, x1, y1, x2, y2, confidence`) and `track_summary.csv` matches (`track_id, start_frame, end_frame, duration_frames, duration_seconds, observed_frames`) | ☐ | — |
| 6 | Annotated tracking video (`annotated_tracking_video.mp4`) rendered with distinct colors per `track_id` and verified readable | ☐ | — |
| 7 | At least 3 concrete ID-switch / fragmentation observations with frame numbers logged in `notes.md` | ☐ | — |
| 8 | All local CI-parity checks (`ruff check`, `ruff format --check`, `pytest`) and remote GitHub Actions CI pass 100% | ☐ | — |

### Micro-Task Breakdown

- [x] 1. `docs(roadmap): define Phase 3 person tracking scope and micro-task checklist`
- [ ] 2. `fix(inspection): make fetch_mot17_sequence thread-safe with single ranged GETs`
- [ ] 3. `test(inspection): add unit tests for fetch_mot17_sequence extraction and CRC resume`
- [ ] 4. `docs(inspection): record MOT17-11-FRCNN inspection and spec validation in notes`
- [ ] 5. `docs(detection): record MOT17-11-FRCNN benchmark and detection validation in notes`
- [ ] 6. `docs(roadmap): tick Phase 1b acceptance criteria with MOT17 validation evidence`
- [ ] 7. `docs(tracking): scaffold cv_projects/03_person_tracking README and notes`
- [ ] 8. `feat(tracking): add default tracking configuration in configs/tracking.yaml`
- [ ] 9. `test(helpers): expose Phase 3 tracking scripts directory to test suite`
- [ ] 10. `types(tracking): define TrackDetection and FrameTracks schemas in track_people.py`
- [ ] 11. `feat(tracking): implement tracks.csv writer and reader helpers`
- [ ] 12. `test(tracking): add unit tests for TrackDetection and tracks.csv serialization`
- [ ] 13. `types(tracking): define TrackSummary dataclass and compute_track_summaries`
- [ ] 14. `feat(tracking): implement track_summary.csv writer and reader helpers`
- [ ] 15. `test(tracking): add unit tests for track summary computation and CSV round-trip`
- [ ] 16. `feat(tracking): implement track fragmentation and ID-switch heuristic analyzer`
- [ ] 17. `test(tracking): add unit tests for track anomaly and gap detection`
- [ ] 18. `feat(tracking): implement PersonTracker wrapper for ByteTrack and BoT-SORT`
- [ ] 19. `test(tracking): add unit tests for PersonTracker class filtering and untracked box handling`
- [ ] 20. `feat(tracking): implement deterministic color palette and draw_tracks renderer`
- [ ] 21. `test(tracking): add unit tests for draw_tracks and track_color`
- [ ] 22. `feat(tracking): implement run_tracking pipeline and CLI in track_people.py`
- [ ] 23. `test(tracking): add integration tests for run_tracking and CLI error handling`
- [ ] 24. `feat(tracking): implement MOT17 ground-truth parser and IoU matrix in evaluate_mot.py`
- [ ] 25. `test(tracking): add unit tests for MOT17 ground-truth parser and IoU calculation`
- [ ] 26. `feat(tracking): implement CLEAR-MOT (MOTA/MOTP/IDSW) and IDF1 evaluator`
- [ ] 27. `test(tracking): add unit tests for MOTA, IDF1, and ID switch evaluation`
- [ ] 28. `feat(tracking): implement compare_trackers script for ByteTrack vs BoT-SORT checkpoint`
- [ ] 29. `test(tracking): add unit tests for compare_trackers checkpoint generation`
- [ ] 30. `feat(tracking): add exploratory notebook 01_compare_trackers.ipynb`
- [ ] 31. `docs(tracking): record ByteTrack vs BoT-SORT MOTA/IDF1 metrics and ID switch log in notes`
- [ ] 32. `docs(tracking): document tracking CLI and checkpoint evaluation in READMEs`
- [ ] 33. `docs(progress): complete Phase 3 acceptance criteria, decision log, and CI log`
