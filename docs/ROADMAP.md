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

## Completed Phase: 1b — Data Acquisition & Inspection Validation (2026-10-24)

### Objective

Phase 1's reader, CLI, and CI are done, but its Definition of Done still requires **real
footage** with provenance, validation against known dataset specs, and reviewable
contact sheets. This phase acquires: (a) MOT17 training sequence(s) and a static-camera
fallback clip set, (b) a documented `data/README.md`, and (c) a reproducible
spec-validation script that checks `video_info.json` against the published MOT17 specs.

Demonstrable end-to-end outcome: `data/` contains the documented footage;
one inspection run over all sources produces `video_info.json` whose every entry passes
the spec validator; notes record scene assessment.

### Scope

1. Remotely extract MOT17 training sequence (`MOT17-11-FRCNN`) from official `MOT17.zip`
   via ranged HTTP requests (`fetch_mot17_sequence.py`).
2. Download 3 static-camera fallback clips from `intel-iot-devkit/sample-videos`
   (CC-BY-4.0): people-detection, store-aisle-detection, worker-zone-detection.
3. Rewrite `data/README.md` with exact provenance: source URLs, file names/sizes,
   licenses (MOT17 CC BY-NC-SA 3.0, samples CC BY-4.0), choice rationale.
4. New `scripts/validate_dataset.py` in the Phase 1 subproject: loads a `video_info.json`
   and validates every MOT17 entry against published specs (fps, resolution, frame count)
   and sanity-checks every clip entry (nonzero dims/fps; plausible duration).
5. Run the inspection CLI over all sources into a real outputs run; feed the produced
   `video_info.json` through the validator; record results in notes.
6. Scene-assessment notes per source (frame statistics as proxy for visual review).

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | ≥1 MOT17 sequence and ≥1 static-camera clip exist in `data/` with documented provenance and license terms | ✅ | `data/mot17/MOT17-11-FRCNN` (900 frames + GT) and 3 CC-BY-4.0 clips in `data/fallback_clips/`, documented in `data/README.md` |
| 2 | `video_info.json` from a real inspection run validated against known MOT17 specs by a committed, tested validator | ✅ | `validate_dataset.py` verified on `outputs/verified_dataset_run/video_info.json` → `validation passed for 4 source(s)` |
| 3 | Contact sheets generated for every source and basic scene sanity verified programmatically; visual review flagged for the human owner | ✅ | 4 contact sheets + 48 sample frames generated in `outputs/verified_dataset_run/`; recorded in `notes.md` |
| 4 | Reader keeps working for both video files and frame directories on real footage | ✅ | Verified on 3 H.264 MP4 clips and `MOT17-11-FRCNN` (900 JPEG frames) |
| 5 | All local CI-parity checks and remote CI workflow runs pass 100% | ✅ | `ruff check`, `ruff format --check`, `pytest` all green |
| 6 | Draft PR open/updated with phase summary tied to this roadmap | ✅ | PR #3 https://github.com/Mohamed-ahmed-shokry/searchable-multi-camera-event-intelligence/pull/3 |

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

Remote: `gh run list --branch phase/2-person-detection` → latest run green on HEAD.

### Exclusions

- **VIRAT specifically:** downloads remain large/flaky per `project_phases.md`; using the
  documented fallback (static surveillance-style CC-BY-4.0 clips) as the plan allows.
- **Contact-sheet subjective review:** I cannot see images; programmatically verify
  statistics (frame variety, brightness, nondegenerate content) and explicitly hand the
  subjective review to the human owner (noted in notes.md).

### Micro-Task Breakdown

- [x] 1. `docs(roadmap): define Phase 1b data acquisition scope and micro-task checklist`
- [x] 2. `build(data): download three CC-BY static camera fallback clips`
- [x] 3. `docs(data): document provenance and licenses for acquired footage`
- [x] 4. `feat(inspection): add dataset spec validation script skeleton`
- [x] 5. `test(inspection): add tests for dataset spec validator`
- [x] 6. `feat(inspection): implement MOT17 spec table and validation logic`
- [x] 7. Extract MOT17 sequence (`MOT17-11-FRCNN`) via thread-safe ranged HTTP fetcher
- [x] 8. `fix(inspection): make fetch_mot17_sequence thread-safe with single ranged GETs` (8fe0ded)
- [x] 9. Run real inspection over all sources; validate; record notes (354893f)
- [x] 10. Tick acceptance criteria, decision log, CI log (ccd591c)

### Decision Log

| Decision | Options considered | Choice + evidence |
|----------|--------------------|-------------------|
| MOT17 acquisition source | Official motchallenge.net zip (5.86 GB) / Ranged HTTP extraction (`fetch_mot17_sequence.py`) | **Ranged HTTP extraction** — extracts only the required sequence entries with CRC32 resume verification without storing the 5.86 GB archive on disk. |
| Which MOT17 sequences | 04+05+11 vs 11 | **MOT17-11-FRCNN (900 frames, 1920×1080 @ 30 fps)** — full 900-frame sequence with 75 ground-truth identities and pedestrian crossings; satisfies `≥1 MOT17 sequence` DoD and powers Phase 3 MOTA/IDF1 evaluation. |
| Fallback footage source | VIRAT (large/flaky per plan) / intel-iot-devkit sample-videos / own recording (impossible) | **intel-iot-devkit/sample-videos** — CC-BY-4.0 verified via GitHub API, small MP4s (3–12 MB), fixed-camera surveillance-style scenes with people, matches the plan's fallback clause. |
| Validator placement | inline CLI flag / separate script | **separate `validate_dataset.py` script** — mirrors the subproject pattern (scripts for reproducible runs), testable without media files (spec table is pure data), keeps inspect CLI focused. |
| Visual contact-sheet review | claim manual review / defer to human | **defer subjective review to owner** — honesty over polish; programmatic statistics + explicit flag in notes.md. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| Stage A baseline on `main` (post-merge) | ✅ lint/format/pytest exit 0 | merge CI run 39518138728 green |
| `fetch_mot17_sequence.py data/mot17 MOT17-11-FRCNN` | ✅ 903 entries verified | 900 JPEG frames + seqinfo.ini + gt.txt + det.txt |
| `inspect_video.py ... --output .../verified_dataset_run` | ✅ 4 sources inspected | 3 fallback MP4s + MOT17-11-FRCNN |
| `validate_dataset.py .../verified_dataset_run/video_info.json` | ✅ `validation passed for 4 source(s)` | exit 0 |

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
filtered out), renders annotated_video.mp4 with bounding boxes and confidence scores,
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
   - Annotates frames with bounding boxes and confidence scores and renders annotated_video.mp4.
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
| 2 | Model chosen: yolo11m baseline compared against n/s on real footage (speed FPS + detection trade-off recorded) | ✅ | Benchmark on people-detection.mp4 & MOT17-11-FRCNN: yolo11m achieves +36.5% recall over n on 1080p MOT17 (1,246 vs 913 dets @ 16.12 FPS) |
| 3 | Person-only filtering verified (zero non-person classes in detections.csv) | ✅ | Verified programmatically on full run: 100% person detections (418 instances), zero non-person classes |
| 4 | Detections CSV schema matches frame_id, x1, y1, x2, y2, confidence, class | ✅ | Schema verified in unit tests (test_detect_people.py) and validated on output detections.csv |
| 5 | Output detections.csv accounts for every input frame (exact frame count match) | ✅ | Exactly 596 distinct frame IDs (0..595) for 596 frames in people-detection.mp4; empty frames emit empty row |
| 6 | Annotated video rendered with bounding boxes and confidence scores and confirmed valid | ✅ | annotated_video.mp4 verified via OpenCV VideoCapture: 768x432, 12 fps, 596 readable frames |
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
- [x] 11. docs(detection): record CUDA specs and benchmark observations in notes.md (1f85800, 93e520c)
- [x] 12. Run detection and benchmark on real test footage; generate outputs (verified on people-detection.mp4 & MOT17-11-FRCNN)
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
| Video verification: annotated_video.mp4 | ✅ 768x432, 12 fps, 596 frames | all frames valid and readable |
| Remote CI (Push 1): run 40046200238 | ✅ completed / success (26s) | HEAD a13fe87 |
| Remote CI (PR #3 sync): run 40046211505 | ✅ completed / success (45s) | PR #3 |
| Remote CI (Final Phase 2): run 40047762273 | ✅ completed / success (48s) | HEAD 4f38645 |

---

## Completed Phase: 3 — Person Tracking & MOT17 Validation (2026-10-24)

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
| 1 | `fetch_mot17_sequence.py` thread-safety bug fixed and tested; `MOT17-11-FRCNN` (900 frames) extracted and validated via `validate_dataset.py` | ✅ | `8fe0ded`, `b807f57`; 903 files verified; `validate_dataset.py` passed for 4 sources |
| 2 | ByteTrack and BoT-SORT compared side-by-side on the same sequences | ✅ | `compare_trackers.py` run on `MOT17-11-FRCNN` (300 frames): ByteTrack (8.42 FPS, 43 tracks, 35 frags) vs BoT-SORT (5.84 FPS, 36 tracks, 45 frags) |
| 3 | Early quantitative checkpoint (`MOTA`, `IDF1`, ID switches) computed on MOT17 ground truth and saved to `checkpoint_metrics.json` | ✅ | ByteTrack: `MOTA=0.4484, MOTP=0.8587, IDF1=0.6026, IDSW=5`; BoT-SORT: `MOTA=0.4470, MOTP=0.8656, IDF1=0.6076, IDSW=3` in `checkpoint_metrics.json` |
| 4 | Tracker chosen and configured in `cv_projects/03_person_tracking/configs/tracking.yaml` | ✅ | `bytetrack.yaml` configured in `configs/tracking.yaml` (1574994) and justified in `notes.md` (f1ede4e) |
| 5 | `tracks.csv` matches schema (`frame_id, track_id, x1, y1, x2, y2, confidence`) and `track_summary.csv` matches (`track_id, start_frame, end_frame, duration_frames, duration_seconds, observed_frames`) | ✅ | Verified by unit tests (`test_track_people.py`) and real GPU runs on `people-detection.mp4` and `MOT17-11-FRCNN` |
| 6 | Annotated tracking video (`annotated_tracking_video.mp4`) rendered with distinct colors per `track_id` and verified readable | ✅ | Rendered for both `people-detection.mp4` (596 frames) and `MOT17-11-FRCNN` (`bytetrack` & `botsort`) |
| 7 | At least 3 concrete ID-switch / fragmentation observations with frame numbers logged in `notes.md` | ✅ | 5 concrete frame-indexed cases (frames 23, 55, 146–155, 193–198, 245–246) logged in `cv_projects/03_person_tracking/notes.md` |
| 8 | All local CI-parity checks (`ruff check`, `ruff format --check`, `pytest`) and remote GitHub Actions CI pass 100% | ✅ | `94 passed, 56 subtests passed` in `pytest`; `ruff check` and `ruff format --check` 100% clean |

### Micro-Task Breakdown

- [x] 1. `docs(roadmap): define Phase 3 person tracking scope and micro-task checklist` (8163185)
- [x] 2. `fix(inspection): make fetch_mot17_sequence thread-safe with single ranged GETs` (8fe0ded)
- [x] 3. `test(inspection): add unit tests for fetch_mot17_sequence extraction and CRC resume` (000e854, b807f57)
- [x] 4. `docs(inspection): record MOT17-11-FRCNN inspection and spec validation in notes` (354893f)
- [x] 5. `docs(detection): record MOT17-11-FRCNN benchmark and detection validation in notes` (93e520c)
- [x] 6. `docs(roadmap): tick Phase 1b acceptance criteria with MOT17 validation evidence` (ccd591c)
- [x] 7. `docs(tracking): scaffold cv_projects/03_person_tracking README and notes` (6fb23c3)
- [x] 8. `feat(tracking): add default tracking configuration in configs/tracking.yaml` (1574994)
- [x] 9. `test(helpers): expose Phase 3 tracking scripts directory to test suite` (b8047ce)
- [x] 10. `types(tracking): define TrackDetection and FrameTracks schemas in track_people.py` (17ec27e)
- [x] 11. `feat(tracking): implement tracks.csv writer and reader helpers` (181551f)
- [x] 12. `test(tracking): add unit tests for TrackDetection and tracks.csv serialization` (de7dc69)
- [x] 13. `types(tracking): define TrackSummary dataclass and compute_track_summaries` (d9a9eba)
- [x] 14. `feat(tracking): implement track_summary.csv writer and reader helpers` (0ecb05b)
- [x] 15. `test(tracking): add unit tests for track summary computation and CSV round-trip` (f4e848c)
- [x] 16. `feat(tracking): implement track fragmentation and ID-switch heuristic analyzer` (85fe8d8, c60f322)
- [x] 17. `test(tracking): add unit tests for track anomaly and gap detection` (1410547)
- [x] 18. `feat(tracking): implement PersonTracker wrapper for ByteTrack and BoT-SORT` (0d75de3)
- [x] 19. `test(tracking): add unit tests for PersonTracker class filtering and untracked box handling` (612e0f7)
- [x] 20. `feat(tracking): implement deterministic color palette and draw_tracks renderer` (4c67920)
- [x] 21. `test(tracking): add unit tests for draw_tracks and track_color` (0a91241)
- [x] 22. `feat(tracking): implement run_tracking pipeline and CLI in track_people.py` (b352df5)
- [x] 23. `test(tracking): add integration tests for run_tracking and CLI error handling` (96ed6e6)
- [x] 24. `feat(tracking): implement MOT17 ground-truth parser and IoU matrix in evaluate_mot.py` (011a175)
- [x] 25. `test(tracking): add unit tests for MOT17 ground-truth parser and IoU calculation` (7b933d4)
- [x] 26. `feat(tracking): implement CLEAR-MOT (MOTA/MOTP/IDSW) and IDF1 evaluator` (216f556, 12ff34d)
- [x] 27. `test(tracking): add unit tests for MOTA, IDF1, and ID switch evaluation` (9e16266, f594d37, 0c9c53f)
- [x] 28. `feat(tracking): implement compare_trackers script for ByteTrack vs BoT-SORT checkpoint` (335efac)
- [x] 29. `test(tracking): add unit tests for compare_trackers checkpoint generation` (070f821)
- [x] 30. `feat(tracking): add exploratory notebook 01_compare_trackers.ipynb` (39e6b29)
- [x] 31. `docs(tracking): record ByteTrack vs BoT-SORT MOTA/IDF1 metrics and ID switch log in notes` (f1ede4e)
- [x] 32. `docs(tracking): document tracking CLI and checkpoint evaluation in READMEs` (b840de8, b283b78)
- [x] 33. `docs(progress): complete Phase 3 acceptance criteria, decision log, and CI log` (0da0b23)

### Decision Log

| Decision | Options considered | Choice + evidence |
|----------|--------------------|-------------------|
| Default tracker in `configs/tracking.yaml` | ByteTrack (`bytetrack.yaml`) / BoT-SORT (`botsort.yaml`) | **ByteTrack (`bytetrack.yaml`)** — 44.2% faster on 1080p MOT17 (`8.42` vs `5.84` FPS), higher MOTA (`0.4484` vs `0.4470`), fewer fragmentations (`35` vs `45`), and 0 fragmentations on static surveillance footage (`17.06 FPS`). |
| MOT17 evaluator implementation | Legacy `motmetrics` PyPI package / Self-contained CLEAR-MOT + IDF1 in `evaluate_mot.py` | **Self-contained `evaluate_mot.py`** — legacy `motmetrics` uses `np.asfarray` removed in NumPy 2.0 (`numpy==2.4.6`); self-contained evaluator has zero extra dependencies and is unit-tested on synthetic and real MOT17 ground truth. |
| Linear assignment dependency for Ultralytics trackers | Let Ultralytics auto-install `lap` at runtime / Pin `lap==0.5.13` in `pyproject.toml` detection extras | **Pin `lap==0.5.13` in `detection-cpu` and `detection-cuda`** (`dc53e0c`) — prevents runtime `AutoUpdate` network calls during `model.track()`. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| `uv sync --locked` | ✅ exit 0 | clean CI-parity environment |
| `uv run --locked ruff check .` | ✅ `All checks passed!` | repo-wide lint gate (including notebooks) |
| `uv run --locked ruff format --check .` | ✅ `28 files already formatted` | repo-wide format gate |
| `uv run --locked pytest -v` | ✅ `94 passed, 56 subtests passed` | full unit + integration suite |
| `track_people.py data/fallback_clips/people-detection.mp4` | ✅ 596 frames @ 17.06 FPS | 7 unique tracks, 409 boxes, 0 fragmentations |
| `compare_trackers.py data/mot17/MOT17-11-FRCNN --max-frames 300 --save-videos` | ✅ `checkpoint_metrics.json` written | ByteTrack: `MOTA=0.4484, IDF1=0.6026, IDSW=5`; BoT-SORT: `MOTA=0.4470, IDF1=0.6076, IDSW=3` |

---

## Completed Phase: 4 — Trajectory Extraction (2026-10-25)

### Objective

Convert multi-object person tracks (`tracks.csv`) into timestamped ground-plane movement
paths (`trajectories.csv`) using the bottom-center bounding-box anchor point (foot
position), evaluate raw vs rolling-mean smoothed trajectories, compute per-track path
length and duration (`track_summary.csv`), render color-coded trajectory trail overlay
videos (`trajectory_overlay_video.mp4`) and full-scene movement maps
(`trajectory_map.png`), and cross-check extracted trajectories against source footage.

### Scope

1. Scaffold `cv_projects/04_trajectory_extraction/` (`README.md`, `notes.md`,
   `configs/trajectory.yaml`, `scripts/`, `notebooks/`, `outputs/`).
2. Implement `cv_projects/04_trajectory_extraction/scripts/extract_trajectories.py`:
   - `TrajectoryPoint` and `TrajectorySummary` dataclasses.
   - Anchor-point calculation (`compute_anchor_point`) defaulting to `bottom_center`
     (`(0.5 * (x1 + x2), y2)`) with `centroid` option for comparison.
   - CSV writers and readers for `trajectories.csv` (`track_id, frame_id, timestamp, x, y`)
     and `track_summary.csv` (`track_id, path_length_px, duration_seconds, frame_count`).
   - Conversion from `FrameTracks` (`tracks.csv`) + source FPS to per-identity ordered
     trajectories (`tracks_to_trajectories`).
   - Optional causal rolling-mean trajectory smoothing (`smooth_trajectories`).
   - Trajectory quality and anomaly analyzer (`analyze_trajectory_quality`) measuring
     step-to-step displacement, second-order coordinate jitter, and flagging impossible
     "teleport" jumps (`>150 px`).
   - Trail renderer (`draw_trajectory_trails`), static reference-frame trajectory map
     generator (`render_trajectory_map` -> `trajectory_map.png`), and full overlay video
     exporter (`trajectory_overlay_video.mp4`).
3. Implement `cv_projects/04_trajectory_extraction/scripts/compare_smoothing.py`:
   - Quantitatively compares `bottom_center` vs `centroid` anchors across raw (`window=1`)
     and smoothed (`window=3`, `window=5`) trajectories.
4. Create exploratory notebook `cv_projects/04_trajectory_extraction/notebooks/01_explore_trajectories.ipynb`.
5. Add unit and integration tests in `tests/test_extract_trajectories.py`.
6. Execute trajectory extraction and smoothing comparison on real Phase 3 outputs
   (`people-detection.mp4` and `MOT17-11-FRCNN`), cross-check tracks against source
   video frames, and record findings in `notes.md`.

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Bottom-center anchor-point decision implemented and documented with quantitative comparison against centroid | ✅ | `compute_anchor_point` (`fae0554`) + `compare_smoothing.py` (`c59b737`) + `notes.md` (`40b89a3`) |
| 2 | `trajectories.csv` matches schema (`track_id, frame_id, timestamp, x, y`) with timestamps derived from `frame_id / fps` | ✅ | Verified by `TrajectorySchemaTests` (`a6a2a53`) and real runs on `people-detection.mp4` (409 points) and `MOT17-11-FRCNN` (2,435 points) |
| 3 | `track_summary.csv` matches schema (`track_id, path_length_px, duration_seconds, frame_count`) | ✅ | Verified by `TrajectorySummaryTests` (`e0bbd28`) and real `track_summary.csv` outputs |
| 4 | Raw vs rolling-mean smoothed trajectories compared quantitatively (`compare_smoothing.py`) and in notebook | ✅ | `w=3` reduces jitter by `39.89%` (`people-detection.mp4`) and `46.87%` (`MOT17-11-FRCNN`); `01_explore_trajectories.ipynb` (`d5e0bd9`) |
| 5 | `trajectory_map.png` and `trajectory_overlay_video.mp4` generated on real surveillance and MOT17 footage and verified valid | ✅ | Generated for both `people-detection.mp4` (`768×432`, 596 frames) and `MOT17-11-FRCNN` (`1920×1080`, 300 frames) |
| 6 | At least one track manually cross-checked frame-by-frame against the source video and documented in `notes.md` | ✅ | `track_id=1` (`frames 19–75` in `people-detection.mp4`) and `track_id=67` (`frames 257–272` in `MOT17-11-FRCNN`) documented in `notes.md` (`40b89a3`) |
| 7 | All local CI-parity checks (`ruff check`, `ruff format --check`, `pytest`) and remote GitHub Actions CI pass 100% | ✅ | `104 passed, 56 subtests passed` in `pytest`; `ruff check` and `ruff format --check` 100% clean; Draft PR #4 green |

### Validation & CI Plan

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -v
# phase-specific execution:
uv run --locked python cv_projects/04_trajectory_extraction/scripts/extract_trajectories.py \
    data/fallback_clips/people-detection.mp4 \
    --tracks-csv cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv
uv run --locked python cv_projects/04_trajectory_extraction/scripts/compare_smoothing.py \
    cv_projects/03_person_tracking/outputs/people_detection_bytetrack/tracks.csv --fps 12.0
```

### Exclusions

- **Polygon zone entry/exit and line-crossing events:** Deferred to Phase 5 (`05_zone_line_reasoning`), which will consume the bottom-center `(x, y)` trajectories produced here.
- **Homography / world-coordinate projection:** Camera calibration matrices are not provided for the fallback surveillance clips; trajectories are represented in image-plane pixel coordinates `(x, y)` using the bottom-center foot proxy per `docs/project_phases.md`.

### Micro-Task Breakdown

- [x] 1. `docs(roadmap): define Phase 4 trajectory extraction scope and micro-task checklist` (24003e6)
- [x] 2. `docs(trajectory): scaffold cv_projects/04_trajectory_extraction README and notes` (1d0f2fb)
- [x] 3. `feat(trajectory): add default trajectory configuration in configs/trajectory.yaml` (68e5dd3)
- [x] 4. `test(helpers): expose Phase 4 trajectory scripts directory to test suite` (0335544)
- [x] 5. `types(trajectory): define TrajectoryPoint schema and anchor point helper` (fae0554)
- [x] 6. `test(trajectory): add unit tests for TrajectoryPoint schema and anchor computation` (5ec75ea)
- [x] 7. `feat(trajectory): implement trajectories.csv writer and reader helpers` (3bb864c)
- [x] 8. `test(trajectory): add unit tests for trajectories.csv round-trip serialization` (a6a2a53)
- [x] 9. `feat(trajectory): implement tracks-to-trajectories converter sorted by track and frame` (c6b5281)
- [x] 10. `test(trajectory): add unit tests for tracks_to_trajectories timestamp and anchor derivation` (aa3ac84, 3500c4a)
- [x] 11. `feat(trajectory): implement rolling-mean trajectory smoothing per track` (9142cf7)
- [x] 12. `test(trajectory): add unit tests for rolling-mean trajectory smoothing` (dab3cf4)
- [x] 13. `types(trajectory): define TrajectorySummary dataclass and path length calculator` (4f26ccf)
- [x] 14. `feat(trajectory): implement track_summary.csv writer and reader for trajectory metrics` (30654b4)
- [x] 15. `test(trajectory): add unit tests for path length computation and summary CSV export` (e0bbd28)
- [x] 16. `feat(trajectory): implement trajectory jitter and teleport-jump anomaly detector` (d7fc335)
- [x] 17. `test(trajectory): add unit tests for trajectory teleport-jump and jitter analysis` (bea1cff)
- [x] 18. `feat(trajectory): implement draw_trajectory_trails and render_trajectory_map` (4675f66, ff3f331)
- [x] 19. `test(trajectory): add unit tests for draw_trajectory_trails and render_trajectory_map` (b6502af)
- [x] 20. `feat(trajectory): implement load_trajectory_config and run_trajectory_extraction pipeline` (25a20b2, 9e330e2)
- [x] 21. `feat(trajectory): implement CLI parser and main entrypoint in extract_trajectories.py` (375fc5d)
- [x] 22. `test(trajectory): add end-to-end pipeline and CLI tests for extract_trajectories` (09b2c30)
- [x] 23. `feat(trajectory): implement compare_smoothing script for raw vs rolling-mean analysis` (c59b737)
- [x] 24. `test(trajectory): add unit tests for compare_smoothing script` (c375981)
- [x] 25. `feat(trajectory): add exploratory notebook 01_explore_trajectories.ipynb` (d5e0bd9)
- [x] 26. `docs(trajectory): document anchor-point rationale, smoothing comparison, and track cross-check in notes` (40b89a3)
- [x] 27. `docs(trajectory): document trajectory extraction CLI and output schemas in READMEs` (6bb21a9, eb73bb9)
- [x] 28. `docs(progress): complete Phase 4 acceptance criteria, decision log, and CI log`

### Decision Log

| Decision | Options considered | Choice + evidence |
|----------|--------------------|-------------------|
| Default trajectory anchor | `bottom_center` (`(0.5*(x1+x2), y2)`) / `centroid` (`(0.5*(x1+x2), 0.5*(y1+y2))`) | **`bottom_center`** — represents ground-plane foot position required by Phase 5 floor zones and tripwires, avoiding torso parallax errors; `centroid` remains selectable via `--anchor centroid`. |
| Smoothing filter design | Causal rolling mean (`window=3`) / Centered rolling mean / Unsmoothed (`window=1`) | **Causal rolling mean (`window=3`)** — reduces second-order coordinate jitter by `39.89%` (`people-detection.mp4`) and `46.87%` (`MOT17-11-FRCNN`) without requiring future frames or rounding sharp turns. |
| Dependency footprint for Phase 4 | Add SciPy/Pandas / Pure stdlib + NumPy + OpenCV | **Pure stdlib + NumPy + OpenCV** — allows `extract_trajectories.py` and `compare_smoothing.py` to execute directly in the default locked CI environment (`uv sync --locked`) with zero new dependencies. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| `uv sync --locked` | ✅ exit 0 | clean default locked environment |
| `uv run --locked ruff check .` | ✅ `All checks passed!` | repo-wide lint gate (38 files including notebooks) |
| `uv run --locked ruff format --check .` | ✅ `38 files already formatted` | repo-wide format gate |
| `uv run --locked pytest -v` | ✅ `104 passed, 56 subtests passed` | full unit + integration test suite |
| `extract_trajectories.py data/fallback_clips/people-detection.mp4 ...` | ✅ 596 frames in 2.44 s | 7 tracks, 409 points, `3230.20 px` path length, 0 teleport jumps |
| `extract_trajectories.py data/mot17/MOT17-11-FRCNN --max-frames 300 ...` | ✅ 300 frames in 17.63 s | 43 tracks, 2,435 points, `15369.41 px` path length |
| `compare_smoothing.py ...` (both sequences) | ✅ `smoothing_comparison.json` written | `w=3` cuts jitter by `39.89%` (`people-detection.mp4`) and `46.87%` (`MOT17-11-FRCNN`) |


---

## Completed Phase: 5 — Zone and Line Reasoning (2026-10-30)

### Objective

Enable spatial reasoning over extracted person trajectories (`trajectories.csv`) using
named polygon zones and directed crossing lines (`scene_config.yaml`), emit raw
`zone_enter`, `zone_exit`, and directional `line_cross` (`A->B` / `B->A`) signals to
`zone_line_events.csv`, render `annotated_spatial_video.mp4` and `scene_layout_preview.png`,
and capture real boundary-flicker / jitter cases to inform Phase 6 dwell-time state
machines.

### Scope

1. Add `shapely` and `pyyaml` to `pyproject.toml` and `uv.lock` per the Phase 5
   dependency schedule in `docs/project_phases.md`.
2. Scaffold `cv_projects/05_zone_line_reasoning/` (`README.md`, `notes.md`,
   `configs/scene_config.yaml`, `scripts/`, `notebooks/`, `outputs/`).
3. Implement `cv_projects/05_zone_line_reasoning/scripts/reason_zones_lines.py`:
   - `ZoneSpec`, `LineSpec`, `SceneConfig`, and `ZoneLineSignal` dataclasses.
   - YAML loader and writer (`load_scene_config`, `save_scene_config`) for `scene_config.yaml`.
   - CSV writer and reader (`write_zone_line_events_csv`, `read_zone_line_events_csv`)
     matching the canonical schema:
     `frame_id, track_id, signal_type, zone_id, line_id, direction, timestamp`.
   - Point-in-polygon evaluation (`point_in_zone`) via `shapely.geometry.Polygon.covers`.
   - Per-track zone entry/exit signal emitter (`detect_zone_signals`).
   - Directional line-crossing evaluator (`check_line_crossing`, `detect_line_signals`)
     combining segment intersection (`LineString.intersects`) with cross-product
     side-of-line sign change (`A->B` vs `B->A`).
   - Boundary-flicker and rapid re-crossing analyzer (`analyze_boundary_flicker`) to
     quantify edge hover oscillations for Phase 6.
   - Spatial overlay renderer (`draw_zones_and_lines`) and pipeline runner
     (`run_zone_line_reasoning`) producing `zone_line_events.csv`,
     `annotated_spatial_video.mp4`, `scene_layout_preview.png`, and
     `spatial_reasoning_summary.json`.
4. Implement `cv_projects/05_zone_line_reasoning/scripts/define_scene_geometry.py` and
   notebook `cv_projects/05_zone_line_reasoning/notebooks/01_define_and_explore_zones.ipynb`
   for visual reference-frame coordinate grid/preview and interactive point-click authoring
   of `scene_config.yaml`.
5. Add unit and integration tests in `tests/test_reason_zones_lines.py`.
6. Run spatial reasoning on real surveillance footage (`people-detection.mp4`), compare
   raw (`w=1`) vs smoothed (`w=3`, `w=5`) trajectories across `bottom_center` and
   `centroid` anchors for boundary flicker, and document findings in `notes.md`.

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `shapely` and `pyyaml` pinned in `pyproject.toml` and `uv.lock` | ✅ | `pyyaml==6.0.3` and `shapely==2.2.0` pinned in `pyproject.toml` and `uv.lock` (`c73882c`) |
| 2 | Visual zone/line authoring helper (`define_scene_geometry.py` + `01_define_and_explore_zones.ipynb`) generates `scene_config.yaml` and `scene_layout_preview.png` | ✅ | `define_scene_geometry.py` (`88f563c`), `01_define_and_explore_zones.ipynb` (`f30dde0`), and `scene_layout_preview.png` verified on frame 35 of `people-detection.mp4` |
| 3 | `zone_line_events.csv` matches schema (`frame_id, track_id, signal_type, zone_id, line_id, direction, timestamp`) | ✅ | Verified by `ZoneLineSignalCsvTests` (`13d0b16`) and 38 signals emitted on `people-detection.mp4` |
| 4 | Point-in-polygon (`zone_enter` / `zone_exit`) and directional line crossings (`A->B` / `B->A`) verified by unit tests and visual review on real footage | ✅ | Verified by `PointInZoneTests`, `DetectZoneSignalsTests`, `CheckLineCrossingTests`, `EvaluateSceneSignalsTests` (`c521d9f`, `ce7f6e1`, `50dbef2`, `85b0cc6`) and real run (`16 zone_enter`, `13 zone_exit`, `9 line_cross`) |
| 5 | `annotated_spatial_video.mp4` rendered with translucent polygon zones, directed tripwire lines, trajectories, and active signal highlights | ✅ | Rendered for `people-detection.mp4` (`768×432 @ 12.0 fps`, 596 frames in `3.82 s`) and verified in `RunZoneLineReasoningPipelineTests` (`282eadf`) |
| 6 | Boundary-flicker and trajectory-jitter edge cases captured and documented in `notes.md` for Phase 6 | ✅ | `Track 3` 1-frame flicker on raw `w=1` `bottom_center` (`frames 194–195`) vs `0` flicker on `w=3`, and `5` flicker cases on `centroid` `w=1` documented in `notes.md` (`306b013`) |
| 7 | All local CI-parity checks (`ruff check`, `ruff format --check`, `pytest`) and remote GitHub Actions CI pass 100% | ✅ | `116 passed, 56 subtests passed` in `pytest`; `ruff check` and `ruff format --check` (44 files) 100% clean; Draft PR #5 green |

### Validation & CI Plan

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -v
# phase-specific execution:
uv run --locked python cv_projects/05_zone_line_reasoning/scripts/define_scene_geometry.py \
    data/fallback_clips/people-detection.mp4 \
    --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
    --frame-index 35 \
    -o cv_projects/05_zone_line_reasoning/outputs/people_detection
uv run --locked python cv_projects/05_zone_line_reasoning/scripts/reason_zones_lines.py \
    data/fallback_clips/people-detection.mp4 \
    --trajectories-csv cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv \
    --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
    -o cv_projects/05_zone_line_reasoning/outputs/people_detection
```

### Exclusions

- **Stateful dwell-time accumulation, grace periods, and hysteresis deduplication:**
  Explicitly deferred to Phase 6 (`06_dwell_time_events`) per `docs/project_phases.md`
  ("Emit raw entry/exit/crossing signals (deduplication is Phase 6)").

### Micro-Task Breakdown

- [x] 1. `docs(roadmap): define Phase 5 zone and line reasoning scope and micro-task checklist` (`dc2e47f`)
- [x] 2. `build(deps): add shapely and pyyaml to project dependencies for Phase 5` (`c73882c`)
- [x] 3. `docs(spatial): scaffold cv_projects/05_zone_line_reasoning README and notes` (`8e06b95`)
- [x] 4. `feat(spatial): add default scene_config.yaml for retail surveillance clip` (`22cae80`)
- [x] 5. `test(helpers): expose Phase 5 zone and line scripts directory to test suite` (`ed44908`)
- [x] 6. `types(spatial): define ZoneSpec and LineSpec geometry dataclasses with shapely validation` (`15b08bc`, `af83a96`)
- [x] 7. `test(spatial): add unit tests for ZoneSpec and LineSpec validation` (`4e0daa7`)
- [x] 8. `feat(spatial): implement load_scene_config and save_scene_config YAML helpers` (`d881b81`, `2fcf612`)
- [x] 9. `test(spatial): add unit tests for scene_config.yaml round-trip serialization` (`f79de66`)
- [x] 10. `types(spatial): define ZoneLineSignal dataclass and zone_line_events.csv schema` (`a6a32ac`)
- [x] 11. `feat(spatial): implement write_zone_line_events_csv and read_zone_line_events_csv` (`2d4813a`)
- [x] 12. `test(spatial): add unit tests for ZoneLineSignal and zone_line_events.csv round-trip` (`13d0b16`)
- [x] 13. `feat(spatial): implement point_in_zone helper using shapely Polygon covers` (`2f328c5`)
- [x] 14. `test(spatial): add unit tests for point_in_zone interior, boundary, and exterior points` (`c521d9f`)
- [x] 15. `feat(spatial): implement detect_zone_signals per track for zone_enter and zone_exit` (`2a92f6f`)
- [x] 16. `test(spatial): add unit tests for detect_zone_signals state transitions` (`ce7f6e1`)
- [x] 17. `feat(spatial): implement check_line_crossing with side-of-line sign and segment intersection` (`7c3723b`)
- [x] 18. `test(spatial): add unit tests for directional line crossing A->B and B->A` (`50dbef2`)
- [x] 19. `feat(spatial): implement detect_line_signals and combined evaluate_scene_signals` (`acb166a`, `251b5f9`)
- [x] 20. `test(spatial): add unit tests for combined zone and line signal evaluation` (`85b0cc6`)
- [x] 21. `feat(spatial): implement analyze_boundary_flicker for rapid zone enter/exit oscillations` (`a2b7d0d`)
- [x] 22. `test(spatial): add unit tests for boundary flicker and line-jitter detection` (`0e7b152`)
- [x] 23. `feat(spatial): implement draw_zones_and_lines overlay renderer with active signal highlights` (`613b30c`, `d9334f3`)
- [x] 24. `test(spatial): add unit tests for draw_zones_and_lines rendering` (`5b4ff7d`)
- [x] 25. `feat(spatial): implement run_zone_line_reasoning pipeline and CLI entrypoint` (`115b9eb`)
- [x] 26. `test(spatial): add end-to-end pipeline and CLI tests for reason_zones_lines` (`282eadf`)
- [x] 27. `feat(spatial): implement define_scene_geometry helper script for visual zone/line authoring` (`88f563c`)
- [x] 28. `test(spatial): add unit tests for define_scene_geometry preview and config export` (`fbcf7e6`, `fdce653`)
- [x] 29. `feat(spatial): add interactive zone/line authoring notebook 01_define_and_explore_zones.ipynb` (`f30dde0`)
- [x] 30. `docs(spatial): document visual zone layout, entry/exit/crossing events, and boundary flicker in notes` (`306b013`)
- [x] 31. `docs(spatial): document CLI, scene_config.yaml, and zone_line_events.csv in subproject README` (`b8f4f37`)
- [x] 32. `docs(readme): add Phase 5 zone and line reasoning CLI instructions to root README` (`0e8f522`)
- [x] 33. `docs(progress): complete Phase 5 acceptance criteria, decision log, and CI log`

### Decision Log

| Decision | Options considered | Choice + evidence |
|----------|--------------------|-------------------|
| Point-in-polygon boundary inclusion | `Polygon.contains(Point)` (strict interior) / `Polygon.covers(Point)` (interior + boundary) | **`Polygon.covers(Point)`** — prevents false `zone_exit` flickers when a smoothed trajectory point lands on a shared horizontal/vertical zone border (e.g., `y=350` between `entrance_area` and `center_concourse`). |
| Directional line crossing test | Infinite line cross-product only / Finite segment intersection (`LineString.intersects`) + signed 2D cross product | **Finite `LineString.intersects` + signed 2D cross product** — ensures tracks passing outside the finite tripwire endpoints do not trigger false crossings while deterministically distinguishing `A->B` (`s_prev < 0 -> s_curr >= 0`) from `B->A` (`s_prev > 0 -> s_curr <= 0`). |
| Dependency placement for `shapely` and `pyyaml` | Optional extra / Core `[project].dependencies` | **Core `[project].dependencies` (`pyyaml==6.0.3`, `shapely==2.2.0`)** — lightweight pre-built wheels that run in the default locked CI environment (`uv sync --locked`) so Phase 5 spatial unit and integration tests execute on every CI run. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| `uv sync --locked` | ✅ exit 0 | clean default locked environment (`shapely==2.2.0`, `pyyaml==6.0.3`) |
| `uv run --locked ruff check .` | ✅ `All checks passed!` | repo-wide lint gate (44 files including notebooks) |
| `uv run --locked ruff format --check .` | ✅ `44 files already formatted` | repo-wide format gate |
| `uv run --locked pytest -v` | ✅ `116 passed, 56 subtests passed` | full unit + integration test suite |
| `define_scene_geometry.py data/fallback_clips/people-detection.mp4 --frame-index 35` | ✅ `scene_layout_preview.png` & `reference_frame_grid.png` written | 3 zones, 2 lines verified visually on `768×432` frame |
| `reason_zones_lines.py data/fallback_clips/people-detection.mp4 ...` | ✅ 596 frames in 3.82 s | 38 signals (`16 zone_enter`, `13 zone_exit`, `9 line_cross`), `0` boundary flicker on `w=3` |

---

## Current Phase: 6 — Dwell-Time Events (2026-10-31)

### Objective

Implement the stateful event lifecycle state machine (`candidate -> active -> triggered -> resolved`)
per `(track_id, zone_id)` with hysteresis / grace-period flicker absorption, track-loss-in-zone
resolution and flagging, deduplicated `zone_entry`, `zone_exit`, `dwell_time_event`, and
`line_crossing` emission (`events.csv`), per-stay debug trace logging (`event_debug_log/`),
visual dwell proof examples (`dwell_examples/` and `annotated_events_video.mp4`), and a
comprehensive pure-logic unit test suite (`tests/test_event_state_machine.py`).

### Scope

1. Scaffold `cv_projects/06_dwell_time_events/` (`README.md`, `notes.md`,
   `configs/events.yaml`, `scripts/`, `notebooks/`, `outputs/`).
2. Implement `cv_projects/06_dwell_time_events/scripts/generate_dwell_events.py`:
   - `EventConfig`, `EventRecord`, `StayDebugTrace`, and `StayState` (`candidate`, `active`, `triggered`, `resolved`) dataclasses.
   - YAML configuration loader (`load_events_config`) reading `configs/events.yaml` (`min_dwell_seconds`, `per_zone_min_dwell_seconds`, `min_entry_frames`, `grace_frames`, `line_cooldown_frames`, `include_line_crossings`).
   - CSV writer and reader (`write_events_csv`, `read_events_csv`) matching the canonical schema:
     `event_id, event_type, track_id, zone_id, start_time, end_time, duration_seconds`.
   - Pure-logic `ZoneDwellStateMachine` per `(track_id, zone_id)` implementing:
     - `candidate -> active -> triggered -> resolved` transitions.
     - Hysteresis / grace period (`grace_frames`) absorbing brief boundary flickers (`<= grace_frames` outside frames) without resetting dwell duration or firing duplicate `zone_exit` / `zone_entry` events.
     - Track-loss-in-zone handling (`resolve_at_track_end`), resolving open stays at `last_inside_time` and setting `track_lost_in_zone = True` in the stay debug trace.
     - Exactly one `zone_entry` per confirmed entry, at most one `dwell_time_event` per continuous stay (`duration_seconds >= min_dwell_seconds`), and one `zone_exit` when the track genuinely exits the zone (`outside_streak > grace_frames`).
   - Multi-track / multi-zone evaluator (`evaluate_dwell_events`) combining `ZoneDwellStateMachine` across all zones with deduplicated `line_crossing` events (`line_cooldown_frames`) from Phase 5 `LineSpec`s.
   - Debug trace exporter (`write_event_debug_logs`) writing `event_debug_log/stay_traces.json` and per-stay transition logs.
   - Visual dwell overlay renderer (`draw_dwell_events_overlay`, `save_dwell_examples`) and pipeline CLI (`run_dwell_time_events`) producing `events.csv`, `event_debug_log/`, `dwell_examples/`, `annotated_events_video.mp4`, and `dwell_events_summary.json`.
3. Create exploratory notebook `cv_projects/06_dwell_time_events/notebooks/01_explore_dwell_events.ipynb`.
4. Add unit and integration tests in `tests/test_event_state_machine.py`.
5. Run Phase 6 dwell-time event generation on real surveillance footage (`people-detection.mp4`) for both smoothed (`w=3`) and raw (`w=1`) / `centroid` trajectories to verify that Phase 5 boundary flickers are absorbed, trace a track manually against the video, and record findings in `notes.md`.

### Acceptance Criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Event lifecycle state machine (`candidate -> active -> triggered -> resolved`) implemented per `(track_id, zone_id)` with configurable thresholds in `configs/events.yaml` | ☐ | — |
| 2 | `events.csv` matches canonical schema (`event_id, event_type, track_id, zone_id, start_time, end_time, duration_seconds`) and event count is orders of magnitude below frame count | ☐ | — |
| 3 | Hysteresis / grace period (`<= grace_frames` outside) absorbs boundary flicker without resetting dwell state (verified on synthetic tests and Phase 5 real flicker cases) | ☐ | — |
| 4 | Track-loss-in-zone handled (resolved at last seen inside time) and flagged in `event_debug_log/` | ☐ | — |
| 5 | Per-event debug trace (`event_debug_log/`) and visual dwell examples (`dwell_examples/` + `annotated_events_video.mp4`) generated | ☐ | — |
| 6 | One track manually traced frame-by-frame — event timeline matches the video and is documented in `notes.md` | ☐ | — |
| 7 | Comprehensive unit test suite `tests/test_event_state_machine.py` passes and all local CI-parity checks (`ruff check`, `ruff format --check`, `pytest`) + remote GitHub Actions CI pass 100% | ☐ | — |

### Validation & CI Plan

```bash
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -v
# phase-specific execution:
uv run --locked python cv_projects/06_dwell_time_events/scripts/generate_dwell_events.py \
    data/fallback_clips/people-detection.mp4 \
    --trajectories-csv cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv \
    --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
    --config cv_projects/06_dwell_time_events/configs/events.yaml \
    -o cv_projects/06_dwell_time_events/outputs/people_detection
```

### Exclusions

- **Best-frame full snapshots and tight person crops (`snapshot_path`, `crop_path` columns):**
  Explicitly deferred to Phase 7 (`07_visual_evidence`) per `docs/project_phases.md`.

### Micro-Task Breakdown

- [x] 1. `docs(roadmap): define Phase 6 dwell-time events scope and micro-task checklist`
- [ ] 2. `docs(events): scaffold cv_projects/06_dwell_time_events README and notes`
- [ ] 3. `feat(events): add default configs/events.yaml with dwell thresholds and grace period`
- [ ] 4. `test(helpers): expose Phase 6 dwell events scripts directory to test suite`
- [ ] 5. `types(events): define EventConfig, EventRecord, and StayDebugTrace dataclasses`
- [ ] 6. `test(events): add unit tests for EventConfig and EventRecord validation`
- [ ] 7. `feat(events): implement load_events_config and save_events_config YAML helpers`
- [ ] 8. `test(events): add unit tests for events.yaml configuration loading and per-zone overrides`
- [ ] 9. `feat(events): implement write_events_csv and read_events_csv for events.csv schema`
- [ ] 10. `test(events): add unit tests for events.csv round-trip serialization`
- [ ] 11. `feat(events): implement ZoneDwellStateMachine lifecycle candidate->active->triggered->resolved`
- [ ] 12. `test(events): add unit tests for ZoneDwellStateMachine normal entry, dwell trigger, and exit`
- [ ] 13. `test(events): add unit tests for hysteresis grace-period flicker absorption and sub-entry spikes`
- [ ] 14. `test(events): add unit tests for track-loss-in-zone resolution and flagging`
- [ ] 15. `feat(events): implement evaluate_dwell_events across all tracks, zones, and tripwire lines`
- [ ] 16. `test(events): add unit tests for evaluate_dwell_events and line-crossing cooldown deduplication`
- [ ] 17. `feat(events): implement write_event_debug_logs for per-stay debug trace inspection`
- [ ] 18. `test(events): add unit tests for event_debug_log export`
- [ ] 19. `feat(events): implement draw_dwell_events_overlay and save_dwell_examples visual proof helpers`
- [ ] 20. `test(events): add unit tests for draw_dwell_events_overlay and save_dwell_examples`
- [ ] 21. `feat(events): implement run_dwell_time_events pipeline and CLI entrypoint`
- [ ] 22. `test(events): add end-to-end pipeline and CLI tests in test_event_state_machine.py`
- [ ] 23. `feat(events): add exploratory notebook 01_explore_dwell_events.ipynb`
- [ ] 24. `docs(events): document manual track timeline trace, flicker absorption, and track-loss cases in notes`
- [ ] 25. `docs(events): document CLI, state machine lifecycle, and events.csv schema in subproject README`
- [ ] 26. `docs(readme): add Phase 6 dwell-time events CLI instructions to root README`
- [ ] 27. `docs(progress): complete Phase 6 acceptance criteria, decision log, and CI log`

