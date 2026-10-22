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

## Executed Phase: 1b — Data Acquisition & Inspection Validation (2026-10-22)

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
