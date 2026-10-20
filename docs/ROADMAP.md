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
| 1 | `pytest` runs the full existing suite in one command; all tests pass | ☐ | — |
| 2 | `ruff check` passes with the repo-managed strictness (default + line-length config) | ☐ | — |
| 3 | `ruff format --check` passes on all tracked Python files | ☐ | — |
| 4 | GitHub Actions workflow executes `uv sync --locked`, ruff checks, and pytest on `windows-latest` (matching the dev OS) | ☐ | — |
| 5 | All local CI-parity checks and the remote CI workflow run pass 100% | ☐ | — |
| 6 | Draft PR open with phase summary tied to this roadmap | ☐ | — |
| 7 | README + progress record + decision log updated (separate commits) | ☐ | — |

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

- [ ] 1. `docs(roadmap): define CI-0 phase scope and micro-task checklist`
- [ ] 2. `build(dev): add pytest runner to a dev dependency group`
- [ ] 3. `build(dev): configure pytest testpaths in pyproject.toml`
- [ ] 4. `test(runner): verify pytest executes the existing 46-test suite`
- [ ] 5. `build(dev): add ruff to the dev dependency group`
- [ ] 6. `build(lint): configure ruff line length and target version`
- [ ] 7. `style: fix ruff findings per file (small atomic commits)`
- [ ] 8. `style: apply ruff format repo-wide (one file-group per commit)`
- [ ] 9. `test(regression): confirm full pytest suite green after style commits`
- [ ] 10. `ci: add GitHub Actions workflow (uv sync, ruff check, ruff format, pytest)`
- [ ] 11. `ci: verify local parity of every workflow command`
- [ ] 12. `docs(readme): document pytest and ruff commands replacing unittest instructions`
- [ ] 13. Push branch, create draft PR
- [ ] 14. Monitor remote CI; fix any failure per its own micro-commit
- [ ] 15. Final: tick acceptance criteria, decision log, CI log (separate docs commits)

### Decision Log

| Decision | Options considered | Choice + repository evidence |
|----------|--------------------|------------------------------|
| Test runner | stdlib unittest (status quo) / pytest dev group | **pytest** as runner — existing tests are `unittest.TestCase`, so pytest executes them without rewriting tests; runner unification is reversible and low-risk. unittest stays the base style per existing code. |
| Linter/formatter | ruff / flake8+black / none | **ruff** — single tool, uv-native era standard, zero-config default behavior closest to repo's existing clean style; flake8+black is two tools and slower. |
| CI runner OS | windows-latest / ubuntu-latest / matrix | **windows-latest only** — dev machine is Windows (`win32`), `helpers.make_video` uses MJPG `VideoWriter` whose Linux behavior can't be verified locally; ubuntu can be added as a separate verified follow-up. |
| CI env install scope | default only / + detection extras / + notebooks | **default only** — tests only need numpy+opencv+pytest+ruff (Stage A evidence); CUDA extras ~2.5 GB on CI and untestable without GPU; README already treats extras as opt-in. |
| pytest rewrite of tests | keep unittest classes / rewrite to pytest style | **keep unittest classes** — pytest runs them natively; no churn, smallest diff. |
| Roadmap file | extend project_phases.md / new docs/ROADMAP.md | **new `docs/ROADMAP.md`** — phase-brief records stay separate from the canonical locked CV DoD plan; README already links to `project_phases.md` for phases. |

### CI & Validation Log

| Command | Result | Notes |
|---------|--------|-------|
| `uv run python -m unittest discover -s tests -v` (Stage A baseline) | ✅ 46 tests OK | pre-change baseline |
| `uv run pytest` (final) | ☐ | awaiting |
| `uv run ruff check .` (final) | ☐ | awaiting |
| `uv run ruff format --check .` (final) | ☐ | awaiting |
| Remote CI run on HEAD | ☐ | awaiting |
