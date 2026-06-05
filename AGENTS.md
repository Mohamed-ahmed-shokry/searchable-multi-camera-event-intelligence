# Agent Rules

## Project Identity And Phase

This repository is for **Searchable Multi-Camera Event Intelligence**.

The long-term goal is a full searchable video intelligence system that can turn camera footage into detections, tracks, trajectories, events, evidence, summaries, and searchable insights.

The guiding pipeline is:

```text
video -> vision -> behavior -> events -> search -> insight
```

This project may become a full software product later, but the current phase is focused on the computer vision foundation first.

For now, prioritize understanding and building the CV parts carefully before moving into product, backend, frontend, deployment, authentication, complex database design, or large architecture work.

Do not rush into productization unless the user explicitly asks for it.

## Current CV Focus

The early project work should focus on computer vision capabilities, especially:

- video inspection
- person detection
- person tracking
- trajectory extraction
- zone and line reasoning
- dwell-time events
- visual evidence
- heatmaps and analytics
- CV evaluation
- searchable outputs after the CV outputs are meaningful

Avoid treating the project as a dashboard, backend, or deployment project before the CV system produces useful structured outputs.

## Recommended Phase Roadmap

Build the project in focused phases:

```text
0. Project foundation
1. Video inspection
2. Person detection
3. Person tracking
4. Trajectory extraction
5. Zone and line reasoning
6. Dwell-time events
7. Visual evidence
8. Heatmaps and analytics
9. CV evaluation
10. Searchable outputs
11. Productization later
```

Do not skip directly to later phases unless the user explicitly changes the project direction.

## CV Subprojects

Use CV subprojects for research-style experiments and milestones.

Recommended pattern:

```text
cv_projects/
  <number>_<capability>/
    README.md
    notebooks/
    scripts/
    configs/
    outputs/
    notes.md
```

Each subproject should focus on one meaningful CV capability, such as video inspection, person detection, person tracking, trajectory extraction, zone reasoning, dwell-time events, heatmap generation, or evaluation.

Use notebooks for exploration, visualization, debugging, and observations.

Use scripts for reproducible experiment runs.

Move cleaned, stable, reusable logic into shared source modules later, after the experiment proves useful.

Keep generated outputs separate from reusable source code.

## 1. Think First, Code Later

Before writing code, explain what you are about to do, why that approach fits, and what alternatives exist.

Also explain the smallest useful unit of work, the expected output, and how the change will be verified.

Do not jump straight to implementation.

Start with a short brainstorming message.

Be verbose enough to make the reasoning clear. Think out loud during brainstorming, planning, trade-off analysis, and implementation updates so the user can follow the decision process.

Wait for explicit approval such as:

```text
go ahead
implement it
start coding
```

before writing code.

If the user asks for a plan, stay in planning and do not implement until they explicitly approve implementation.

## 2. Small Steps Only

Break features into the smallest logical, testable unit.

One step should correspond to one focused change that can be reviewed on its own.

Do not bundle multiple unrelated features into one edit.

Do not mix CV experiments, reusable module work, storage, dashboard work, and documentation in one change unless the user explicitly asks for that bundle.

If a change feels large, split it and ask which part to start with.

## 3. Always Explain Every Change

For every file or function you add, explain what it does and why it exists.

When changing existing behavior, explain the before/after intent and why the change is needed.

If you add a dependency, explain what it is, why it fits, and whether a simpler option exists.

After implementation, summarize what changed, what was not changed, how it was verified, and any known limitations.

## 4. Ask Before Assuming

If behavior, UI, shortcuts, library choice, or architecture is unclear, ask.

If you must choose a default, say what default you picked and why.

Important choices to clarify include dataset source, video source, model or tracker choice, event rule behavior, zone or line format, output artifact format, evaluation method, and storage format.

## 5. Never Delete Or Overwrite Silently

If you need to replace working code, show the before/after intent and explain the reason.

Ask for approval before replacing behavior that already works.

Do not remove generated outputs, sample videos, notebooks, experiment notes, or existing results unless the user explicitly approves.

Do not revert user changes unless the user explicitly asks for that.

## 6. Smallest-Part Commit Workflow

Commit each logical, testable part separately.

A commit should represent one focused change that can be reviewed and verified on its own.

Do not bundle unrelated edits into the same commit.

Before committing, check the worktree and make sure only the intended files are staged.

If the user asks for multiple parts, commit after each part before starting the next one.

## 7. Commit Message Guidelines

Use Conventional Commits in this short format:

```text
<type>[optional scope]: <description>
```

Use `feat:` for new features and `fix:` for bug fixes.

Use `docs:`, `test:`, `refactor:`, `style:`, `chore:`, `build:`, or `ci:` when they fit.

Add a scope when it clarifies the area, such as:

```text
feat(ocr): add text extraction
```

Mark breaking changes with `!` before the colon or a `BREAKING CHANGE:` footer.

Examples:

```text
docs: add commit guidelines
fix(ocr): handle empty output
feat(ui): add capture overlay
```
