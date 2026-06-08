# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Searchable Multi-Camera Event Intelligence is a computer-vision-first project for turning video footage into structured intelligence: detections, tracks, trajectories, events, visual evidence, summaries, and searchable outputs.

**Current phase**: Foundation and planning. No CV code has been implemented yet.

**Pipeline concept**: video -> vision -> behavior -> events -> search -> insight

## Development Commands

```bash
# Package management (uses uv)
uv sync                    # Install dependencies
uv add <package>           # Add a dependency
uv run python main.py      # Run main script

# Python version
# Requires Python 3.12+ (see .python-version)
```

## Project Structure

The project follows a notebook-based research workflow:

```
cv_projects/                    # CV experiments and milestones
  <number>_<capability>/
    README.md
    notebooks/                  # Exploration, visualization, debugging
    scripts/                    # Reproducible experiment runs
    configs/
    outputs/
    notes.md

outputs/runs/<run_id>/         # Per-run artifacts (when implemented)
```

**Workflow progression**: notebook exploration -> reproducible script -> reusable module

## Phase Roadmap

Build in order, do not skip to later phases without explicit direction:

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
11. Productization (later)

## Agent Workflow Rules

See AGENTS.md for complete rules. Key points:

- **Think first, code later**: Explain approach and wait for explicit approval ("go ahead", "implement it") before writing code
- **Small steps only**: One focused, testable change at a time
- **Ask before assuming**: Clarify model choice, video source, event rules, output formats
- **Never delete silently**: Show before/after and get approval for replacing working code
- **Commit each part separately**: One logical change per commit

## Commit Message Format

Use Conventional Commits:

```
<type>[scope]: <description>

# Examples:
feat(detection): add YOLO person detector
fix(tracking): handle empty frame edge case
docs: add trajectory format specification
```

Types: feat, fix, docs, test, refactor, style, chore, build, ci

## CV-First Philosophy

- Do not rush into backend, frontend, dashboard, or deployment
- Focus on CV capabilities that produce useful structured outputs first
- Event logic should generate one clean event per meaningful behavior, not frame-by-frame noise
- Deferred: React dashboard, backend API, auth, deployment, RTSP streams, custom training, multi-camera re-id
