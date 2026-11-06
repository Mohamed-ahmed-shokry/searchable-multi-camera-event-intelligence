# Phase 9 — CV Evaluation (`09_cv_evaluation`)

## Goal

Measure the computer vision pipeline honestly against ground truth (`MOT17-11-FRCNN`) and
by manual playback verification on static surveillance footage (`people-detection.mp4`),
collect per-stage processing FPS benchmarks, and export annotated visual failure examples
with concrete root-cause explanations.

## Directory Structure

```text
cv_projects/09_cv_evaluation/
  README.md
  notes.md
  configs/
    evaluation.yaml
  notebooks/
    01_explore_cv_evaluation.ipynb
  scripts/
    evaluate_pipeline.py
  outputs/
    full_evaluation/
      evaluation_report.md
      metrics.json
      evaluation.yaml
      failure_examples/
        failure_01_mot17_crowded_id_switch.jpg
        failure_02_mot17_small_occluded_misses.jpg
        failure_03_boundary_truncated_crop.jpg
        failure_04_track_loss_in_right_aisle.jpg
```

## Quick Start

```bash
uv run --locked python cv_projects/09_cv_evaluation/scripts/evaluate_pipeline.py \
    --config cv_projects/09_cv_evaluation/configs/evaluation.yaml \
    --output-dir cv_projects/09_cv_evaluation/outputs/full_evaluation
```
