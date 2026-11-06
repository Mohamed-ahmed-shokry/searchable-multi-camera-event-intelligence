# Phase 9 — CV Evaluation (`09_cv_evaluation`)

## Goal

Measure the computer vision pipeline honestly against ground truth (`MOT17-11-FRCNN`) and
by manual playback verification on static surveillance footage (`people-detection.mp4`),
collect per-stage processing FPS benchmarks across Phases 1–8, and export annotated visual
failure examples (`failure_examples/*.jpg`) with concrete root-cause explanations.

## Key Results (`outputs/full_evaluation/`)

- **Quantitative Detection (`MOT17-11-FRCNN`, `300` frames, `3,559` GT boxes, `IoU >= 0.50`):**
  - Precision = **`0.8287` (`82.87%`)**, Recall = **`0.5670` (`56.70%`)**, F1 = **`0.6733`**,
    Mean Matched IoU = **`0.8587`** (`TP = 2,018`, `FP = 417`, `FN = 1,541`).
- **Quantitative Tracking (`MOT17-11-FRCNN`, `33` GT tracks):**
  - `ByteTrack`: **`MOTA = 0.4484`**, `MOTP = 0.8587`, `IDF1 = 0.6026`, `ID Switches = 5`,
    `Fragmentations = 35`, `MT = 9/33`, `PT = 10/33`, `ML = 14/33`.
  - `BoT-SORT`: `MOTA = 0.4470`, **`MOTP = 0.8656`**, **`IDF1 = 0.6076`**, **`ID Switches = 3`**,
    `Fragmentations = 45`, `MT = 9/33`, `PT = 10/33`, `ML = 14/33`.
- **Qualitative Surveillance Spot-Check (`people-detection.mp4`, `48` events):**
  - Zone entry accuracy = **`100.0%`** (`16/16`), Zone exit accuracy = **`100.0%`** (`13/13`),
    Line crossing accuracy = **`100.0%`** (`9/9`), Dwell event count match = **`True`** (`10/10`,
    max duration error = `0.0000s`), Track-loss-in-zone count match = **`True`** (`3/3`).
- **Per-Stage Throughput (`596` frames, RTX 3060 Laptop GPU + CPU):**
  - `01_video_inspection`: `1419.05 FPS` | `02_person_detection`: `18.37 FPS` |
    `03_person_tracking`: `17.06 FPS` | `04_trajectory_extraction`: `244.46 FPS` |
    `05_zone_line_reasoning`: `142.41 FPS` | `06_dwell_time_events`: `112.22 FPS` |
    `07_visual_evidence`: `537.47 FPS` | `08_heatmaps_analytics`: `1876.57 FPS`.

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
