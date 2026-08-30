# 02 — Person Detection

**Phase 2** of the [project roadmap](../../docs/project_phases.md).

## Goal

Detect people and produce annotated visual proof, with GPU (CUDA) verified.

## Approach

1. Verify CUDA is available and record the GPU model/VRAM (expected: RTX 3060
   Mobile, 6 GB).
2. Run pretrained YOLO11 (`yolo11m` baseline; compare `n`/`s`/`m` speed vs.
   accuracy in a notebook), filtered to the `person` class only.
3. Confidence filtering (start at 0.4; tune and record the final value in
   `configs/detection.yaml`).
4. Draw boxes + confidence on frames; render an annotated video.
5. Save detections to `detections.csv`.
6. Pin `ultralytics`/`torch` versions and record the model weights version
   used, in the config.

## Planned Outputs

```text
outputs/
  detections.csv          # frame_id, x1, y1, x2, y2, confidence, class
  annotated_video.mp4
  annotated_frames/
configs/
  detection.yaml           # model, confidence, device, weights version
notes.md                   # model comparison, FPS on GPU, failure observations
```

## Status

- Dependencies added: `ultralytics`, `torch`, `torchvision` (CUDA build, see
  [pyproject.toml](../../pyproject.toml)).
- CUDA verified: RTX 3060 Laptop GPU, ~6.4 GB VRAM — see `notes.md`.
- Not yet done: running the detector itself. That needs real footage in
  `data/`, which Phase 1 is still waiting on (see
  [01_video_inspection](../01_video_inspection/)).

Install the exactly pinned detection stack with `uv sync --locked --extra
detection-cuda` (Windows/Linux NVIDIA) or `uv sync --locked --extra detection-cpu`.
The default environment only installs the Phase 1 reader dependencies. Notebook
tools are optional via `--group notebooks`; CPU and CUDA extras cannot be enabled
together. The CUDA extra retains the existing `cu128` wheel source.

## Layout

```text
02_person_detection/
  README.md          # this file
  notebooks/         # model-size comparison, confidence tuning
  scripts/            # reproducible detection runs
  configs/            # detection.yaml
  outputs/            # generated artifacts (gitignored)
  notes.md            # model comparison, FPS, failure observations
```
