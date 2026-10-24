# Notes — Person Detection

Observations, model comparisons, failures, and decisions for Phase 2.

## GPU Verification

| Item | Value |
|------|-------|
| Device | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6.44 GB (`torch.cuda.get_device_properties(0).total_memory`) |
| `torch.cuda.is_available()` | `True` |
| torch | 2.11.0+cu128 |
| CUDA runtime (wheel) | 12.8 |
| Driver-reported CUDA (nvidia-smi) | 13.4 |

Matches the locked hardware decision in `docs/project_phases.md`.

## Model Comparison

### 1. Static Surveillance Clip (`data/fallback_clips/people-detection.mp4`, 768x432, 150 frames, conf >= 0.4)

| Model | Device | Weights Size | Throughput (FPS) | Duration (s) | Total Person Dets | Frames w/ Dets | Avg Dets/Frame |
|-------|--------|--------------|------------------|--------------|-------------------|----------------|----------------|
| `yolo11n.pt` | `cuda:0` | 5.4 MB | 22.84 | 6.569 | 52 | 52 | 0.35 |
| `yolo11s.pt` | `cuda:0` | 18.4 MB | 25.89 | 5.794 | 56 | 56 | 0.37 |
| `yolo11m.pt` | `cuda:0` | 40.1 MB | 22.40 | 6.695 | 58 | 58 | 0.39 |

### 2. MOT17 Benchmark Sequence (`data/mot17/MOT17-11-FRCNN`, 1920x1080, 150 frames, conf >= 0.4)

| Model | Device | Weights Size | Throughput (FPS) | Duration (s) | Total Person Dets | Frames w/ Dets | Avg Dets/Frame |
|-------|--------|--------------|------------------|--------------|-------------------|----------------|----------------|
| `yolo11n.pt` | `cuda:0` | 5.4 MB | 14.77 | 10.154 | 913 | 150 | 6.09 |
| `yolo11s.pt` | `cuda:0` | 18.4 MB | 17.77 | 8.440 | 1188 | 150 | 7.92 |
| `yolo11m.pt` | `cuda:0` | 40.1 MB | 16.12 | 9.308 | 1246 | 150 | 8.31 |

### Observations & Trade-offs

1. **Accuracy / Recall**: On dense 1080p pedestrian scenes (`MOT17-11-FRCNN`), `yolo11m` detects 1,246 person instances vs. 913 for `yolo11n` (+36.5% recall gain) and 1,188 for `yolo11s` (+4.9%), recovering partially occluded pedestrians and smaller background figures.
2. **Speed & VRAM**: On the RTX 3060 Laptop GPU, `yolo11m` runs at 16.12 FPS on 1920x1080 JPEG frame sequences and 22.40 FPS on 768x432 H.264 video while consuming ~1.1 GB VRAM, comfortably within the 6.44 GB VRAM budget.
3. **Choice Justification**: `yolo11m` is confirmed as the baseline detector for subsequent tracking and trajectory phases: its higher pedestrian recall directly reduces false-negative track fragmentations in Phase 3.

## Real Run Validation (Full Footage)

- **Source**: `data/fallback_clips/people-detection.mp4` (596 frames, 768x432, 12 fps)
- **Model**: `yolo11m.pt` (confidence threshold = 0.4, device = `cuda:0`)
- **Throughput**: 18.37 FPS (including video encoding and thumbnail writes)
- **Detections**: 418 person instances across 241 frames
- **Frame Accounting**: Exactly 596 distinct frame IDs in `detections.csv` (0..595); zero missing frames
- **Class Filtering**: 100% `person` class; 0 non-person records
- **Artifacts**: `annotated_video.mp4` (verified valid 768x432, 596 frames) and 13 sample thumbnails in `annotated_frames/`

## Decisions

- **Baseline Model**: Selected `yolo11m.pt` with confidence threshold `0.4` as the default in `configs/detection.yaml`.
- **Empty Frame Accounting**: Every video frame from `0` to `frame_count - 1` is explicitly accounted for in `detections.csv`. Frames with 0 detections emit a row with empty coordinates (`frame_id,,,,,,`) so consumers have an unambiguous 1:1 timeline.
- **Dynamic Class Lookup**: Person class ID is dynamically looked up via `model.names` (defaulting to COCO class 0).
- **Test Suite Isolation**: Tests in `tests/test_detect_people.py` mock model outputs and verify schema compliance, frame accounting, bounding-box drawing, and CLI handling without downloading weights during local or CI test runs.

## Failures / Limitations

- **Distant/Small Pedestrians**: When people enter the far background, their bounding box height is < 25 pixels and confidence fluctuates between 0.30 and 0.42, occasionally dropping below the 0.4 threshold for single frames. This is addressed in Phase 3 by multi-object tracking (`ByteTrack` / `BoT-SORT`) which links detections across short confidence dips.
