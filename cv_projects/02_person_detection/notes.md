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

Benchmark executed on real test footage (`data/fallback_clips/people-detection.mp4`, 768x432, 150 frames, conf >= 0.4):

| Model | Device | Weights Size | Throughput (FPS) | Duration (s) | Total Person Dets | Frames w/ Dets | Avg Dets/Frame |
|-------|--------|--------------|------------------|--------------|-------------------|----------------|----------------|
| `yolo11n.pt` | `cuda:0` | 5.4 MB | 22.84 | 6.569 | 52 | 52 | 0.35 |
| `yolo11s.pt` | `cuda:0` | 18.4 MB | 25.89 | 5.794 | 56 | 56 | 0.37 |
| `yolo11m.pt` | `cuda:0` | 40.1 MB | 22.40 | 6.695 | 58 | 58 | 0.39 |

### Observations & Trade-offs

1. **Accuracy / Recall**: `yolo11m` detected 58 person instances compared to 52 for `yolo11n` (+11.5% increase), successfully identifying distant pedestrians in the upper third of the frame that `yolo11n` missed or filtered out due to confidence < 0.4.
2. **Speed & VRAM**: On the RTX 3060 Laptop GPU, all three models comfortably exceed real-time processing speeds (>22 FPS on 12 FPS source). `yolo11m` consumes ~1.1 GB VRAM during batch-1 inference, comfortably within the 6.44 GB total VRAM.
3. **Choice Justification**: `yolo11m` is confirmed as the baseline detector for subsequent tracking and trajectory phases: it maximizes detection stability with negligible latency difference.

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

- **Distant/Small Pedestrians**: When people enter the far background, their bounding box height is < 25 pixels and confidence fluctuates between 0.30 and 0.42, occasionally dropping below the 0.4 threshold for single frames. This is expected and will be addressed by multi-object tracking (Phase 3) which links detections across time.
