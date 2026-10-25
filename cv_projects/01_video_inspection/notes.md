# Notes — Video Inspection

Observations, scene assessments, failures, and decisions for Phase 1.

## Scene Assessment

Assessed from real inspection runs (metadata + sampled frames), plus the source
publishers' scene descriptions. **Subjective visual review is explicitly handed
back to the human owner** — the numbers below are programmatic proxies.

| Source | Type | Lighting (sampled luminance) | Camera | Crowd density (resp.) | Occlusion | Suitable for |
|--------|------|------------------------------|--------|----------------------|-----------|--------------|
| `fallback_clips/people-detection.mp4` (768x432, 12 fps, 596 f) | video file | mid, stable (106–115) | fixed frontal | low (few pedestrians) | minor | Phase 2 quick tests, notebook iterations |
| `fallback_clips/store-aisle-detection.mp4` (720x404, 59.94 fps, 3921 f) | video file | mid, moderate drift (120–146 — slow camera/shutter variation) | fixed overhead-ish | medium (shoppers) | frequent (fixture occlusion) | Phases 5–8 zone/dwell scenarios |
| `fallback_clips/worker-zone-detection.mp4` (1920x1080, 59.94 fps, 4548 f) | video file | mid, stable (95–105) | fixed wide | low-medium (workers) | minor | Phases 5–6 zone entry/exit reasoning |
| `mot17/MOT17-04-FRCNN` (1920x1080, 30 fps, 1050 f) | frame sequence | nighttime street | elevated static | very dense (83 GT objects) | heavy | Phases 2–4 + 9 detection/tracking stress, MOTA/IDF1 gate |
| `mot17/MOT17-05-FRCNN` (640x480, 30 fps, 837 f) | frame sequence | mixed indoor | moving car-mounted with multi-part camera motion | medium (133 GT objects), frequent crossings | frequent | ID-switch checks, crossing conflicts |
| `mot17/MOT17-11-FRCNN` (1920x1080, 30 fps, 900 f) | frame sequence | daytime indoor/mall walkway | forward-moving camera | medium (75 GT objects) | frequent crossings | Phases 2–3 detection & quantitative MOTA/IDF1 tracking checkpoint |

MOT17 crowd-density figures are the published ground-truth object counts from the
motchallenge.net archived sequence table. Contact sheets and sampled frames for
all extracted sources were produced by a real inspection run (see Validation
below); every sampled frame decodes and is nondegenerate.

## Real-Run Validation (2026-10-24)

- Extracted `data/mot17/MOT17-11-FRCNN` (900 JPEG frames + `seqinfo.ini` + `gt/gt.txt` + `det/det.txt`, 903 files CRC32-verified) using `scripts/fetch_mot17_sequence.py`.
- Inspected all 4 local sources (`people-detection.mp4`, `store-aisle-detection.mp4`, `worker-zone-detection.mp4`, and `data/mot17/MOT17-11-FRCNN`) into `outputs/verified_dataset_run/video_info.json`:
  - `MOT17-11-FRCNN`: `source_type: frame_sequence`, `fps: 30.0`, `width: 1920`, `height: 1080`, `frame_count: 900`, `duration_seconds: 30.0`, 12 sampled frames + contact sheet (`contact_sheet_MOT17-11-FRCNN_69407af75567.png`).
- Validated via `uv run python cv_projects/01_video_inspection/scripts/validate_dataset.py cv_projects/01_video_inspection/outputs/verified_dataset_run/video_info.json` → `validation passed for 4 source(s)` (exit code 0).

## Decisions

- `VideoSource` auto-detects frame-sequence sources by the presence of
  `seqinfo.ini` in the given directory; everything else is treated as a video
  file opened via OpenCV `VideoCapture`. Keeps the two input shapes required
  by MOT17 (frames) and everything else (video files) behind one interface
  without a format flag the caller has to get right.
- Fixed a concurrency race condition in `fetch_mot17_sequence.py` where `ThreadPoolExecutor`
  workers previously shared a mutable `zipfile.ZipFile` / `RemoteFile` handle when reading
  local 30-byte ZIP headers (`Bad magic number for file header`). Workers now perform a
  single stateless ranged HTTP GET covering the 30-byte local header plus compressed
  payload (`_fetch_entry_payload`), eliminating shared state and halving HTTP round-trips.

## Failures / Limitations

- Initial `fetch_mot17_sequence.py` run hit `zipfile.BadZipFile: Bad magic number for file header` at 774/900 frames due to unsynchronized `archive.open(info)` calls across threads; fixed in commit `8fe0ded` and verified with unit tests (`tests/test_fetch_mot17_sequence.py`) and full 903-entry CRC-verified extraction.
