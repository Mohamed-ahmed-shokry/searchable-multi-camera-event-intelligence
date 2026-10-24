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
| `mot17/MOT17-11-FRCNN` (1920x1080, 30 fps, 900 f) | frame sequence | daytime street | elevated static | low (75 GT objects) | light | calm control sequence; tracker sanity checks |

MOT17 crowd-density figures are the published ground-truth object counts from the
motchallenge.net archived sequence table. Contact sheets for all 6 sources were
produced by a real inspection run (see Validation below); every sampled frame
decodes and is nondegenerate.

## Real-Run Validation (2026-10-22)

- Fallback clips inspected in run
  `outputs/run_20261022T191903_437412Z_dac52167` — `video_info.json` passes
  `validate_dataset.py` for all 3 sources (clip sanity: positive metadata;
  duration consistent with frame_count/fps).
- MOT17 sequences will be inspected + spec-validated right after extraction
  (see below; filled in as they complete).

## Decisions

- `VideoSource` auto-detects frame-sequence sources by the presence of
  `seqinfo.ini` in the given directory; everything else is treated as a video
  file opened via OpenCV `VideoCapture`. Keeps the two input shapes required
  by MOT17 (frames) and everything else (video files) behind one interface
  without a format flag the caller has to get right.
- Verified `video_source.py` and `inspect_video.py` against synthetic test
  data (`data/_synthetic_test/`, not committed — gitignored) since no real
  footage is in `data/` yet. This confirms the code is correct; it does not
  substitute for the real scene-quality assessment below, which needs actual
  MOT17/VIRAT (or fallback) footage.

## Failures / Limitations

_(none yet — pending real footage)_
