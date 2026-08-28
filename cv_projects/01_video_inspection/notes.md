# Notes — Video Inspection

Observations, scene assessments, failures, and decisions for Phase 1.

## Scene Assessment

_One entry per source once footage is acquired and inspected._

| Source | Type | Lighting | Camera angle | Crowd density | Occlusion | Suitable for |
|--------|------|----------|--------------|---------------|-----------|--------------|
| _tbd_  |      |          |              |               |           |              |

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
