# 01 — Video Inspection

**Phase 1** of the [project roadmap](../../docs/project_phases.md).

## Goal

Acquire test footage and fully understand it *before* running any models:
read metadata, sample frames, and assess scene quality. The deliverable is
knowledge plus a reusable reader — not detection.

## Approach

1. Place test footage in `data/` (video file or frame-sequence directory).
2. Build a `VideoSource` reader that handles **both** input shapes:
   - video files (OpenCV `VideoCapture`)
   - MOT17-style JPEG frame sequences + `seqinfo.ini`
3. Read metadata: FPS, resolution, frame count, duration, codec.
4. Sample frames at regular intervals; build a contact sheet per source.
5. Assess scene quality: lighting, angle, crowd density, occlusion.

## Planned Outputs

```text
outputs/
  video_info.json          # one entry per source
  sample_frames/
  contact_sheet_<source>.png
notes.md                   # scene assessment per clip
```

## Status

Step 0 — scaffold. Reader, inspection, and footage handling come in later steps.

## Layout

```text
01_video_inspection/
  README.md          # this file
  notebooks/         # exploration, visual debugging
  scripts/           # reproducible inspection runs
  configs/           # source lists / inspection settings
  outputs/           # generated artifacts (gitignored)
  notes.md           # observations, scene assessment
```
