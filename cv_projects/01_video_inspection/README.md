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

- `scripts/video_source.py` — dual-mode reader (video file / MOT17-style frame
  sequence), verified against synthetic test data.
- `scripts/inspect_video.py` — CLI producing `video_info.json`, sampled frames,
  and contact sheets, verified end-to-end against synthetic test data.
- Not yet done: real MOT17/VIRAT (or fallback) footage in `data/`, and the
  manual scene-quality assessment in `notes.md` that depends on it.

Run it:

```bash
uv run python cv_projects/01_video_inspection/scripts/inspect_video.py <video_file_or_sequence_dir> [...]
```

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
