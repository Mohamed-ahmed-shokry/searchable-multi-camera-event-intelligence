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
outputs/<run_id>/
  run.json                 # complete/failed status
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

Each invocation creates a new directory under `outputs/`. Use `--output-dir
outputs/my_inspection` to select a **new** directory; existing directories are
never overwritten. Source artifact names include a path hash so equal filenames
from different folders remain distinct. `video_info.json` lists the relative
paths of every sampled image and contact sheet.

Only `run.json` with `status: "complete"` marks a successful batch. A read or write
failure returns a nonzero exit code and retains partial artifacts with failed
status when the output device is writable. Earlier runs are left untouched.
Runs created by the older CLI directly under `outputs/` remain there unchanged.

Use `--samples 24` to change the default of 12 (allowed: 1–120). Samples include
the first and last frame when at least two are requested; a single sample uses
frame 0. Contact sheets preserve aspect ratio with black padding. Only small
thumbnails are retained in memory; individual JPEG samples keep full resolution.
Metadata includes zero-based sampled indices and timestamps computed as
`index / fps`.

Inspection decodes only the sampled frames. A successful contact sheet does not
prove that every unsampled frame is readable; full processing must still handle
decode failures.

Frame-sequence metadata must declare a finite, positive `frameRate`, positive
`seqLength`, `imWidth`, and `imHeight`. Images must be numbered consecutively
from 1, match the declared count and dimensions, and stay inside the sequence
directory. Invalid or incomplete sequences fail instead of silently shifting
frame IDs. The reader's public indices are zero-based: index 0 is MOT image 1.

Inputs must be existing local files or sequence directories. Live streams,
unknown frame counts/FPS, failed seeks, and changes in decoded dimensions are
rejected. Video timing uses the reported constant FPS; variable-frame-rate
timestamps are not supported. Metadata records the resolved absolute source
path. Close the reader (prefer `with VideoSource(...)`) before releasing a run;
reads after closing fail explicitly.

For bounded-memory processing, `source.iter_frames(start=0, stop=None, step=1)`
yields `(zero_based_index, bgr_image)` pairs. `stop` is exclusive. Consecutive
video reads decode sequentially; random access seeks only when needed. No
decoding speed or detector throughput claim is implied by this optimization.

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
