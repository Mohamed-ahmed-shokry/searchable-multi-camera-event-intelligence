# Data

Test footage for the project. **Contents are gitignored** — only this README is
tracked. Do not commit or redistribute footage.

## Layout

```text
data/
  fallback_clips/   # static surveillance-style clips (fallbacks for Phases 1-8)
  mot17/            # MOT17 training sequences (frame-sequence layout)
```

MOT17 sequences are stream-extracted range-by-range from the official archive with
`cv_projects/01_video_inspection/scripts/fetch_mot17_sequence.py`; the ~5.9 GB
archive is never stored on disk in full. Repeat with:

```bash
uv run python cv_projects/01_video_inspection/scripts/fetch_mot17_sequence.py \
    data/mot17 MOT17-04-FRCNN MOT17-05-FRCNN MOT17-11-FRCNN
```

## Acquired Footage (2026-10-22)

### Static-camera fallback clips (`fallback_clips/`)

| File | Source (URL) | License | Local size | Specs (CV2 metadata) | Why chosen |
|------|--------------|---------|-----------|-----------------------|------------|
| `people-detection.mp4` | https://github.com/intel-iot-devkit/sample-videos (master, 2026-10-22) | CC BY-4.0 | 5.2 MB | 768x432, 12 fps, 596 frames | pedestrians crossing a fixed camera; small/fast for notebooks |
| `store-aisle-detection.mp4` | https://github.com/intel-iot-devkit/sample-videos (master, 2026-10-22) | CC BY-4.0 | 8.8 MB | 720x404, 59.94 fps, 3921 frames | fixed store-aisle scene; several dwell-type zones |
| `worker-zone-detection.mp4` | https://github.com/intel-iot-devkit/sample-videos (master, 2026-10-22) | CC BY-4.0 | 12.0 MB | 1920x1080, 59.94 fps, 4548 frames | fixed workplace camera with obvious zones; good for Phase 5+ reasoning |

Rationale: `project_phases.md` allows static surveillance-style fallback footage
instead of VIRAT (large/flaky downloads). These clips are small, fixed-camera, and
contain people moving through definable areas — exactly what Phases 5–8 need.

### MOT17 training sequences (`mot17/`)

| Directory | Source | License | Why chosen |
|-----------|--------|---------|------------|
| `MOT17-04-FRCNN/` | official https://motchallenge.net/data/MOT17.zip (ranged extraction, 2026-10-22) | CC BY-NC-SA 3.0 | dense pedestrian street at night, elevated viewpoint; crowd/occlusion stress case |
| `MOT17-05-FRCNN/` | official https://motchallenge.net/data/MOT17.zip (ranged extraction, 2026-10-22) | CC BY-NC-SA 3.0 | crossing conflicts with camera motion; hard for stable IDs |
| `MOT17-11-FRCNN/` | official https://motchallenge.net/data/MOT17.zip (ranged extraction, 2026-10-22) | CC BY-NC-SA 3.0 | simpler street scene, fewer people; useful as the calm control sequence |

Each sequence directory keeps the original MOT17 layout (`img1/`, `det/`, `gt/`,
`seqinfo.ini`). Phase 1 code reads the `seqinfo.ini` frame-sequence layout directly.`

## License Notes

- **MOT17** — CC BY-NC-SA 3.0 (per motchallenge.net). Fine for evaluation and
  learning; do NOT redistribute footage. Evaluation results do not contaminate
  a future product.
- **Fallback clips** — CC BY-4.0 (intel-iot-devkit/sample-videos); attribution
  preserved above. Redistribution of the footage itself is still avoided here.

Record the source URL, license, and selection rationale for every clip added
here.
