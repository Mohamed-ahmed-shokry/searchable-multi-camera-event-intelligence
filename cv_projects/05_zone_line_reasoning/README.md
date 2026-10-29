# 05 — Zone and Line Reasoning

Phase 5 adds **2D spatial geometry reasoning** (`shapely==2.2.0` + `pyyaml==6.0.3`) on top of Phase 4 trajectories (`trajectories.csv`):
- **Named Polygon Zones (`ZoneSpec`)**: evaluates point-in-polygon (`Polygon.covers(Point(x, y))`) per track and frame, emitting raw `zone_enter` and `zone_exit` state-transition signals.
- **Directed Tripwire Lines (`LineSpec`)**: evaluates consecutive trajectory segments `(p_{t-1} -> p_t)` against finite 2-point lines `(A -> B)`, emitting `line_cross` signals with direction `"A->B"` or `"B->A"`.
- **Boundary Flicker & Jitter Analysis (`analyze_boundary_flicker`)**: flags rapid `zone_enter <-> zone_exit` oscillations and back-and-forth line-crossing jitter within `<= max_flicker_frames` frames.
- **Visual Scene Authoring (`define_scene_geometry.py`)**: extracts a sample frame, overlays a pixel coordinate ruler grid (`reference_frame_grid.png`), renders polygon zones and tripwire lines (`scene_layout_preview.png`), and supports interactive OpenCV mouse-callback authoring (`--interactive`).

---

## Directory Layout

```text
cv_projects/05_zone_line_reasoning/
├── README.md
├── notes.md
├── configs/
│   └── scene_config.yaml
├── notebooks/
│   └── 01_define_and_explore_zones.ipynb
├── scripts/
│   ├── define_scene_geometry.py
│   └── reason_zones_lines.py
└── outputs/                  # gitignored generated artifacts
```

---

## Configuration Schema (`configs/scene_config.yaml`)

```yaml
zones:
  - id: entrance_area
    polygon: [[140, 350], [520, 350], [560, 431], [110, 431]]
  - id: center_concourse
    polygon: [[180, 250], [560, 250], [540, 350], [160, 350]]
  - id: right_aisle
    polygon: [[460, 200], [750, 200], [750, 360], [460, 360]]

lines:
  - id: doorway_line
    points: [[100, 360], [620, 360]]
  - id: aisle_split_line
    points: [[370, 180], [370, 420]]
```

---

## Output Schema (`zone_line_events.csv`)

| Column | Type | Description |
| --- | --- | --- |
| `frame_id` | `int` | Zero-based frame index where the signal occurred |
| `track_id` | `int` | Persistent tracker ID |
| `signal_type` | `str` | One of `zone_enter`, `zone_exit`, or `line_cross` |
| `zone_id` | `str` | Target polygon zone ID (empty for `line_cross`) |
| `line_id` | `str` | Target tripwire line ID (empty for `zone_enter` / `zone_exit`) |
| `direction` | `str` | Crossing direction `A->B` or `B->A` (empty for zone signals) |
| `timestamp` | `float` | Timestamp in seconds (`frame_id / fps`, formatted to 4 decimals) |

---

## Usage

### 1. Preview or Interactively Author Scene Geometry (`define_scene_geometry.py`)

```bash
uv run --locked python cv_projects/05_zone_line_reasoning/scripts/define_scene_geometry.py \
  data/fallback_clips/people-detection.mp4 \
  --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
  --frame-index 35 \
  -o cv_projects/05_zone_line_reasoning/outputs/people_detection
```

Pass `--interactive` on a desktop session to click polygon vertices (`z` to commit zone) and tripwire endpoints (`l` to commit line) directly on the sample frame.

### 2. Run Zone and Line Reasoning (`reason_zones_lines.py`)

```bash
uv run --locked python cv_projects/05_zone_line_reasoning/scripts/reason_zones_lines.py \
  data/fallback_clips/people-detection.mp4 \
  --trajectories-csv cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv \
  --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
  -o cv_projects/05_zone_line_reasoning/outputs/people_detection
```

Generated artifacts in `outputs/people_detection/`:
- `zone_line_events.csv` — ordered raw `zone_enter`, `zone_exit`, and `line_cross` signals
- `scene_config.yaml` — normalized scene geometry configuration used for the run
- `scene_layout_preview.png` — reference frame overlaid with polygon zones and tripwire lines
- `annotated_spatial_video.mp4` — full-clip spatial reasoning overlay video with active-signal highlights and motion trails
- `spatial_reasoning_summary.json` — signal counts per zone/line, boundary flicker diagnostics, and timing metrics
