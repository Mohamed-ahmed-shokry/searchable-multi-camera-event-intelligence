# 06 — Dwell-Time Events

Phase 6 implements the stateful event lifecycle state machine (`candidate -> active -> triggered -> resolved`) per `(track_id, zone_id)` on top of Phase 4 trajectories (`trajectories.csv`) and Phase 5 scene geometry (`scene_config.yaml`):
- **One Behavior -> One Clean Event**: converts frame-by-frame zone occupancy into exactly one `zone_entry` per confirmed entry, at most one `dwell_time_event` per continuous stay (`duration_seconds >= min_dwell_seconds`), and one `zone_exit` when the track genuinely leaves the zone.
- **Hysteresis / Grace Period (`grace_frames`)**: brief boundary flickers (`<= grace_frames` frames outside the polygon) are absorbed without resetting accumulated dwell time or emitting duplicate `zone_exit` / `zone_entry` alerts.
- **Track-Loss-in-Zone Handling**: if a track terminates while inside a zone (or within the grace window), the stay resolves cleanly at `last_inside_time` and sets `track_lost_in_zone: true` in the audit trace without fabricating a false `zone_exit`.
- **Per-Stay Debug Audit Log (`event_debug_log/`) & Visual Proof (`dwell_examples/`)**: exports JSON state-transition traces for every stay and saves annotated visual proof frames at peak dwell timestamps alongside `annotated_events_video.mp4`.

---

## Directory Layout

```text
cv_projects/06_dwell_time_events/
├── README.md
├── notes.md
├── configs/
│   └── events.yaml
├── notebooks/
│   └── 01_explore_dwell_events.ipynb
├── scripts/
│   └── generate_dwell_events.py
└── outputs/                  # gitignored generated artifacts
    └── people_detection/
        ├── events.csv
        ├── events.yaml
        ├── dwell_events_summary.json
        ├── annotated_events_video.mp4
        ├── event_debug_log/
        │   ├── stay_traces.json
        │   └── stay_t<id>_<zone>_<seq>.json
        └── dwell_examples/
            └── stay_t<id>_<zone>_<seq>_frame_<frame>.png
```

---

## State Machine Lifecycle (`ZoneDwellStateMachine`)

```text
outside ──(is_inside)──► candidate ──(inside_frames >= min_entry_frames)──► active ──(duration >= min_dwell_seconds)──► triggered
  ▲                          │                                                │                                            │
  └────(discarded)───────────┴──────(outside_streak > grace_frames OR finish_track)────────────────────────────────────────┘
                                                               │
                                                               ▼
                                                           resolved
```

---

## Configuration (`configs/events.yaml`)

```yaml
min_dwell_seconds: 1.2
min_entry_frames: 1
grace_frames: 5
line_cooldown_frames: 5
include_line_crossings: true

per_zone_min_dwell_seconds:
  entrance_area: 1.2
  center_concourse: 1.2
  right_aisle: 1.5
```

---

## Output Schema (`events.csv`)

| Column | Type | Description |
| --- | --- | --- |
| `event_id` | `str` | Deterministic sequential event ID (`evt_0001`, `evt_0002`, ...) |
| `event_type` | `str` | One of `zone_entry`, `zone_exit`, `dwell_time_event`, or `line_crossing` |
| `track_id` | `int` | Persistent tracker ID |
| `zone_id` | `str` | Polygon zone ID (or `<line_id>:<direction>` for `line_crossing`) |
| `start_time` | `float` | Event start timestamp in seconds (formatted to 4 decimals) |
| `end_time` | `float` | Event end timestamp in seconds (`== start_time` for instantaneous transitions) |
| `duration_seconds` | `float` | Continuous dwell duration in seconds (`0.0000` for instantaneous transitions) |

---

## Usage

```bash
uv run --locked python cv_projects/06_dwell_time_events/scripts/generate_dwell_events.py \
  data/fallback_clips/people-detection.mp4 \
  --trajectories-csv cv_projects/04_trajectory_extraction/outputs/people_detection/trajectories.csv \
  --scene-config cv_projects/05_zone_line_reasoning/configs/scene_config.yaml \
  --config cv_projects/06_dwell_time_events/configs/events.yaml \
  -o cv_projects/06_dwell_time_events/outputs/people_detection
```
