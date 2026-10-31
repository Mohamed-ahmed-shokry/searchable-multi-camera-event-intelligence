# Phase 6 Observations — Dwell-Time Events (`06_dwell_time_events`)

## 1. Event Lifecycle State Machine & Compression Ratio (`people-detection.mp4`, `596` frames @ `12.0 fps`)

Running `scripts/generate_dwell_events.py` on `data/fallback_clips/people-detection.mp4` (`596` frames, `7` tracked people, `409` trajectory observations across `3` floor zones and `2` tripwire lines) completed in **`5.31 s`** (`~112 fps` including full MP4 video rendering and PNG dwell example extraction):

| Metric | Value | Notes |
| --- | ---: | --- |
| Source video frames | `596` | `768×432 @ 12.0 fps` (`49.67 s`) |
| Total trajectory points | `409` | Across `7` unique tracks (`1, 2, 3, 5, 6, 7, 9`) |
| Total zone stays (`StayDebugTrace`) | `16` | `10` reached `triggered`, `6` resolved in `active` (short pass-throughs `< threshold`) |
| **`dwell_time_event` count** | **`10`** | **59.6× lower than frame count (`0.0168` dwell events/frame)** |
| Total events in `events.csv` | `48` | `16 zone_entry` + `13 zone_exit` + `10 dwell_time_event` + `9 line_crossing` (`0.0805` events/frame) |
| Track-loss-in-zone resolutions | `3` | `stay_t1_right_aisle_001`, `stay_t7_right_aisle_001`, `stay_t6_right_aisle_001` |

### Stays by Zone (`configs/events.yaml`: `entrance_area=1.2s`, `center_concourse=1.2s`, `right_aisle=1.5s`)
| Stay ID | Track | Zone | Enter Frame (`t`) | Last Inside Frame (`t`) | Duration (`s`) | Peak State | Resolution Reason |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `stay_t1_entrance_area_001` | `1` | `entrance_area` | `19 (1.5833s)` | `35 (2.9167s)` | **`1.3333`** | `triggered` | `exited` (frame `36`) |
| `stay_t1_center_concourse_001` | `1` | `center_concourse` | `36 (3.0000s)` | `55 (4.5833s)` | **`1.5833`** | `triggered` | `exited` (frame `56`) |
| `stay_t1_right_aisle_001` | `1` | `right_aisle` | `47 (3.9167s)` | `75 (6.2500s)` | **`2.3333`** | `triggered` | **`track_lost_in_zone`** |
| `stay_t3_entrance_area_001` | `3` | `entrance_area` | `181 (15.0833s)` | `204 (17.0000s)` | **`1.9167`** | `triggered` | `exited` (frame `205`) |
| `stay_t2_entrance_area_001` | `2` | `entrance_area` | `183 (15.2500s)` | `205 (17.0833s)` | **`1.8333`** | `triggered` | `exited` (frame `206`) |
| `stay_t3_center_concourse_001` | `3` | `center_concourse` | `205 (17.0833s)` | `225 (18.7500s)` | **`1.6667`** | `triggered` | `exited` (frame `226`) |
| `stay_t2_center_concourse_001` | `2` | `center_concourse` | `206 (17.1667s)` | `216 (18.0000s)` | `0.8333` | `active` | `exited` (short pass-through) |
| `stay_t5_entrance_area_001` | `5` | `entrance_area` | `340 (28.3333s)` | `364 (30.3333s)` | **`2.0000`** | `triggered` | `exited` (frame `365`) |
| `stay_t6_entrance_area_001` | `6` | `entrance_area` | `345 (28.7500s)` | `368 (30.6667s)` | **`1.9167`** | `triggered` | `exited` (frame `369`) |
| `stay_t7_entrance_area_001` | `7` | `entrance_area` | `353 (29.4167s)` | `363 (30.2500s)` | `0.8333` | `active` | `exited` (short pass-through) |
| `stay_t7_center_concourse_001` | `7` | `center_concourse` | `364 (30.3333s)` | `377 (31.4167s)` | `1.0833` | `active` | `exited` (short pass-through) |
| `stay_t5_center_concourse_001` | `5` | `center_concourse` | `365 (30.4167s)` | `379 (31.5833s)` | `1.1667` | `active` | `exited` (short pass-through) |
| `stay_t7_right_aisle_001` | `7` | `right_aisle` | `371 (30.9167s)` | `410 (34.1667s)` | **`3.2500`** | `triggered` | **`track_lost_in_zone`** |
| `stay_t6_right_aisle_001` | `6` | `right_aisle` | `372 (31.0000s)` | `401 (33.4167s)` | **`2.4167`** | `triggered` | **`track_lost_in_zone`** |
| `stay_t9_entrance_area_001` | `9` | `entrance_area` | `523 (43.5833s)` | `536 (44.6667s)` | `1.0833` | `active` | `exited` (short pass-through) |
| `stay_t9_center_concourse_001` | `9` | `center_concourse` | `537 (44.7500s)` | `547 (45.5833s)` | `0.8333` | `active` | `exited` (short pass-through) |

---

## 2. Manual Frame-by-Frame Track Trace: `Track 1` (`frames 19–75`, `t = 1.5833s – 6.2500s`)

We manually traced `Track 1` across `annotated_events_video.mp4`, `dwell_examples/stay_t1_*`, and `event_debug_log/stay_t1_*.json`:

1. **Frame `19` (`t = 1.5833s`) — Entry into `entrance_area`:**
   - Shopper `Track 1` steps into the lower threshold at `(301.25, 429.91)` inside `entrance_area`.
   - `ZoneDwellStateMachine(1, "entrance_area")` transitions `outside -> candidate -> active` (`min_entry_frames=1`) and emits `evt_0001` (`zone_entry`, `start_time=1.5833`).
2. **Frame `34` (`t = 2.8333s`) — Dwell threshold reached in `entrance_area`:**
   - Accumulated stay duration reaches `2.8333 - 1.5833 = 1.2500s >= 1.2s`.
   - State machine transitions `active -> triggered`.
3. **Frame `35` (`t = 2.9167s`) — Last frame inside `entrance_area`:**
   - Foot anchor reaches `(360.83, 369.51)` (`duration = 1.3334s`; visual proof saved at `dwell_examples/stay_t1_entrance_area_001_frame_0035.png`).
4. **Frame `36` (`t = 3.0000s`) — Transition from `entrance_area` across `doorway_line` into `center_concourse`:**
   - Foot anchor moves to `(368.12, 345.01)` (`y < 350`), crossing `doorway_line` (`y = 360`) in direction `B->A` (`evt_0003`) and entering `center_concourse` (`evt_0004`, `zone_entry`).
   - After `grace_frames = 5` outside frames (`frame 41`), `stay_t1_entrance_area_001` resolves as `exited` at its first outside frame (`frame 36`, `t = 3.0000s`), emitting `evt_0002` (`dwell_time_event`, `1.5833s -> 2.9167s`, `duration = 1.3334s`) and `evt_0005` (`zone_exit`, `t = 3.0000s`).
5. **Frame `37` (`t = 3.0833s`) — Crossing `aisle_split_line`:**
   - Foot anchor moves from `x = 368.12` to `x = 374.93` across `aisle_split_line` (`x = 370`) in direction `B->A`, emitting `evt_0007` (`line_crossing`, `aisle_split_line:B->A`).
6. **Frame `47` (`t = 3.9167s`) — Entry into `right_aisle` (while still overlapping `center_concourse`):**
   - At `(460.87, 267.97)`, `Track 1` enters `right_aisle` (`x >= 460`), emitting `evt_0008` (`zone_entry`). Because `center_concourse` (`x <= 560`) and `right_aisle` (`x >= 460`) overlap in `x ∈ [460, 560]`, `Track 1` is simultaneously tracked in both independent per-zone state machines from frame `47` to frame `55`.
7. **Frame `51` (`t = 4.2500s`) & Frame `56` (`t = 4.6667s`) — Trigger and exit from `center_concourse`:**
   - `stay_t1_center_concourse_001` triggers at frame `51` (`1.2500s >= 1.2s`), reaches its last inside frame at `55` (`t = 4.5833s`, `duration = 1.5833s`; visual proof at `dwell_examples/stay_t1_center_concourse_001_frame_0055.png`), and exits `center_concourse` at frame `56` (`t = 4.6667s`), emitting `evt_0006` (`dwell_time_event`, `1.5833s`) and `evt_0010` (`zone_exit`).
8. **Frame `65` (`t = 5.4167s`) & Frame `75` (`t = 6.2500s`) — Dwell trigger and track loss inside `right_aisle`:**
   - `stay_t1_right_aisle_001` triggers at frame `65` (`1.5000s >= 1.5s`) and continues to frame `75` (`t = 6.2500s`, `(737.04, 236.79)`), where the shopper walks behind the right-hand wall/pillar and ByteTrack terminates `Track 1`.
   - `finish_track()` resolves `stay_t1_right_aisle_001` at `last_inside_time = 6.2500s` with `resolution_reason = "track_lost_in_zone"` and `track_lost_in_zone = true`, emitting `evt_0009` (`dwell_time_event`, `3.9167s -> 6.2500s`, `duration = 2.3333s`; visual proof at `dwell_examples/stay_t1_right_aisle_001_frame_0075.png`) without fabricating a false `zone_exit`.

---

## 3. Verification of Hysteresis / Boundary Flicker Absorption (Phase 5 Flicker Cases)

To verify that `grace_frames = 5` absorbs real boundary flicker, we ran `evaluate_dwell_events` on Phase 5's unsmoothed (`w=1`) `bottom_center` trajectories (where `Track 3` suffered a 1-frame boundary exit/re-entry oscillation on `entrance_area` at frames `194–195`) with `grace_frames = 0` vs `grace_frames = 5`:

| Trajectory Input | `grace_frames` | Total Stays | `zone_entry` | `zone_exit` | `dwell_time_event` | Flickers Absorbed | Effect on `Track 3` in `entrance_area` (`frames 181–204`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw `bottom_center` (`w=1`) | `0` (no grace) | `17` | `17` | `14` | **`9` (1 lost!)** | `0` | Split into 2 short stays (`1.00s` at frames `181–193` and `0.75s` at frames `195–204`); **neither reaches `1.2s`, losing Track 3's dwell event!** |
| **Raw `bottom_center` (`w=1`)** | **`5` (default)** | **`16`** | **`16`** | **`13`** | **`10`** | **`1`** | **1-frame excursion at frame `194` is absorbed (`flicker_absorbed_after_1_frames`); single `1.9167s` dwell event preserved!** |
| Smoothed `bottom_center` (`w=3`) | `5` (default) | `16` | `16` | `13` | `10` | `0` | Clean `1.9167s` dwell event (`48` total events match `w=1, g=5` 100%) |

This demonstrates defense in depth: Phase 4's causal 3-frame rolling mean (`w=3`) smooths out sub-pixel bbox jitter geometrically, and Phase 6's hysteresis grace period (`grace_frames = 5`) guarantees that even unsmoothed boundary oscillations up to 5 frames (`0.42s @ 12 fps`) never split a continuous stay or drop a `dwell_time_event`.
