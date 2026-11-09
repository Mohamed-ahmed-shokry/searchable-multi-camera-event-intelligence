# Phase 10 Worked Search Examples — `search_examples.md`

Every query below was executed directly against `intelligence.db` via `scripts/search_events.py`.

---

## Query 1 — Dwell-Time Events in `right_aisle`

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --event-type dwell_time_event --zone right_aisle
```

**Returned `3` events:**

| `event_id` | `event_type` | `track_id` | `zone_id` | `start_time` | `end_time` | `duration_seconds` | `snapshot_path` | `crop_path` |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `evt_0009` | `dwell_time_event` | `1` | `right_aisle` | `3.9167` | `6.2500` | `2.3333` | `snapshots/evt_0009_1.jpg` | `crops/evt_0009_1.jpg` |
| `evt_0038` | `dwell_time_event` | `7` | `right_aisle` | `30.9167` | `34.1667` | `3.2500` | `snapshots/evt_0038_7.jpg` | `crops/evt_0038_7.jpg` |
| `evt_0041` | `dwell_time_event` | `6` | `right_aisle` | `31.0000` | `33.4167` | `2.4167` | `snapshots/evt_0041_6.jpg` | `crops/evt_0041_6.jpg` |

---

## Query 2 — `doorway_line` Crossings Between `15.0s` and `20.0s`

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --event-type line_crossing --zone doorway_line --min-time 15.0 --max-time 20.0
```

**Returned `2` events:**

| `event_id` | `event_type` | `track_id` | `zone_id` | `start_time` | `snapshot_path` |
| --- | --- | ---: | --- | ---: | --- |
| `evt_0016` | `line_crossing` | `3` | `doorway_line:B->A` | `17.0000` | `snapshots/evt_0016_3.jpg` |
| `evt_0020` | `line_crossing` | `2` | `doorway_line:B->A` | `17.1667` | `snapshots/evt_0020_2.jpg` |

---

## Query 3 — Complete Event Timeline for `track_id = 1`

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --track-id 1
```

**Returned `10` events:**

| `event_id` | `event_type` | `zone_id` | `start_time` | `end_time` | `duration_seconds` |
| --- | --- | --- | ---: | ---: | ---: |
| `evt_0001` | `zone_entry` | `entrance_area` | `1.5833` | `1.5833` | `0.0000` |
| `evt_0002` | `dwell_time_event` | `entrance_area` | `1.5833` | `2.9167` | `1.3334` |
| `evt_0003` | `line_crossing` | `doorway_line:B->A` | `3.0000` | `3.0000` | `0.0000` |
| `evt_0004` | `zone_entry` | `center_concourse` | `3.0000` | `3.0000` | `0.0000` |
| `evt_0005` | `zone_exit` | `entrance_area` | `3.0000` | `3.0000` | `0.0000` |
| `evt_0006` | `dwell_time_event` | `center_concourse` | `3.0000` | `4.5833` | `1.5833` |
| `evt_0007` | `line_crossing` | `aisle_split_line:B->A` | `3.0833` | `3.0833` | `0.0000` |
| `evt_0008` | `zone_entry` | `right_aisle` | `3.9167` | `3.9167` | `0.0000` |
| `evt_0009` | `dwell_time_event` | `right_aisle` | `3.9167` | `6.2500` | `2.3333` |
| `evt_0010` | `zone_exit` | `center_concourse` | `4.6667` | `4.6667` | `0.0000` |

---

## Query 4 — Tracks with Duration `>= 5.0` Seconds

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --tracks --min-duration 5.0
```

**Returned `4` tracks:**

| `track_id` | `start_frame` | `end_frame` | `duration_frames` | `duration_seconds` | `path_length_px` | `event_count` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3` | `180` | `249` | `70` | `5.8333` | `538.99` | `8` |
| `2` | `180` | `242` | `63` | `5.2500` | `356.26` | `6` |
| `5` | `339` | `401` | `63` | `5.2500` | `454.76` | `6` |
| `6` | `341` | `401` | `61` | `5.0833` | `419.16` | `6` |

---

## Query 5 — Visual Evidence Lookup for `evt_0002`

```bash
uv run --locked python cv_projects/10_searchable_outputs/scripts/search_events.py \
    --db cv_projects/10_searchable_outputs/outputs/people_detection/intelligence.db \
    --evidence-event-id evt_0002
```

- **Event:** `evt_0002` (`dwell_time_event`, `track_id=1`, `zone_id=entrance_area`, `duration=1.3334s`)
- **Relative Snapshot Path:** `snapshots/evt_0002_1.jpg` (`exists=True`, `66109` bytes)
- **Relative Crop Path:** `crops/evt_0002_1.jpg` (`exists=True`, `11438` bytes)
- **Evidence Valid:** `True`
