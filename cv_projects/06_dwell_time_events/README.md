# 06 — Dwell-Time Events

Phase 6 implements stateful event lifecycle reasoning (`candidate -> active -> triggered -> resolved`) per `(track_id, zone_id)` with hysteresis / grace-period flicker absorption, track-loss-in-zone handling, per-stay debug traces (`event_debug_log/`), and visual dwell examples (`dwell_examples/`).
