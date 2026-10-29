# 05 — Zone and Line Reasoning

Spatial reasoning over extracted trajectories (`trajectories.csv`): point-in-polygon zone containment (`zone_enter` / `zone_exit`) and directional line crossings (`line_cross`, `A->B` vs `B->A`).

---

## Layout

```text
cv_projects/05_zone_line_reasoning/
  README.md
  notes.md
  configs/
    scene_config.yaml
  notebooks/
    01_define_and_explore_zones.ipynb
  scripts/
    reason_zones_lines.py
    define_scene_geometry.py
  outputs/                # gitignored generated artifacts
```
