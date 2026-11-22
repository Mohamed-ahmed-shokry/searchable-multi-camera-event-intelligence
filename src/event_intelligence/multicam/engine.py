"""Multi-camera spatial-temporal correlation engine and topology helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from event_intelligence.multicam.types import (
    CameraTopology,
    CameraTransitionRule,
)


def load_camera_topology(path: Path | str) -> CameraTopology:
    """Load a CameraTopology specification from a YAML file."""
    topology_path = Path(path)
    if not topology_path.exists():
        raise FileNotFoundError(f"Camera topology YAML not found: {topology_path}")
    raw = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    if raw is None:
        return CameraTopology()
    if not isinstance(raw, dict):
        raise TypeError(
            f"Camera topology YAML must contain a mapping, got {type(raw).__name__}"
        )
    return CameraTopology.from_dict(raw)


def save_camera_topology(topology: CameraTopology, path: Path | str) -> Path:
    """Write a CameraTopology specification to a YAML file."""
    topology_path = Path(path)
    topology_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path.write_text(
        yaml.safe_dump(topology.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return topology_path


def infer_default_topology(
    camera_ids: Sequence[str],
    *,
    min_transit_seconds: float = 0.0,
    max_transit_seconds: float = 30.0,
    max_matches_per_track: int = 1,
) -> CameraTopology:
    """Build a bidirectional all-pairs topology for cameras when no YAML is provided."""
    unique_cameras = sorted({str(cam) for cam in camera_ids if str(cam)})
    rules: list[CameraTransitionRule] = []
    for src in unique_cameras:
        for dst in unique_cameras:
            if src == dst:
                continue
            rules.append(
                CameraTransitionRule(
                    from_camera=src,
                    to_camera=dst,
                    min_transit_seconds=min_transit_seconds,
                    max_transit_seconds=max_transit_seconds,
                    label=f"{src} -> {dst}",
                )
            )
    return CameraTopology(
        name="inferred_all_pairs",
        description=(
            f"Auto-inferred all-pairs topology across {len(unique_cameras)} camera(s)."
        ),
        transitions=tuple(rules),
        default_min_transit_seconds=min_transit_seconds,
        default_max_transit_seconds=max_transit_seconds,
        max_matches_per_track=max_matches_per_track,
    )
