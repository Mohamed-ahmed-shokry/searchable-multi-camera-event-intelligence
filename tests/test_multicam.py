"""Unit and CLI integration tests for Phase 11j multi-camera correlation analytics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from event_intelligence.multicam.engine import (
    infer_default_topology,
    load_camera_topology,
    save_camera_topology,
)
from event_intelligence.multicam.types import (
    CameraTopology,
    CameraTransitionRule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestCameraTopology(unittest.TestCase):
    """Tests for CameraTopology YAML I/O and default topology inference."""

    def test_save_and_load_camera_topology_roundtrip(self) -> None:
        topology = CameraTopology(
            name="campus_west",
            description="Lobby to corridor handoff",
            camera_time_offsets={"lobby": 0.0, "corridor": 1.5},
            transitions=(
                CameraTransitionRule(
                    from_camera="lobby",
                    to_camera="corridor",
                    min_transit_seconds=0.5,
                    max_transit_seconds=12.0,
                    from_zone="exit_door",
                    to_zone="north_entry",
                    expected_transit_seconds=3.0,
                    label="lobby_to_corridor",
                ),
            ),
            default_min_transit_seconds=0.5,
            default_max_transit_seconds=20.0,
            max_matches_per_track=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topology.yaml"
            save_camera_topology(topology, path)
            loaded = load_camera_topology(path)

        self.assertEqual(loaded.name, "campus_west")
        self.assertEqual(loaded.camera_time_offsets, {"lobby": 0.0, "corridor": 1.5})
        self.assertEqual(loaded.max_matches_per_track, 2)
        self.assertEqual(len(loaded.transitions), 1)
        rule = loaded.transitions[0]
        self.assertEqual(rule.from_camera, "lobby")
        self.assertEqual(rule.to_camera, "corridor")
        self.assertEqual(rule.from_zone, "exit_door")
        self.assertEqual(rule.to_zone, "north_entry")
        self.assertEqual(rule.expected_transit_seconds, 3.0)
        self.assertEqual(rule.label, "lobby_to_corridor")

    def test_load_default_repo_topology_config(self) -> None:
        loaded = load_camera_topology(PROJECT_ROOT / "configs" / "camera_topology.yaml")
        self.assertEqual(loaded.name, "default_site_topology")
        self.assertEqual(len(loaded.transitions), 2)

    def test_load_camera_topology_missing_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.yaml"
            with self.assertRaises(FileNotFoundError):
                load_camera_topology(missing)

            invalid = Path(tmp) / "bad.yaml"
            invalid.write_text("- not-a-mapping\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_camera_topology(invalid)

    def test_infer_default_topology_all_pairs(self) -> None:
        topo = infer_default_topology(
            ["cam_b", "cam_a", "cam_c", "cam_a"],
            min_transit_seconds=1.0,
            max_transit_seconds=15.0,
        )
        self.assertEqual(topo.name, "inferred_all_pairs")
        # 3 unique cameras -> 3 * 2 = 6 directed pairs
        self.assertEqual(len(topo.transitions), 6)
        pairs = {(r.from_camera, r.to_camera) for r in topo.transitions}
        self.assertIn(("cam_a", "cam_b"), pairs)
        self.assertNotIn(("cam_a", "cam_a"), pairs)


if __name__ == "__main__":
    unittest.main()
