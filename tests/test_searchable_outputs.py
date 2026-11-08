"""Unit and integration tests for Phase 10 Searchable Outputs (search_events.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from search_events import (
    DatabaseSummary,
    EventSearchResult,
    EvidenceLookupResult,
    SearchConfig,
    TrackSearchResult,
    load_search_config,
    save_search_config,
)


class TestSearchDataclasses(unittest.TestCase):
    """Test validation and serialization of Phase 10 search dataclasses."""

    def test_search_config_defaults_and_validation(self) -> None:
        cfg = SearchConfig()
        self.assertEqual(cfg.source_name, "people-detection.mp4")
        self.assertEqual(cfg.default_query_limit, 50)
        with self.assertRaises(ValueError):
            SearchConfig(source_name="")
        with self.assertRaises(ValueError):
            SearchConfig(default_query_limit=0)

    def test_event_and_track_search_results_and_evidence_lookup(self) -> None:
        ev = EventSearchResult(
            event_id="evt_0002",
            event_type="dwell_time_event",
            track_id=1,
            zone_id="entrance_area",
            start_time=1.5833,
            end_time=2.9167,
            duration_seconds=1.3334,
            snapshot_path="snapshots/evt_0002_1.jpg",
            crop_path="crops/evt_0002_1.jpg",
        )
        self.assertEqual(ev.to_dict()["event_id"], "evt_0002")
        with self.assertRaises(ValueError):
            EventSearchResult("", "dwell_time_event", 1, "z", 0.0, 1.0, 1.0, "s", "c")

        trk = TrackSearchResult(
            track_id=3,
            start_frame=180,
            end_frame=249,
            duration_frames=70,
            duration_seconds=5.8333,
            observed_frames=70,
            path_length_px=538.99,
            event_count=6,
        )
        self.assertEqual(trk.to_dict()["path_length_px"], 538.99)
        with self.assertRaises(ValueError):
            TrackSearchResult(3, 0, 10, 0, 1.0, 1, 10.0)

        ev_lookup = EvidenceLookupResult(
            event=ev,
            resolved_snapshot_path="/tmp/snapshots/evt_0002_1.jpg",
            resolved_crop_path="/tmp/crops/evt_0002_1.jpg",
            snapshot_exists=True,
            crop_exists=True,
            snapshot_size_bytes=1024,
            crop_size_bytes=512,
        )
        self.assertTrue(ev_lookup.is_valid)
        self.assertTrue(ev_lookup.to_dict()["is_valid"])

        summary = DatabaseSummary(
            db_path="intelligence.db",
            source_name="people-detection.mp4",
            detections_count=418,
            track_observations_count=409,
            tracks_count=7,
            trajectories_count=409,
            events_count=48,
            zone_counts_rows=3,
            line_counts_rows=2,
        )
        self.assertEqual(summary.to_dict()["events_count"], 48)


class TestSearchConfigYaml(unittest.TestCase):
    """Test load_search_config and save_search_config YAML serialization."""

    def test_search_yaml_round_trip_and_errors(self) -> None:
        self.assertEqual(load_search_config(None), SearchConfig())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = SearchConfig(source_name="custom.mp4", default_query_limit=25)
            yaml_path = save_search_config(cfg, root / "search.yaml")
            loaded = load_search_config(yaml_path)
            self.assertEqual(loaded, cfg)

            empty = root / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(load_search_config(empty), SearchConfig())

            bad = root / "bad.yaml"
            bad.write_text("- item\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_search_config(bad)

            with self.assertRaises(FileNotFoundError):
                load_search_config(root / "missing.yaml")


if __name__ == "__main__":
    unittest.main()
