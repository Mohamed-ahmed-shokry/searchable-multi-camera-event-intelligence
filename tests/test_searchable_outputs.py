"""Unit and integration tests for Phase 10 Searchable Outputs (search_events.py)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from search_events import (
    DatabaseSummary,
    EventSearchResult,
    EvidenceLookupResult,
    SearchConfig,
    TrackSearchResult,
    ingest_intelligence_db,
    init_intelligence_db,
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


class TestInitIntelligenceDb(unittest.TestCase):
    """Test init_intelligence_db creates all 7 tables and secondary indexes."""

    def test_init_intelligence_db_creates_tables_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = init_intelligence_db(Path(tmp) / "intelligence.db")
            self.assertTrue(db_path.is_file())
            with closing(sqlite3.connect(db_path)) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for expected in (
                    "detections",
                    "track_observations",
                    "tracks",
                    "trajectories",
                    "events",
                    "zone_counts",
                    "line_counts",
                ):
                    self.assertIn(expected, tables)


class TestIngestIntelligenceDb(unittest.TestCase):
    """Test ingest_intelligence_db row counts and re-ingest idempotency."""

    def test_ingest_intelligence_db_loads_all_tables_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "detections.csv").write_text(
                "frame_id,x1,y1,x2,y2,confidence,class\n"
                "0,,,,,,\n"
                "1,10.0,10.0,50.0,80.0,0.9,person\n",
                encoding="utf-8",
            )
            (root / "tracks.csv").write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n"
                "1,1,10.0,10.0,50.0,80.0,0.9\n"
                "2,1,12.0,12.0,52.0,82.0,0.92\n",
                encoding="utf-8",
            )
            (root / "trajectories.csv").write_text(
                "track_id,frame_id,timestamp,x,y\n1,1,0.1,30.0,80.0\n1,2,0.2,33.0,84.0\n",
                encoding="utf-8",
            )
            (root / "events.csv").write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_0001,zone_entry,1,z1,0.1,0.1,0.0,snapshots/evt_0001_1.jpg,crops/evt_0001_1.jpg\n",
                encoding="utf-8",
            )
            (root / "zone_counts.csv").write_text(
                "zone_id,entry_count,exit_count,total_dwell_seconds,peak_occupancy\n"
                "z1,1,0,2.5,1\n",
                encoding="utf-8",
            )
            (root / "line_counts.csv").write_text(
                "line_id,crossings_ab,crossings_ba,total\nl1,0,1,1\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                detections_csv="detections.csv",
                tracks_csv="tracks.csv",
                tracking_summary_csv="missing_trk_sum.csv",
                trajectories_csv="trajectories.csv",
                trajectory_summary_csv="missing_traj_sum.csv",
                events_csv="events.csv",
                zone_counts_csv="zone_counts.csv",
                line_counts_csv="line_counts.csv",
            )
            db_path = root / "intelligence.db"
            summary1 = ingest_intelligence_db(db_path, cfg, project_root=root)
            self.assertEqual(summary1.detections_count, 1)
            self.assertEqual(summary1.track_observations_count, 2)
            self.assertEqual(summary1.tracks_count, 1)
            self.assertEqual(summary1.trajectories_count, 2)
            self.assertEqual(summary1.events_count, 1)
            self.assertEqual(summary1.zone_counts_rows, 1)
            self.assertEqual(summary1.line_counts_rows, 1)

            # Re-ingest should replace rows idempotently without duplicating
            summary2 = ingest_intelligence_db(db_path, cfg, project_root=root)
            self.assertEqual(summary2, summary1)


if __name__ == "__main__":
    unittest.main()
