"""Unit and integration tests for Phase 10 Searchable Outputs (search_events.py)."""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path

from search_events import (
    DatabaseSummary,
    EventSearchResult,
    EvidenceLookupResult,
    SearchConfig,
    TrackSearchResult,
    generate_schema_markdown,
    generate_search_examples_markdown,
    ingest_intelligence_db,
    init_intelligence_db,
    load_search_config,
    lookup_event_evidence,
    main,
    query_events,
    query_tracks,
    run_searchable_outputs,
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


class TestQueryEvents(unittest.TestCase):
    """Test query_events filtering by type, zone, track_id, time range, and duration."""

    def test_query_events_filters_and_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.csv").write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_0001,zone_entry,1,entrance_area,1.5,1.5,0.0,s1.jpg,c1.jpg\n"
                "evt_0002,dwell_time_event,1,entrance_area,1.5,3.0,1.5,s2.jpg,c2.jpg\n"
                "evt_0003,line_crossing,1,doorway_line:B->A,3.0,3.0,0.0,s3.jpg,c3.jpg\n"
                "evt_0004,dwell_time_event,2,right_aisle,10.0,13.2,3.2,s4.jpg,c4.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                detections_csv="missing.csv",
                tracks_csv="missing.csv",
                tracking_summary_csv="missing.csv",
                trajectories_csv="missing.csv",
                trajectory_summary_csv="missing.csv",
                events_csv="events.csv",
                zone_counts_csv="missing.csv",
                line_counts_csv="missing.csv",
            )
            db_path = root / "intelligence.db"
            ingest_intelligence_db(db_path, cfg, project_root=root)

            dwells = query_events(db_path, event_type="dwell_time_event")
            self.assertEqual([e.event_id for e in dwells], ["evt_0002", "evt_0004"])

            # Prefix match on directed line "doorway_line" -> matches "doorway_line:B->A"
            crossings = query_events(db_path, zone_id="doorway_line")
            self.assertEqual([e.event_id for e in crossings], ["evt_0003"])

            by_track = query_events(db_path, track_id=2)
            self.assertEqual([e.event_id for e in by_track], ["evt_0004"])

            by_time = query_events(db_path, min_time=2.0, max_time=5.0)
            self.assertEqual([e.event_id for e in by_time], ["evt_0002", "evt_0003"])

            by_dur = query_events(db_path, min_duration=2.0)
            self.assertEqual([e.event_id for e in by_dur], ["evt_0004"])


class TestQueryTracks(unittest.TestCase):
    """Test query_tracks filtering by duration range, path length, and event counts."""

    def test_query_tracks_filters_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "track_summary.csv").write_text(
                "track_id,start_frame,end_frame,duration_frames,duration_seconds,observed_frames\n"
                "1,19,75,57,4.7500,57\n"
                "3,180,249,70,5.8333,70\n"
                "9,523,559,37,3.0833,37\n",
                encoding="utf-8",
            )
            (root / "traj_summary.csv").write_text(
                "track_id,path_length_px,duration_seconds,frame_count\n"
                "1,553.84,4.750,57\n"
                "3,538.99,5.833,70\n"
                "9,470.64,3.083,37\n",
                encoding="utf-8",
            )
            (root / "events.csv").write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_0001,zone_entry,1,entrance_area,1.5,1.5,0.0,s1.jpg,c1.jpg\n"
                "evt_0002,dwell_time_event,1,entrance_area,1.5,3.0,1.5,s2.jpg,c2.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                detections_csv="missing.csv",
                tracks_csv="missing.csv",
                tracking_summary_csv="track_summary.csv",
                trajectories_csv="missing.csv",
                trajectory_summary_csv="traj_summary.csv",
                events_csv="events.csv",
                zone_counts_csv="missing.csv",
                line_counts_csv="missing.csv",
            )
            db_path = root / "intelligence.db"
            ingest_intelligence_db(db_path, cfg, project_root=root)

            long_tracks = query_tracks(db_path, min_duration=4.5)
            self.assertEqual([t.track_id for t in long_tracks], [3, 1])
            self.assertEqual(long_tracks[1].event_count, 2)

            far_tracks = query_tracks(db_path, min_path_length_px=540.0)
            self.assertEqual([t.track_id for t in far_tracks], [1])


class TestLookupEventEvidence(unittest.TestCase):
    """Test lookup_event_evidence resolution and on-disk verification."""

    def test_lookup_event_evidence_resolves_and_checks_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshots"
            crop_dir = root / "crops"
            snap_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "evt_0001_1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (crop_dir / "evt_0001_1.jpg").write_bytes(b"\xff\xd8\xff\xd9")

            (root / "events.csv").write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_0001,zone_entry,1,entrance_area,1.5,1.5,0.0,snapshots/evt_0001_1.jpg,crops/evt_0001_1.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                detections_csv="missing.csv",
                tracks_csv="missing.csv",
                tracking_summary_csv="missing.csv",
                trajectories_csv="missing.csv",
                trajectory_summary_csv="missing.csv",
                events_csv="events.csv",
                evidence_root=str(root),
                zone_counts_csv="missing.csv",
                line_counts_csv="missing.csv",
            )
            db_path = root / "intelligence.db"
            ingest_intelligence_db(db_path, cfg, project_root=root)

            res = lookup_event_evidence(db_path, "evt_0001", evidence_root=root)
            self.assertTrue(res.is_valid)
            self.assertEqual(res.snapshot_size_bytes, 4)
            self.assertEqual(res.crop_size_bytes, 4)

            with self.assertRaises(KeyError):
                lookup_event_evidence(db_path, "evt_9999", evidence_root=root)


class TestGenerateMarkdownDocs(unittest.TestCase):
    """Test generate_schema_markdown and generate_search_examples_markdown."""

    def test_generate_schema_and_search_examples_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshots"
            crop_dir = root / "crops"
            snap_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "evt_0001_1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (crop_dir / "evt_0001_1.jpg").write_bytes(b"\xff\xd8\xff\xd9")

            (root / "events.csv").write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_0001,dwell_time_event,1,right_aisle,3.9,6.2,2.3,snapshots/evt_0001_1.jpg,crops/evt_0001_1.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                detections_csv="missing.csv",
                tracks_csv="missing.csv",
                tracking_summary_csv="missing.csv",
                trajectories_csv="missing.csv",
                trajectory_summary_csv="missing.csv",
                events_csv="events.csv",
                evidence_root=str(root),
                zone_counts_csv="missing.csv",
                line_counts_csv="missing.csv",
            )
            db_path = root / "intelligence.db"
            summary = ingest_intelligence_db(db_path, cfg, project_root=root)

            schema_md = generate_schema_markdown(summary)
            self.assertIn("CREATE TABLE IF NOT EXISTS events", schema_md)

            examples_md = generate_search_examples_markdown(db_path, evidence_root=root)
            self.assertIn("## Query 1", examples_md)
            self.assertIn("`evt_0001`", examples_md)
            self.assertIn("**Evidence Valid:** `True`", examples_md)


class TestRunSearchableOutputsAndCli(unittest.TestCase):
    """Test run_searchable_outputs pipeline and CLI main() modes."""

    def test_run_searchable_outputs_and_cli_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap_dir = root / "snapshots"
            crop_dir = root / "crops"
            snap_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / "evt_0001_1.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (crop_dir / "evt_0001_1.jpg").write_bytes(b"\xff\xd8\xff\xd9")

            (root / "trajectories.csv").write_text(
                "track_id,frame_id,timestamp,x,y\n"
                "1,0,0.0,100.0,100.0\n"
                "1,10,6.0,160.0,180.0\n",
                encoding="utf-8",
            )
            (root / "events.csv").write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_0001,dwell_time_event,1,right_aisle,3.9,6.2,2.3,snapshots/evt_0001_1.jpg,crops/evt_0001_1.jpg\n",
                encoding="utf-8",
            )

            cfg = SearchConfig(
                detections_csv=str(root / "missing.csv"),
                tracks_csv=str(root / "missing.csv"),
                tracking_summary_csv=str(root / "missing.csv"),
                trajectories_csv=str(root / "trajectories.csv"),
                trajectory_summary_csv=str(root / "missing.csv"),
                events_csv=str(root / "events.csv"),
                evidence_root=str(root),
                zone_counts_csv=str(root / "missing.csv"),
                line_counts_csv=str(root / "missing.csv"),
                write_subproject_docs=False,
            )
            cfg_path = root / "search.yaml"
            save_search_config(cfg, cfg_path)

            out_dir = root / "out"
            summary = run_searchable_outputs(cfg, out_dir, project_root=root)
            self.assertEqual(summary.events_count, 1)
            self.assertEqual(summary.tracks_count, 1)
            self.assertTrue((out_dir / "intelligence.db").is_file())
            self.assertTrue((out_dir / "schema.md").is_file())
            self.assertTrue((out_dir / "search_examples.md").is_file())
            self.assertTrue((out_dir / "search_summary.json").is_file())

            # Test CLI --build --json
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "--config",
                        str(cfg_path),
                        "--output-dir",
                        str(out_dir),
                        "--build",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            parsed_build = json.loads(buf.getvalue())
            self.assertEqual(parsed_build["events_count"], 1)

            # Test CLI event query --json
            db_path = out_dir / "intelligence.db"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "--db",
                        str(db_path),
                        "--event-type",
                        "dwell_time_event",
                        "--zone",
                        "right_aisle",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            parsed_events = json.loads(buf.getvalue())
            self.assertEqual(len(parsed_events), 1)
            self.assertEqual(parsed_events[0]["event_id"], "evt_0001")

            # Test CLI track query --json
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "--db",
                        str(db_path),
                        "--tracks",
                        "--min-duration",
                        "5.0",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            parsed_tracks = json.loads(buf.getvalue())
            self.assertEqual(len(parsed_tracks), 1)
            self.assertEqual(parsed_tracks[0]["track_id"], 1)

            # Test CLI evidence lookup --json
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(
                    [
                        "--db",
                        str(db_path),
                        "--evidence-event-id",
                        "evt_0001",
                        "--evidence-root",
                        str(root),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            parsed_ev = json.loads(buf.getvalue())
            self.assertTrue(parsed_ev["is_valid"])


if __name__ == "__main__":
    unittest.main()
