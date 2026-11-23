"""Unit and CLI integration tests for Phase 11j multi-camera correlation analytics."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path

from event_intelligence.multicam.__main__ import main as multicam_main
from event_intelligence.multicam.engine import (
    analyze_multi_camera_correlation,
    build_cross_camera_journeys,
    compute_co_activity_windows,
    correlate_cross_camera_transitions,
    extract_track_endpoints,
    infer_default_topology,
    load_camera_topology,
    save_camera_topology,
    summarize_camera_pair_flows,
)
from event_intelligence.multicam.report import (
    generate_correlation_markdown,
    write_correlation_artifacts,
)
from event_intelligence.multicam.types import (
    CameraTopology,
    CameraTransitionRule,
)
from event_intelligence.storage.search_events import (
    SCHEMA_DDL,
    insert_cross_camera_correlation,
    query_cross_camera_journeys,
    query_cross_camera_transitions,
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


def _build_multicam_db(db_path: Path) -> Path:
    """Create a 3-camera synthetic intelligence.db with known handoff chains."""
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.executescript(SCHEMA_DDL)
        conn.executemany(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "lobby",
                    "lobby.mp4",
                    10.0,
                    640,
                    480,
                    300,
                    "2026-11-22T10:00:00+00:00",
                ),
                (
                    "corridor",
                    "corridor.mp4",
                    10.0,
                    640,
                    480,
                    300,
                    "2026-11-22T10:00:00+00:00",
                ),
                (
                    "exit_hall",
                    "exit_hall.mp4",
                    10.0,
                    640,
                    480,
                    300,
                    "2026-11-22T10:00:00+00:00",
                ),
            ],
        )
        # Journey A: lobby:1 (0..4s) -> corridor:10 (6..11s) -> exit_hall:20 (13..18s)
        # Concurrent track: lobby:2 (6..10s) -> creates simultaneous surge with corridor:10 in [5, 10)
        conn.executemany(
            "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "lobby", 0, 40, 41, 4.0, 3, 120.0),
                (2, "lobby", 60, 100, 41, 4.0, 3, 95.0),
                (10, "corridor", 60, 110, 51, 5.0, 3, 150.0),
                (20, "exit_hall", 130, 180, 51, 5.0, 3, 110.0),
            ],
        )
        conn.executemany(
            "INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "lobby", 0, 0.0, 100.0, 200.0),
                (1, "lobby", 20, 2.0, 150.0, 210.0),
                (1, "lobby", 40, 4.0, 200.0, 220.0),
                (2, "lobby", 60, 6.0, 110.0, 190.0),
                (2, "lobby", 80, 8.0, 130.0, 195.0),
                (2, "lobby", 100, 10.0, 160.0, 205.0),
                (10, "corridor", 60, 6.0, 50.0, 120.0),
                (10, "corridor", 85, 8.5, 120.0, 125.0),
                (10, "corridor", 110, 11.0, 210.0, 130.0),
                (20, "exit_hall", 130, 13.0, 40.0, 100.0),
                (20, "exit_hall", 155, 15.5, 90.0, 105.0),
                (20, "exit_hall", 180, 18.0, 160.0, 110.0),
            ],
        )
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "ev_l1",
                    "lobby",
                    "zone_dwell",
                    1,
                    "lobby_exit",
                    1.0,
                    3.8,
                    2.8,
                    "",
                    "",
                    "",
                    "",
                ),
                (
                    "ev_c10",
                    "corridor",
                    "zone_dwell",
                    10,
                    "corridor_entry",
                    6.2,
                    10.5,
                    4.3,
                    "",
                    "",
                    "",
                    "",
                ),
                (
                    "ev_e20",
                    "exit_hall",
                    "zone_dwell",
                    20,
                    "exit_gate",
                    13.5,
                    17.5,
                    4.0,
                    "",
                    "",
                    "",
                    "",
                ),
            ],
        )
        conn.commit()
    return db_path


class TestMultiCameraCorrelationEngine(unittest.TestCase):
    """Tests for endpoint extraction, transition matching, journeys, and surges."""

    def test_extract_track_endpoints_with_clock_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_multicam_db(Path(tmp) / "intelligence.db")
            endpoints = extract_track_endpoints(
                db_path,
                camera_time_offsets={"corridor": 2.0},
            )

        self.assertEqual(len(endpoints), 4)
        by_key = {(ep.camera_id, ep.track_id): ep for ep in endpoints}
        self.assertAlmostEqual(by_key[("lobby", 1)].start_time, 0.0)
        self.assertAlmostEqual(by_key[("lobby", 1)].end_time, 4.0)
        self.assertEqual(by_key[("lobby", 1)].exit_zones, ("lobby_exit",))
        self.assertAlmostEqual(by_key[("corridor", 10)].start_time, 8.0)
        self.assertAlmostEqual(by_key[("corridor", 10)].end_time, 13.0)

    def test_correlate_transitions_and_stitch_three_camera_journey(self) -> None:
        topology = CameraTopology(
            name="linear_3cam",
            transitions=(
                CameraTransitionRule(
                    from_camera="lobby",
                    to_camera="corridor",
                    min_transit_seconds=0.5,
                    max_transit_seconds=5.0,
                    from_zone="lobby_exit",
                    to_zone="corridor_entry",
                    expected_transit_seconds=2.0,
                    label="lobby_to_corridor",
                ),
                CameraTransitionRule(
                    from_camera="corridor",
                    to_camera="exit_hall",
                    min_transit_seconds=0.5,
                    max_transit_seconds=5.0,
                    expected_transit_seconds=2.0,
                    label="corridor_to_exit",
                ),
            ),
            max_matches_per_track=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_multicam_db(Path(tmp) / "intelligence.db")
            endpoints = extract_track_endpoints(db_path)
            transitions = correlate_cross_camera_transitions(endpoints, topology)
            journeys = build_cross_camera_journeys(transitions)
            flows = summarize_camera_pair_flows(transitions)

        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0].from_camera, "lobby")
        self.assertEqual(transitions[0].to_camera, "corridor")
        self.assertAlmostEqual(transitions[0].transit_seconds, 2.0)
        self.assertGreaterEqual(transitions[0].confidence, 0.95)

        self.assertEqual(len(journeys), 1)
        self.assertEqual(
            journeys[0].cameras_visited,
            ("lobby", "corridor", "exit_hall"),
        )
        self.assertEqual(len(journeys[0].hops), 2)
        self.assertAlmostEqual(journeys[0].total_span_seconds, 9.0)

        self.assertEqual(len(flows), 2)
        self.assertEqual(flows[0].transition_count, 1)

    def test_co_activity_windows_and_full_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_multicam_db(Path(tmp) / "intelligence.db")
            windows = compute_co_activity_windows(db_path, bucket_seconds=5.0)
            report = analyze_multi_camera_correlation(db_path, bucket_seconds=5.0)

        surges = [w for w in windows if w.is_simultaneous_surge]
        self.assertGreaterEqual(len(surges), 1)
        # Bucket [5, 10) has lobby:2 and corridor:10 active simultaneously
        bucket_1 = next(w for w in windows if w.bucket_index == 1)
        self.assertEqual(bucket_1.active_cameras, ("corridor", "lobby"))
        self.assertEqual(bucket_1.total_active_tracks, 2)
        self.assertTrue(bucket_1.is_simultaneous_surge)

        self.assertEqual(report.topology_name, "inferred_all_pairs")
        self.assertEqual(report.cameras, ("corridor", "exit_hall", "lobby"))
        self.assertEqual(report.track_endpoints_count, 4)
        self.assertGreaterEqual(len(report.transitions), 2)
        self.assertEqual(report.peak_simultaneous_cameras, 3)
        self.assertIn("Analyzed 3 camera(s)", report.summary_text)


class TestMultiCameraCorrelationReporting(unittest.TestCase):
    """Tests for Markdown rendering and JSON/CSV artifact exports."""

    def test_generate_markdown_and_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_multicam_db(tmp_path / "intelligence.db")
            report = analyze_multi_camera_correlation(db_path, bucket_seconds=5.0)
            md = generate_correlation_markdown(report)
            artifacts = write_correlation_artifacts(report, tmp_path / "out")

            self.assertIn("# Multi-Camera Spatial-Temporal Correlation Report", md)
            self.assertIn("## Camera-to-Camera Transition Flow Matrix", md)
            self.assertIn("## Stitched Multi-Hop Journeys", md)
            self.assertIn("## Synchronized Co-Activity Windows", md)

            for key in (
                "json",
                "markdown",
                "transitions_csv",
                "pair_flows_csv",
                "co_activity_windows_csv",
            ):
                self.assertTrue(artifacts[key].exists(), f"Missing artifact: {key}")

            payload = json.loads(artifacts["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["camera_count"], 3)
            self.assertEqual(payload["transition_count"], len(report.transitions))

            with artifacts["transitions_csv"].open(encoding="utf-8", newline="") as f:
                tr_rows = list(csv.DictReader(f))
            self.assertEqual(len(tr_rows), len(report.transitions))

            with artifacts["pair_flows_csv"].open(encoding="utf-8", newline="") as f:
                flow_rows = list(csv.DictReader(f))
            self.assertEqual(len(flow_rows), len(report.pair_flows))

            with artifacts["co_activity_windows_csv"].open(
                encoding="utf-8", newline=""
            ) as f:
                win_rows = list(csv.DictReader(f))
            self.assertEqual(len(win_rows), len(report.co_activity_windows))


class TestMultiCameraCli(unittest.TestCase):
    """CLI integration tests for python -m event_intelligence.multicam."""

    def test_cli_json_and_output_dir_from_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run_demo"
            run_dir.mkdir()
            _build_multicam_db(run_dir / "intelligence.db")
            out_dir = tmp_path / "artifacts"

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = multicam_main(
                    [
                        "--db",
                        str(run_dir),
                        "--output-dir",
                        str(out_dir),
                        "--json",
                    ]
                )

            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["camera_count"], 3)
            self.assertIn("artifacts", payload)
            self.assertTrue((out_dir / "cross_camera_transitions.csv").exists())
            self.assertTrue((out_dir / "multicam_correlation_report.md").exists())

    def test_cli_with_explicit_topology_and_text_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = _build_multicam_db(tmp_path / "intelligence.db")
            topo_path = save_camera_topology(
                CameraTopology(
                    name="explicit_test",
                    transitions=(
                        CameraTransitionRule(
                            from_camera="lobby",
                            to_camera="corridor",
                            min_transit_seconds=0.5,
                            max_transit_seconds=5.0,
                            expected_transit_seconds=2.0,
                        ),
                    ),
                ),
                tmp_path / "topo.yaml",
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = multicam_main(
                    [
                        "--db",
                        str(db_path),
                        "--topology",
                        str(topo_path),
                    ]
                )

            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("# Multi-Camera Spatial-Temporal Correlation Report", text)
            self.assertIn("`explicit_test`", text)
            self.assertIn("`lobby#1`", text)

    def test_cli_missing_db_exits_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.db"
            with self.assertRaises(SystemExit) as ctx:
                multicam_main(["--db", str(missing)])
            self.assertEqual(ctx.exception.code, 2)


class TestCrossCameraCorrelationPersistence(unittest.TestCase):
    """Tests for SQLite persistence, legacy migration, and correlation queries."""

    def test_insert_and_query_transitions_and_journeys_with_filters(self) -> None:
        topology = CameraTopology(
            name="linear_3cam",
            transitions=(
                CameraTransitionRule(
                    from_camera="lobby",
                    to_camera="corridor",
                    min_transit_seconds=0.5,
                    max_transit_seconds=5.0,
                    from_zone="lobby_exit",
                    to_zone="corridor_entry",
                    expected_transit_seconds=2.0,
                    label="lobby_to_corridor",
                ),
                CameraTransitionRule(
                    from_camera="corridor",
                    to_camera="exit_hall",
                    min_transit_seconds=0.5,
                    max_transit_seconds=5.0,
                    expected_transit_seconds=2.0,
                    label="corridor_to_exit",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = _build_multicam_db(Path(tmp) / "intelligence.db")
            report = analyze_multi_camera_correlation(db_path, topology=topology)
            tr_count, jr_count = insert_cross_camera_correlation(db_path, report)

            self.assertEqual(tr_count, 2)
            self.assertEqual(jr_count, 1)

            all_tr = query_cross_camera_transitions(db_path)
            self.assertEqual(len(all_tr), 2)
            self.assertEqual(all_tr[0].topology_name, "linear_3cam")

            lobby_tr = query_cross_camera_transitions(
                db_path,
                from_camera="lobby",
                to_camera="corridor",
                min_confidence=0.9,
            )
            self.assertEqual(len(lobby_tr), 1)
            self.assertEqual(lobby_tr[0].from_track_id, 1)
            self.assertEqual(lobby_tr[0].to_track_id, 10)

            track_tr = query_cross_camera_transitions(db_path, track_id=20)
            self.assertEqual(len(track_tr), 1)
            self.assertEqual(track_tr[0].to_camera, "exit_hall")

            journeys = query_cross_camera_journeys(
                db_path,
                camera_id="corridor",
                min_hops=2,
                min_confidence=0.5,
            )
            self.assertEqual(len(journeys), 1)
            self.assertEqual(
                journeys[0].cameras_visited,
                ("lobby", "corridor", "exit_hall"),
            )
            self.assertEqual(len(journeys[0].hops), 2)

            none_jr = query_cross_camera_journeys(db_path, min_hops=3)
            self.assertEqual(none_jr, [])

    def test_legacy_db_graceful_query_and_nondestructive_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "legacy.db"
            with closing(sqlite3.connect(str(db_path))) as conn:
                conn.execute(
                    "CREATE TABLE cameras (camera_id TEXT PRIMARY KEY, source_name TEXT)"
                )
                conn.execute("INSERT INTO cameras VALUES ('cam_a', 'a.mp4')")
                conn.commit()

            # Legacy DB lacks correlation tables -> queries return empty cleanly
            self.assertEqual(query_cross_camera_transitions(db_path), [])
            self.assertEqual(query_cross_camera_journeys(db_path), [])

            # Non-destructive migration preserves existing cameras table
            tr_count, jr_count = insert_cross_camera_correlation(
                db_path,
                type(
                    "_EmptyReport",
                    (),
                    {"topology_name": "empty", "transitions": (), "journeys": ()},
                )(),
            )
            self.assertEqual((tr_count, jr_count), (0, 0))
            with closing(sqlite3.connect(str(db_path))) as conn:
                cams = conn.execute("SELECT camera_id FROM cameras").fetchall()
            self.assertEqual(cams, [("cam_a",)])


if __name__ == "__main__":
    unittest.main()
