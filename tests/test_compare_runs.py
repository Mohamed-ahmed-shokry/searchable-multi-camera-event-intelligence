"""Unit and integration tests for Phase 11h multi-run comparison and baseline analytics."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import cv2
import yaml

from event_intelligence.compare import (
    analyze_multi_run_trends,
    compare_runs,
    compute_insight_deltas,
    compute_line_deltas,
    compute_spatial_density_diff,
    compute_temporal_bucket_deltas,
    compute_zone_deltas,
    diff_run_configs,
    discover_runs,
    generate_comparison_markdown,
    generate_trends_markdown,
    load_run_snapshot,
    write_comparison_artifacts,
    write_trends_artifacts,
)
from event_intelligence.compare.__main__ import main as compare_main
from event_intelligence.compare.types import (
    ConfigDiffEntry,
    InsightTypeDelta,
    LineMetricDelta,
    MultiRunTrendReport,
    RunBaselineAnomaly,
    RunComparisonReport,
    RunSnapshot,
    SpatialShiftSummary,
    TemporalBucketDelta,
    ZoneMetricDelta,
)
from event_intelligence.storage.search_events import (
    CameraMetadata,
    SearchConfig,
    ingest_intelligence_db,
    insert_insights,
)


def _create_synthetic_run(
    run_dir: Path,
    *,
    camera_id: str = "cam_main",
    trajectories_rows: list[tuple[int, int, float, float, float]] | None = None,
    events_rows: list[tuple[str, str, int, str, float, float, float]] | None = None,
    zone_counts_rows: list[tuple[str, int, int, float, int]] | None = None,
    line_counts_rows: list[tuple[str, int, int, int]] | None = None,
    insights: list[dict[str, Any]] | None = None,
    run_config: dict[str, Any] | None = None,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Create a minimal synthetic run directory with intelligence.db and run_config.yaml."""
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_csv = run_dir / "trajectories.csv"
    traj_lines = ["track_id,frame_id,timestamp,x,y"]
    for tid, fid, ts, x, y in trajectories_rows or []:
        traj_lines.append(f"{tid},{fid},{ts},{x},{y}")
    traj_csv.write_text("\n".join(traj_lines) + "\n", encoding="utf-8")

    ev_csv = run_dir / "events.csv"
    ev_lines = [
        "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path"
    ]
    for eid, etype, tid, zid, st, et, dur in events_rows or []:
        ev_lines.append(f"{eid},{etype},{tid},{zid},{st},{et},{dur},snap.jpg,crop.jpg")
    ev_csv.write_text("\n".join(ev_lines) + "\n", encoding="utf-8")

    zc_csv = run_dir / "zone_counts.csv"
    zc_lines = ["zone_id,entry_count,exit_count,total_dwell_seconds,peak_occupancy"]
    for zid, ent, ex, dw, pk in zone_counts_rows or []:
        zc_lines.append(f"{zid},{ent},{ex},{dw},{pk}")
    zc_csv.write_text("\n".join(zc_lines) + "\n", encoding="utf-8")

    lc_csv = run_dir / "line_counts.csv"
    lc_lines = ["line_id,crossings_ab,crossings_ba,total"]
    for lid, ab, ba, tot in line_counts_rows or []:
        lc_lines.append(f"{lid},{ab},{ba},{tot}")
    lc_csv.write_text("\n".join(lc_lines) + "\n", encoding="utf-8")

    db_path = run_dir / "intelligence.db"
    cfg = SearchConfig(
        source_name=f"{camera_id}.mp4",
        camera_id=camera_id,
        trajectories_csv=str(traj_csv),
        events_csv=str(ev_csv),
        zone_counts_csv=str(zc_csv),
        line_counts_csv=str(lc_csv),
        detections_csv="",
        tracks_csv="",
        tracking_summary_csv="",
        trajectory_summary_csv="",
    )
    ingest_intelligence_db(
        db_path,
        cfg,
        project_root=run_dir,
        clear_existing=True,
        camera_metadata=CameraMetadata(
            camera_id=camera_id,
            source_name=f"{camera_id}.mp4",
            fps=10.0,
            width=width,
            height=height,
            total_frames=100,
        ),
    )
    if insights:
        insert_insights(db_path, insights)
    if run_config is not None:
        (run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(run_config, sort_keys=False),
            encoding="utf-8",
        )
    return run_dir


class TestCompareDomainTypes(unittest.TestCase):
    """Validate compare domain dataclasses, delta properties, and serialization."""

    def test_run_snapshot_validation_and_to_dict(self) -> None:
        snap = RunSnapshot(
            run_id="run_a",
            run_dir="/tmp/run_a",
            db_path="/tmp/run_a/intelligence.db",
            cameras_count=1,
            detections_count=100,
            tracks_count=10,
            trajectories_count=95,
            events_count=8,
            insights_count=2,
            high_critical_insights_count=1,
            total_dwell_seconds=15.5,
            mean_track_duration_seconds=4.2,
            mean_path_length_px=210.4,
            config_snapshot={"confidence_threshold": 0.4},
        )
        d = snap.to_dict()
        self.assertEqual(d["run_id"], "run_a")
        self.assertEqual(d["high_critical_insights_count"], 1)
        self.assertEqual(d["config_snapshot"]["confidence_threshold"], 0.4)

        with self.assertRaises(ValueError):
            RunSnapshot(
                run_id="",
                run_dir="/tmp/x",
                db_path="/tmp/x/intelligence.db",
                cameras_count=1,
                detections_count=0,
                tracks_count=0,
                trajectories_count=0,
                events_count=0,
                insights_count=0,
                high_critical_insights_count=0,
                total_dwell_seconds=0.0,
                mean_track_duration_seconds=0.0,
                mean_path_length_px=0.0,
            )

        with self.assertRaises(ValueError):
            RunSnapshot(
                run_id="bad_neg",
                run_dir="/tmp/x",
                db_path="/tmp/x/intelligence.db",
                cameras_count=1,
                detections_count=-1,
                tracks_count=0,
                trajectories_count=0,
                events_count=0,
                insights_count=0,
                high_critical_insights_count=0,
                total_dwell_seconds=0.0,
                mean_track_duration_seconds=0.0,
                mean_path_length_px=0.0,
            )

    def test_delta_properties_and_comparison_report_serialization(self) -> None:
        cfg_diff = ConfigDiffEntry("confidence_threshold", 0.4, 0.3)
        self.assertEqual(cfg_diff.to_dict()["key"], "confidence_threshold")
        with self.assertRaises(ValueError):
            ConfigDiffEntry("", 1, 2)

        z_delta = ZoneMetricDelta(
            camera_id="cam1",
            zone_id="lobby",
            baseline_entries=5,
            candidate_entries=8,
            baseline_exits=4,
            candidate_exits=7,
            baseline_dwell_seconds=10.0,
            candidate_dwell_seconds=16.5,
            baseline_peak_occupancy=2,
            candidate_peak_occupancy=4,
        )
        self.assertEqual(z_delta.delta_entries, 3)
        self.assertEqual(z_delta.delta_dwell_seconds, 6.5)
        self.assertEqual(z_delta.delta_peak_occupancy, 2)

        l_delta = LineMetricDelta("cam1", "door", 3, 5, 2, 1, 5, 6)
        self.assertEqual(l_delta.delta_ab, 2)
        self.assertEqual(l_delta.delta_ba, -1)
        self.assertEqual(l_delta.delta_total, 1)

        i_delta = InsightTypeDelta("loitering", 1, 3, 0, 2)
        self.assertEqual(i_delta.delta_count, 2)
        self.assertEqual(i_delta.delta_high_critical, 2)

        t_delta = TemporalBucketDelta(0, 0.0, 5.0, 2, 4, 3, 5, 0, 1)
        self.assertEqual(t_delta.delta_active_tracks, 2)
        self.assertEqual(t_delta.delta_events, 2)
        self.assertEqual(t_delta.delta_insights, 1)

        shift = SpatialShiftSummary(
            grid_shape=(27, 48),
            frame_width=768,
            frame_height=432,
            mean_abs_delta=0.12,
            max_increase=0.8,
            max_decrease=-0.4,
            hotspot_increase_xy=(120.0, 200.0),
            hotspot_decrease_xy=(300.0, 150.0),
            diff_image_path="heatmap_diff.png",
        )
        base = RunSnapshot("base", "/b", "/b/db", 1, 10, 2, 10, 2, 0, 0, 4.0, 2.0, 50.0)
        cand = RunSnapshot("cand", "/c", "/c/db", 1, 15, 3, 14, 5, 2, 1, 9.0, 3.0, 75.0)
        rep = RunComparisonReport(
            baseline=base,
            candidate=cand,
            config_diffs=[cfg_diff],
            zone_deltas=[z_delta],
            line_deltas=[l_delta],
            insight_deltas=[i_delta],
            temporal_deltas=[t_delta],
            spatial_shift=shift,
        )
        payload = rep.to_dict()
        self.assertEqual(payload["kpi_deltas"]["events_delta"], 3)
        self.assertEqual(payload["kpi_deltas"]["total_dwell_seconds_delta"], 5.0)
        self.assertEqual(
            payload["spatial_shift"]["hotspot_increase_xy"], [120.0, 200.0]
        )

        anomaly = RunBaselineAnomaly(
            "cand", "events_count", 25.0, 10.0, 3.0, 5.0, "above"
        )
        self.assertEqual(anomaly.to_dict()["direction"], "above")
        with self.assertRaises(ValueError):
            RunBaselineAnomaly("cand", "events_count", 25.0, 10.0, 3.0, 5.0, "sideways")

        trends = MultiRunTrendReport(
            runs=[base, cand],
            z_score_threshold=1.5,
            metric_baselines={"events_count": {"mean": 3.5, "std": 1.5}},
            anomalies=[anomaly],
        )
        self.assertEqual(trends.to_dict()["runs_count"], 2)


class TestCompareEngine(unittest.TestCase):
    """Test run snapshot loading, config diffing, deltas, spatial diff, and multi-run trends."""

    def test_load_run_snapshot_from_directory_and_db_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _create_synthetic_run(
                Path(tmpdir) / "run_alpha",
                camera_id="entrance",
                trajectories_rows=[
                    (1, 0, 0.0, 20.0, 30.0),
                    (1, 10, 2.0, 60.0, 30.0),
                    (2, 5, 1.0, 80.0, 80.0),
                    (2, 15, 4.0, 80.0, 120.0),
                ],
                events_rows=[
                    ("e1", "zone_entry", 1, "lobby", 0.0, 0.0, 0.0),
                    ("e2", "dwell_time_event", 1, "lobby", 0.0, 2.0, 2.0),
                    ("e3", "dwell_time_event", 2, "lobby", 1.0, 4.0, 3.0),
                ],
                insights=[
                    {
                        "insight_id": "ins_1",
                        "camera_id": "entrance",
                        "insight_type": "loitering",
                        "severity": "high",
                        "track_id": 2,
                        "zone_id": "lobby",
                        "start_time": 1.0,
                        "end_time": 4.0,
                        "duration": 3.0,
                        "description": "Loitering",
                    }
                ],
                run_config={
                    "confidence_threshold": 0.4,
                    "tracker_config": "bytetrack.yaml",
                },
            )
            snap = load_run_snapshot(run_dir)
            self.assertEqual(snap.run_id, "run_alpha")
            self.assertEqual(snap.tracks_count, 2)
            self.assertEqual(snap.trajectories_count, 4)
            self.assertEqual(snap.events_count, 3)
            self.assertEqual(snap.insights_count, 1)
            self.assertEqual(snap.high_critical_insights_count, 1)
            self.assertAlmostEqual(snap.total_dwell_seconds, 5.0)
            self.assertEqual(snap.config_snapshot["confidence_threshold"], 0.4)

            # Direct DB path and camera filter
            snap_db = load_run_snapshot(
                run_dir / "intelligence.db", run_id="custom_id", camera_id="entrance"
            )
            self.assertEqual(snap_db.run_id, "custom_id")
            self.assertEqual(snap_db.tracks_count, 2)

            with self.assertRaises(FileNotFoundError):
                load_run_snapshot(Path(tmpdir) / "missing_run")

    def test_diff_run_configs_flattens_nested_keys(self) -> None:
        base_cfg = {
            "confidence_threshold": 0.4,
            "tracker_config": "bytetrack.yaml",
            "insight_config": {
                "min_loitering_seconds": 15.0,
                "max_zone_occupancy": 3,
            },
        }
        cand_cfg = {
            "confidence_threshold": 0.3,
            "tracker_config": "bytetrack.yaml",
            "insight_config": {
                "min_loitering_seconds": 15.0,
                "max_zone_occupancy": 5,
            },
            "extract_video_clips": True,
        }
        diffs = diff_run_configs(base_cfg, cand_cfg)
        by_key = {d.key: d for d in diffs}
        self.assertEqual(
            set(by_key.keys()),
            {
                "confidence_threshold",
                "extract_video_clips",
                "insight_config.max_zone_occupancy",
            },
        )
        self.assertEqual(by_key["confidence_threshold"].baseline_value, 0.4)
        self.assertEqual(by_key["confidence_threshold"].candidate_value, 0.3)
        self.assertIsNone(by_key["extract_video_clips"].baseline_value)
        self.assertTrue(by_key["extract_video_clips"].candidate_value)
        self.assertEqual(by_key["insight_config.max_zone_occupancy"].candidate_value, 5)
        self.assertEqual(diff_run_configs(base_cfg, base_cfg), [])

    def test_compute_zone_and_line_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = _create_synthetic_run(
                root / "run_a",
                camera_id="cam_mon",
                zone_counts_rows=[
                    ("lobby", 5, 5, 12.0, 2),
                    ("aisle", 2, 2, 4.0, 1),
                ],
                line_counts_rows=[("door", 3, 2, 5)],
            )
            run_b = _create_synthetic_run(
                root / "run_b",
                camera_id="cam_tue",
                zone_counts_rows=[
                    ("lobby", 8, 7, 20.5, 4),
                    ("vault", 1, 1, 3.0, 1),
                ],
                line_counts_rows=[("door", 6, 1, 7), ("side_gate", 2, 0, 2)],
            )

            z_deltas = compute_zone_deltas(run_a, run_b)
            z_by_id = {z.zone_id: z for z in z_deltas}
            self.assertEqual(set(z_by_id.keys()), {"aisle", "lobby", "vault"})
            self.assertEqual(z_by_id["lobby"].delta_entries, 3)
            self.assertAlmostEqual(z_by_id["lobby"].delta_dwell_seconds, 8.5)
            self.assertEqual(z_by_id["lobby"].delta_peak_occupancy, 2)
            self.assertEqual(z_by_id["aisle"].delta_entries, -2)
            self.assertEqual(z_by_id["vault"].delta_entries, 1)

            l_deltas = compute_line_deltas(run_a, run_b)
            l_by_id = {ln.line_id: ln for ln in l_deltas}
            self.assertEqual(l_by_id["door"].delta_ab, 3)
            self.assertEqual(l_by_id["door"].delta_ba, -1)
            self.assertEqual(l_by_id["door"].delta_total, 2)
            self.assertEqual(l_by_id["side_gate"].delta_total, 2)

    def test_compute_insight_and_temporal_bucket_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = _create_synthetic_run(
                root / "run_a",
                trajectories_rows=[(1, 0, 1.0, 20.0, 20.0)],
                events_rows=[("e1", "dwell_time_event", 1, "lobby", 1.0, 3.0, 2.0)],
                insights=[
                    {
                        "insight_id": "ins_a1",
                        "camera_id": "cam_main",
                        "insight_type": "loitering",
                        "severity": "medium",
                        "track_id": 1,
                        "zone_id": "lobby",
                        "start_time": 1.0,
                        "end_time": 3.0,
                        "duration": 2.0,
                        "description": "Loitering A",
                    }
                ],
            )
            run_b = _create_synthetic_run(
                root / "run_b",
                trajectories_rows=[
                    (1, 0, 1.0, 20.0, 20.0),
                    (2, 1, 2.0, 30.0, 30.0),
                    (2, 2, 7.0, 40.0, 40.0),
                ],
                events_rows=[
                    ("e1", "dwell_time_event", 1, "lobby", 1.0, 3.0, 2.0),
                    ("e2", "dwell_time_event", 2, "lobby", 2.0, 8.0, 6.0),
                ],
                insights=[
                    {
                        "insight_id": "ins_b1",
                        "camera_id": "cam_main",
                        "insight_type": "loitering",
                        "severity": "high",
                        "track_id": 2,
                        "zone_id": "lobby",
                        "start_time": 2.0,
                        "end_time": 8.0,
                        "duration": 6.0,
                        "description": "Loitering B",
                    },
                    {
                        "insight_id": "ins_b2",
                        "camera_id": "cam_main",
                        "insight_type": "crowding",
                        "severity": "critical",
                        "track_id": 1,
                        "zone_id": "lobby",
                        "start_time": 6.0,
                        "end_time": 8.0,
                        "duration": 2.0,
                        "description": "Crowding B",
                    },
                ],
            )

            ins_deltas = compute_insight_deltas(run_a, run_b)
            by_type = {d.insight_type: d for d in ins_deltas}
            self.assertEqual(by_type["loitering"].delta_count, 0)
            self.assertEqual(by_type["loitering"].delta_high_critical, 1)
            self.assertEqual(by_type["crowding"].delta_count, 1)
            self.assertEqual(by_type["crowding"].delta_high_critical, 1)

            t_deltas = compute_temporal_bucket_deltas(run_a, run_b, bucket_seconds=5.0)
            self.assertEqual(len(t_deltas), 2)
            self.assertEqual(t_deltas[0].delta_active_tracks, 1)
            self.assertEqual(t_deltas[0].delta_events, 1)
            self.assertEqual(t_deltas[1].delta_active_tracks, 1)
            self.assertEqual(t_deltas[1].delta_insights, 1)

    def test_compute_spatial_density_diff_and_heatmap_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Baseline activity concentrated on left (x~40, y~120)
            run_a = _create_synthetic_run(
                root / "run_left",
                trajectories_rows=[
                    (1, i, float(i) * 0.2, 40.0, 120.0) for i in range(6)
                ],
                width=320,
                height=240,
            )
            # Candidate activity shifted to right (x~260, y~120)
            run_b = _create_synthetic_run(
                root / "run_right",
                trajectories_rows=[
                    (1, i, float(i) * 0.2, 260.0, 120.0) for i in range(6)
                ],
                width=320,
                height=240,
            )
            diff_img = root / "out" / "heatmap_diff.png"
            shift = compute_spatial_density_diff(
                run_a,
                run_b,
                grid_bin_size_px=16,
                gaussian_sigma_px=12.0,
                output_image_path=diff_img,
            )
            self.assertEqual(shift.frame_width, 320)
            self.assertEqual(shift.frame_height, 240)
            self.assertGreater(shift.max_increase, 0.5)
            self.assertLess(shift.max_decrease, -0.5)
            self.assertIsNotNone(shift.hotspot_increase_xy)
            self.assertIsNotNone(shift.hotspot_decrease_xy)
            assert shift.hotspot_increase_xy is not None
            assert shift.hotspot_decrease_xy is not None
            # Increase hotspot on right side, decrease hotspot on left side
            self.assertGreater(shift.hotspot_increase_xy[0], 200.0)
            self.assertLess(shift.hotspot_decrease_xy[0], 100.0)
            self.assertTrue(diff_img.is_file())
            loaded_img = cv2.imread(str(diff_img))
            self.assertIsNotNone(loaded_img)
            assert loaded_img is not None
            self.assertEqual(loaded_img.shape, (240, 320, 3))

    def test_discover_runs_and_analyze_multi_run_trends_flags_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs"
            runs_root.mkdir()
            # Non-run folder without intelligence.db should be ignored
            (runs_root / "not_a_run").mkdir()

            # 3 normal runs with 2 events each, 1 outlier run with 20 events
            for name in ("run_01", "run_02", "run_03"):
                _create_synthetic_run(
                    runs_root / name,
                    events_rows=[
                        ("e1", "dwell_time_event", 1, "lobby", 0.0, 2.0, 2.0),
                        ("e2", "dwell_time_event", 2, "lobby", 2.0, 4.0, 2.0),
                    ],
                )
            _create_synthetic_run(
                runs_root / "run_04_spike",
                events_rows=[
                    (
                        f"e{i}",
                        "dwell_time_event",
                        i,
                        "lobby",
                        0.0,
                        10.0,
                        10.0,
                    )
                    for i in range(1, 21)
                ],
            )

            found = discover_runs(runs_root)
            self.assertEqual(
                [p.name for p in found],
                ["run_01", "run_02", "run_03", "run_04_spike"],
            )

            trends = analyze_multi_run_trends(found, z_score_threshold=1.5)
            self.assertEqual(len(trends.runs), 4)
            self.assertIn("events_count", trends.metric_baselines)
            flagged_ids = {
                (a.run_id, a.metric_name, a.direction) for a in trends.anomalies
            }
            self.assertIn(("run_04_spike", "events_count", "above"), flagged_ids)
            self.assertIn(("run_04_spike", "total_dwell_seconds", "above"), flagged_ids)

            with self.assertRaises(ValueError):
                analyze_multi_run_trends(found, z_score_threshold=0.0)

    def test_markdown_report_generators_and_artifact_writers(self) -> None:
        base = RunSnapshot(
            "run_a", "/a", "/a/db", 1, 10, 2, 10, 2, 0, 0, 4.0, 2.0, 50.0
        )
        cand = RunSnapshot(
            "run_b", "/b", "/b/db", 1, 18, 4, 20, 6, 2, 1, 11.0, 3.0, 80.0
        )
        comp = RunComparisonReport(
            baseline=base,
            candidate=cand,
            config_diffs=[ConfigDiffEntry("confidence_threshold", 0.4, 0.3)],
            zone_deltas=[ZoneMetricDelta("cam1", "lobby", 2, 5, 2, 4, 4.0, 11.0, 1, 3)],
            line_deltas=[LineMetricDelta("cam1", "door", 1, 3, 1, 2, 2, 5)],
            insight_deltas=[InsightTypeDelta("loitering", 0, 2, 0, 1)],
            temporal_deltas=[TemporalBucketDelta(0, 0.0, 5.0, 2, 4, 2, 6, 0, 2)],
            spatial_shift=SpatialShiftSummary(
                grid_shape=(15, 20),
                frame_width=320,
                frame_height=240,
                mean_abs_delta=0.1,
                max_increase=0.7,
                max_decrease=-0.3,
                hotspot_increase_xy=(200.0, 120.0),
                hotspot_decrease_xy=(40.0, 120.0),
                diff_image_path="/tmp/heatmap_diff.png",
            ),
        )
        comp_md = generate_comparison_markdown(comp)
        self.assertIn("# Run Comparison Report — `run_a` vs `run_b`", comp_md)
        self.assertIn("## Configuration Drift", comp_md)
        self.assertIn("## Zone Activity Deltas", comp_md)
        self.assertIn("## Line Crossing Deltas", comp_md)
        self.assertIn("## Behavioral Insight Deltas", comp_md)
        self.assertIn("## Temporal Activity Histogram Deltas", comp_md)
        self.assertIn("[heatmap_diff.png](heatmap_diff.png)", comp_md)

        trends = MultiRunTrendReport(
            runs=[base, cand],
            z_score_threshold=1.0,
            metric_baselines={
                "events_count": {"mean": 4.0, "std": 2.0, "min": 2.0, "max": 6.0}
            },
            anomalies=[
                RunBaselineAnomaly("run_b", "events_count", 6.0, 4.0, 2.0, 1.0, "above")
            ],
        )
        trends_md = generate_trends_markdown(trends)
        self.assertIn("# Multi-Run Baseline & Trend Analytics (2 Runs)", trends_md)
        self.assertIn("Cross-Run Baseline Anomalies", trends_md)
        self.assertIn("`run_b`", trends_md)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            c_json, c_md = write_comparison_artifacts(comp, out / "comp")
            t_json, t_md = write_trends_artifacts(trends, out / "trends")
            self.assertTrue(c_json.is_file())
            self.assertTrue(c_md.is_file())
            self.assertTrue(t_json.is_file())
            self.assertTrue(t_md.is_file())

    def test_compare_runs_and_cli_pairwise_and_trends_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs_root = root / "runs"
            run_a = _create_synthetic_run(
                runs_root / "run_baseline",
                trajectories_rows=[
                    (1, 0, 0.5, 40.0, 100.0),
                    (1, 5, 2.5, 45.0, 105.0),
                ],
                events_rows=[("e1", "dwell_time_event", 1, "lobby", 0.5, 2.5, 2.0)],
                zone_counts_rows=[("lobby", 1, 1, 2.0, 1)],
                line_counts_rows=[("gate", 1, 0, 1)],
                run_config={
                    "confidence_threshold": 0.4,
                    "tracker_config": "bytetrack.yaml",
                },
            )
            run_b = _create_synthetic_run(
                runs_root / "run_candidate",
                trajectories_rows=[
                    (1, 0, 0.5, 240.0, 100.0),
                    (2, 2, 1.5, 245.0, 105.0),
                    (2, 8, 6.5, 250.0, 110.0),
                ],
                events_rows=[
                    ("e1", "dwell_time_event", 1, "lobby", 0.5, 3.5, 3.0),
                    ("e2", "dwell_time_event", 2, "lobby", 1.5, 6.5, 5.0),
                ],
                zone_counts_rows=[("lobby", 2, 2, 8.0, 2)],
                line_counts_rows=[("gate", 3, 1, 4)],
                insights=[
                    {
                        "insight_id": "ins_1",
                        "camera_id": "cam_main",
                        "insight_type": "loitering",
                        "severity": "high",
                        "track_id": 2,
                        "zone_id": "lobby",
                        "start_time": 1.5,
                        "end_time": 6.5,
                        "duration": 5.0,
                        "description": "Loitering detected",
                    }
                ],
                run_config={
                    "confidence_threshold": 0.3,
                    "tracker_config": "botsort.yaml",
                },
            )

            # Direct compare_runs call
            out_direct = root / "direct_out"
            rep = compare_runs(run_a, run_b, output_dir=out_direct)
            self.assertEqual(rep.kpi_deltas["events_delta"], 1)
            self.assertEqual(len(rep.config_diffs), 2)
            self.assertTrue((out_direct / "comparison_report.json").is_file())
            self.assertTrue((out_direct / "comparison_report.md").is_file())
            self.assertTrue((out_direct / "heatmap_diff.png").is_file())

            # Pairwise CLI (JSON & text)
            cli_out = root / "cli_pair_out"
            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc = compare_main(
                    [
                        "--baseline",
                        str(run_a),
                        "--candidate",
                        str(run_b),
                        "--output-dir",
                        str(cli_out),
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf_json.getvalue())
            self.assertEqual(payload["baseline"]["run_id"], "run_baseline")
            self.assertEqual(payload["candidate"]["run_id"], "run_candidate")
            self.assertTrue((cli_out / "heatmap_diff.png").is_file())

            buf_txt = io.StringIO()
            with redirect_stdout(buf_txt):
                rc_txt = compare_main(
                    [
                        "--baseline",
                        str(run_a),
                        "--candidate",
                        str(run_b),
                        "--no-heatmap-diff",
                    ]
                )
            self.assertEqual(rc_txt, 0)
            self.assertIn(
                "Run Comparison: run_baseline vs run_candidate", buf_txt.getvalue()
            )

            # Multi-run trends CLI (JSON & text)
            trends_out = root / "cli_trends_out"
            buf_tr = io.StringIO()
            with redirect_stdout(buf_tr):
                rc_tr = compare_main(
                    [
                        "--runs-root",
                        str(runs_root),
                        "--z-threshold",
                        "0.5",
                        "--output-dir",
                        str(trends_out),
                        "--json",
                    ]
                )
            self.assertEqual(rc_tr, 0)
            tr_payload = json.loads(buf_tr.getvalue())
            self.assertEqual(tr_payload["runs_count"], 2)
            self.assertTrue((trends_out / "trends_report.json").is_file())
            self.assertTrue((trends_out / "trends_report.md").is_file())

            buf_tr_txt = io.StringIO()
            with redirect_stdout(buf_tr_txt):
                rc_tr_txt = compare_main(["--runs", str(run_a), str(run_b)])
            self.assertEqual(rc_tr_txt, 0)
            self.assertIn("Multi-Run Baseline Trends (2 run(s)", buf_tr_txt.getvalue())

    def test_spatial_diff_prefers_video_info_json_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = _create_synthetic_run(
                root / "run_vinfo_a",
                trajectories_rows=[(1, 0, 0.0, 100.0, 100.0)],
                width=1920,
                height=1080,
            )
            run_b = _create_synthetic_run(
                root / "run_vinfo_b",
                trajectories_rows=[(1, 0, 0.0, 500.0, 300.0)],
                width=1920,
                height=1080,
            )
            (run_b / "video_info.json").write_text(
                json.dumps([{"name": "clip.mp4", "width": 768, "height": 432}]),
                encoding="utf-8",
            )
            shift = compute_spatial_density_diff(run_a, run_b)
            self.assertEqual((shift.frame_width, shift.frame_height), (768, 432))

    def test_regression_policy_yaml_and_evaluate_regression_gate(self) -> None:
        from event_intelligence.compare.engine import (
            evaluate_regression_gate,
            load_regression_policy,
            save_regression_policy,
        )
        from event_intelligence.compare.types import RegressionPolicy

        with self.assertRaises(ValueError):
            RegressionPolicy(max_tracks_drop_pct=150.0)
        with self.assertRaises(ValueError):
            RegressionPolicy(max_high_critical_insights_increase=-1)
        with self.assertRaises(ValueError):
            RegressionPolicy(min_detections_ratio=-0.5)

        pol = RegressionPolicy(
            max_tracks_drop_pct=20.0,
            max_events_drop_pct=20.0,
            max_dwell_drop_pct=25.0,
            max_high_critical_insights_increase=1,
            max_spatial_shift_mean_abs=0.15,
            min_detections_ratio=0.8,
            allow_config_drift=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p_path = Path(tmpdir) / "policy.yaml"
            save_regression_policy(pol, p_path)
            loaded = load_regression_policy(p_path)
            self.assertEqual(loaded, pol)

        # Passing report
        base = RunSnapshot(
            "base", "/b", "/b/db", 1, 100, 10, 100, 10, 1, 1, 20.0, 2.0, 50.0
        )
        good_cand = RunSnapshot(
            "cand", "/c", "/c/db", 1, 95, 10, 95, 10, 2, 2, 19.0, 2.0, 50.0
        )
        good_rep = RunComparisonReport(
            baseline=base,
            candidate=good_cand,
            spatial_shift=SpatialShiftSummary((10, 10), 160, 160, 0.05, 0.1, -0.1),
        )
        pass_res = evaluate_regression_gate(good_rep, pol)
        self.assertTrue(pass_res.passed)
        self.assertEqual(pass_res.error_count, 0)
        self.assertEqual(pass_res.checks_evaluated, 7)

        # Failing report triggering all 7 rules
        bad_cand = RunSnapshot(
            "bad", "/c", "/c/db", 1, 50, 5, 50, 4, 5, 5, 10.0, 1.0, 20.0
        )
        bad_rep = RunComparisonReport(
            baseline=base,
            candidate=bad_cand,
            config_diffs=[ConfigDiffEntry("confidence_threshold", 0.4, 0.7)],
            spatial_shift=SpatialShiftSummary((10, 10), 160, 160, 0.40, 0.8, -0.8),
        )
        fail_res = evaluate_regression_gate(bad_rep, pol)
        self.assertFalse(fail_res.passed)
        self.assertEqual(fail_res.error_count, 7)
        violated_rules = {v.rule_name for v in fail_res.violations}
        self.assertEqual(
            violated_rules,
            {
                "max_tracks_drop_pct",
                "max_events_drop_pct",
                "max_dwell_drop_pct",
                "max_high_critical_insights_increase",
                "max_spatial_shift_mean_abs",
                "min_detections_ratio",
                "allow_config_drift",
            },
        )

    def test_regression_gate_markdown_and_csv_artifact_exports(self) -> None:
        from event_intelligence.compare.engine import evaluate_regression_gate
        from event_intelligence.compare.types import RegressionPolicy

        base = RunSnapshot(
            "run_a", "/a", "/a/db", 1, 100, 10, 100, 10, 0, 0, 20.0, 2.0, 50.0
        )
        cand = RunSnapshot(
            "run_b", "/b", "/b/db", 1, 50, 5, 50, 4, 2, 2, 8.0, 1.5, 40.0
        )
        raw_rep = RunComparisonReport(
            baseline=base,
            candidate=cand,
            zone_deltas=[ZoneMetricDelta("cam1", "lobby", 5, 2, 5, 2, 20.0, 8.0, 3, 1)],
            line_deltas=[LineMetricDelta("cam1", "door", 4, 2, 1, 1, 5, 3)],
            temporal_deltas=[TemporalBucketDelta(0, 0.0, 5.0, 5, 2, 5, 2, 0, 2)],
        )
        gate = evaluate_regression_gate(raw_rep, RegressionPolicy())
        rep = RunComparisonReport(
            baseline=base,
            candidate=cand,
            zone_deltas=raw_rep.zone_deltas,
            line_deltas=raw_rep.line_deltas,
            temporal_deltas=raw_rep.temporal_deltas,
            regression_gate=gate,
        )
        md = generate_comparison_markdown(rep)
        self.assertIn("## Automated Regression Gate", md)
        self.assertIn("**`FAIL`**", md)
        self.assertIn("`max_tracks_drop_pct`", md)

        trends = MultiRunTrendReport(
            runs=[base, cand],
            z_score_threshold=1.0,
            metric_baselines={
                "events_count": {"mean": 7.0, "std": 3.0, "min": 4.0, "max": 10.0}
            },
            anomalies=[
                RunBaselineAnomaly(
                    "run_b", "events_count", 4.0, 7.0, 3.0, -1.0, "below"
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            write_comparison_artifacts(rep, out / "comp")
            for fname in ("zone_deltas.csv", "line_deltas.csv", "temporal_deltas.csv"):
                csv_file = out / "comp" / fname
                self.assertTrue(csv_file.is_file(), f"Missing {fname}")
                self.assertGreaterEqual(
                    len(csv_file.read_text(encoding="utf-8").splitlines()), 2
                )

            write_trends_artifacts(trends, out / "trends")
            for fname in ("runs_catalog.csv", "baseline_anomalies.csv"):
                csv_file = out / "trends" / fname
                self.assertTrue(csv_file.is_file(), f"Missing {fname}")
                self.assertGreaterEqual(
                    len(csv_file.read_text(encoding="utf-8").splitlines()), 2
                )


if __name__ == "__main__":
    unittest.main()
