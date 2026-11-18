"""Unit and integration tests for Phase 11h multi-run comparison and baseline analytics."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
