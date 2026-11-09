"""Unit and integration tests for Phase 11 CV Pipeline Integration (src/event_intelligence/)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from helpers import make_sequence, make_video

from event_intelligence.analytics import (
    accumulate_trajectory_grid,
    compute_line_counts,
    compute_zone_counts,
    smooth_and_normalize_heatmap,
)
from event_intelligence.detection import (
    DETECTIONS_CSV_HEADER,
    Detection,
    FrameDetections,
    read_detections_csv,
    write_detections_csv,
)
from event_intelligence.evaluation import (
    EvaluationConfig,
    evaluate_detection_precision_recall,
    evaluate_tracking_metrics,
)
from event_intelligence.events import (
    EventConfig,
    EventRecord,
    ZoneDwellStateMachine,
    evaluate_dwell_events,
)
from event_intelligence.evidence import (
    EvidenceEventRecord,
    extract_person_crop,
    index_track_detections,
    read_evidence_events_csv,
    select_best_frame_for_event,
    write_evidence_events_csv,
)
from event_intelligence.spatial import (
    LineSpec,
    SceneConfig,
    ZoneSpec,
    check_line_crossing,
    evaluate_scene_signals,
    point_in_zone,
)
from event_intelligence.tracking import (
    FrameTracks,
    TrackDetection,
    compute_track_summaries,
    evaluate_tracking,
    read_tracks_csv,
    write_tracks_csv,
)
from event_intelligence.trajectories import (
    TrajectoryPoint,
    compute_anchor_point,
    compute_trajectory_summaries,
    read_trajectories_csv,
    smooth_trajectories,
    tracks_to_trajectories,
    write_trajectories_csv,
)
from event_intelligence.video import (
    VideoSource,
    inspect_source,
    sample_frame_indices,
    validate_video_info,
)


class TestPromotedVideoPackage(unittest.TestCase):
    """Test event_intelligence.video promotion on video files and MOT sequences."""

    def test_video_source_and_inspect_source_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clip_path = make_video(root, count=6)
            seq_path = make_sequence(root, name="MOT17-11-FRCNN", count=5)

            with VideoSource(clip_path) as vs:
                self.assertEqual(len(vs), 6)
                self.assertEqual(vs.metadata.source_type, "video_file")

            indices = sample_frame_indices(6, 3)
            self.assertEqual(indices, [0, 2, 5])

            out_dir = root / "inspected"
            entry = inspect_source(clip_path, out_dir, sample_count=3)
            self.assertEqual(entry["frame_count"], 6)
            self.assertEqual(validate_video_info([entry]), [])

            with VideoSource(seq_path) as seq_vs:
                self.assertEqual(seq_vs.metadata.source_type, "frame_sequence")


class TestPromotedDetectionPackage(unittest.TestCase):
    """Test event_intelligence.detection CSV schema and round-trip."""

    def test_detections_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "detections.csv"
            records = [
                FrameDetections(0, [Detection(10.0, 20.0, 40.0, 80.0, 0.91)]),
                FrameDetections(1, []),
            ]
            total = write_detections_csv(records, csv_path)
            self.assertEqual(total, 1)
            rows = read_detections_csv(csv_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0].keys()), DETECTIONS_CSV_HEADER)
            self.assertAlmostEqual(rows[0]["confidence"], 0.91)
            self.assertIsNone(rows[1]["x1"])


class TestPromotedTrackingPackage(unittest.TestCase):
    """Test event_intelligence.tracking CSV round-trip, summaries, and CLEAR-MOT evaluation."""

    def test_tracks_csv_summaries_and_evaluate_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "tracks.csv"
            ft0 = FrameTracks(0, [TrackDetection(0, 1, 10.0, 20.0, 40.0, 80.0, 0.95)])
            ft1 = FrameTracks(1, [TrackDetection(1, 1, 12.0, 20.0, 42.0, 80.0, 0.92)])
            write_tracks_csv(csv_path, [ft0, ft1])
            loaded = read_tracks_csv(csv_path)
            self.assertEqual(len(loaded), 2)

            summaries = compute_track_summaries(loaded, fps=10.0)
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].duration_frames, 2)
            self.assertAlmostEqual(summaries[0].duration_seconds, 0.2)

            gt = {
                0: [TrackDetection(0, 10, 10.0, 20.0, 40.0, 80.0, 1.0)],
                1: [TrackDetection(1, 10, 12.0, 20.0, 42.0, 80.0, 1.0)],
            }
            metrics = evaluate_tracking(gt, loaded)
            self.assertEqual(metrics["mota"], 1.0)
            self.assertEqual(metrics["idf1"], 1.0)
            self.assertEqual(metrics["id_switches"], 0)


class TestPromotedTrajectoriesPackage(unittest.TestCase):
    """Test event_intelligence.trajectories anchor math, smoothing, and path length."""

    def test_trajectories_extraction_smoothing_and_summary(self) -> None:
        ax, ay = compute_anchor_point(10.0, 20.0, 30.0, 80.0, anchor="bottom_center")
        self.assertEqual((ax, ay), (20.0, 80.0))

        ft0 = FrameTracks(0, [TrackDetection(0, 1, 10.0, 20.0, 30.0, 80.0, 0.9)])
        ft1 = FrameTracks(1, [TrackDetection(1, 1, 13.0, 24.0, 33.0, 84.0, 0.9)])
        raw = tracks_to_trajectories([ft0, ft1], fps=10.0)
        smoothed = smooth_trajectories(raw, window=1)
        summaries = compute_trajectory_summaries(smoothed, fps=10.0)
        self.assertEqual(len(summaries), 1)
        self.assertAlmostEqual(summaries[0].path_length_px, 5.0, places=4)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trajectories.csv"
            write_trajectories_csv(csv_path, smoothed)
            reloaded = read_trajectories_csv(csv_path)
            self.assertEqual(len(reloaded[1]), 2)


class TestPromotedSpatialPackage(unittest.TestCase):
    """Test event_intelligence.spatial point-in-polygon and directional line crossings."""

    def test_point_in_zone_and_line_crossing(self) -> None:
        zone = ZoneSpec(
            id="lobby",
            polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        )
        line = LineSpec(id="gate", points=((0.0, 50.0), (100.0, 50.0)))
        self.assertTrue(point_in_zone(50.0, 50.0, zone))
        self.assertFalse(point_in_zone(150.0, 50.0, zone))
        self.assertEqual(check_line_crossing((50.0, 20.0), (50.0, 80.0), line), "A->B")
        self.assertEqual(check_line_crossing((50.0, 80.0), (50.0, 20.0), line), "B->A")

        scene = SceneConfig(zones=(zone,), lines=(line,))
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 50.0, 20.0),
                TrajectoryPoint(1, 1, 0.1, 50.0, 80.0),
                TrajectoryPoint(1, 2, 0.2, 150.0, 80.0),
            ]
        }
        sigs = evaluate_scene_signals(trajs, scene)
        self.assertEqual(
            [s.signal_type for s in sigs], ["zone_enter", "line_cross", "zone_exit"]
        )


class TestPromotedEventsPackage(unittest.TestCase):
    """Test event_intelligence.events ZoneDwellStateMachine and evaluate_dwell_events."""

    def test_zone_dwell_state_machine_and_evaluate(self) -> None:
        cfg = EventConfig(min_dwell_seconds=1.0, min_entry_frames=1, grace_frames=2)
        sm = ZoneDwellStateMachine(track_id=1, zone_id="lobby", config=cfg)
        e0 = sm.step(0, 0.0, is_inside=True)
        self.assertEqual([e.event_type for e in e0], ["zone_entry"])
        self.assertEqual(sm.step(1, 0.5, is_inside=False), [])  # grace absorbed
        self.assertEqual(sm.step(2, 1.2, is_inside=True), [])
        resolved = sm.finish_track()
        self.assertEqual([e.event_type for e in resolved], ["dwell_time_event"])

        zone = ZoneSpec(
            id="lobby",
            polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        )
        scene = SceneConfig(zones=(zone,))
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 50.0, 50.0),
                TrajectoryPoint(1, 1, 1.5, 50.0, 50.0),
            ]
        }
        events, traces = evaluate_dwell_events(trajs, scene, cfg)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(traces), 1)


class TestPromotedEvidencePackage(unittest.TestCase):
    """Test event_intelligence.evidence best-frame selection, cropping, and 9-col CSV."""

    def test_best_frame_crop_and_evidence_csv(self) -> None:
        ev = EventRecord("evt_0001", "dwell_time_event", 1, "lobby", 0.0, 0.2, 0.2)
        ft0 = FrameTracks(0, [TrackDetection(0, 1, 10.0, 10.0, 30.0, 40.0, 0.8)])
        ft1 = FrameTracks(1, [TrackDetection(1, 1, 10.0, 10.0, 30.0, 40.0, 0.95)])
        indexed = index_track_detections([ft0, ft1])
        sel = select_best_frame_for_event(ev, indexed, fps=10.0)
        self.assertEqual(sel.frame_id, 1)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        crop, clipped = extract_person_crop(frame, sel)
        self.assertFalse(clipped)
        self.assertGreater(crop.size, 0)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "events.csv"
            rec = EvidenceEventRecord.from_event(
                ev,
                snapshot_path="snapshots/evt_0001_1.jpg",
                crop_path="crops/evt_0001_1.jpg",
                best_frame_id=sel.frame_id,
                best_confidence=sel.confidence,
            )
            write_evidence_events_csv([rec], csv_path)
            loaded = read_evidence_events_csv(csv_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].snapshot_path, "snapshots/evt_0001_1.jpg")


class TestPromotedAnalyticsPackage(unittest.TestCase):
    """Test event_intelligence.analytics heatmap accumulation and zone/line counts."""

    def test_heatmap_grid_and_zone_line_counts(self) -> None:
        pts = [
            TrajectoryPoint(1, 0, 0.0, 20.0, 20.0),
            TrajectoryPoint(1, 1, 0.1, 22.0, 22.0),
        ]
        raw_grid, full_res = accumulate_trajectory_grid(
            pts,
            frame_width=64,
            frame_height=48,
            grid_bin_size_px=16,
        )
        self.assertEqual(raw_grid.shape, (3, 4))
        self.assertEqual(full_res.shape, (48, 64))
        norm = smooth_and_normalize_heatmap(full_res, gaussian_sigma_px=4.0)
        self.assertAlmostEqual(float(norm.max()), 1.0, places=4)

        events = [
            EventRecord("evt_1", "zone_entry", 1, "lobby", 0.0, 0.0, 0.0),
            EventRecord("evt_2", "dwell_time_event", 1, "lobby", 0.0, 2.0, 2.0),
            EventRecord("evt_3", "line_crossing", 1, "gate:A->B", 1.0, 1.0, 0.0),
        ]
        z_counts = compute_zone_counts(events, trajectory_points=pts)
        l_counts = compute_line_counts(events)
        self.assertEqual(z_counts[0].entry_count, 1)
        self.assertEqual(l_counts[0].crossings_ab, 1)


class TestPromotedEvaluationPackage(unittest.TestCase):
    """Test event_intelligence.evaluation detection and tracking evaluators."""

    def test_evaluation_config_and_metrics(self) -> None:
        cfg = EvaluationConfig()
        self.assertEqual(cfg.iou_threshold, 0.5)
        gt = {0: [TrackDetection(0, 1, 10.0, 10.0, 50.0, 50.0, 1.0)]}
        preds = [FrameTracks(0, [TrackDetection(0, 10, 10.0, 10.0, 50.0, 50.0, 0.9)])]
        det_m = evaluate_detection_precision_recall(gt, preds)
        trk_m = evaluate_tracking_metrics("bytetrack", gt, preds)
        self.assertEqual(det_m.precision, 1.0)
        self.assertEqual(trk_m.mota, 1.0)


if __name__ == "__main__":
    unittest.main()
