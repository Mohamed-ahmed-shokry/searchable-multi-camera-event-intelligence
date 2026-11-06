"""Unit and integration tests for Phase 9 CV evaluation (evaluate_pipeline.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluate_pipeline import (
    DetectionMetrics,
    EvaluationConfig,
    FailureCaseRecord,
    QualitativeSpotCheck,
    StageFpsRecord,
    TrackingMetrics,
    collect_stage_fps_table,
    evaluate_detection_precision_recall,
    evaluate_qualitative_events,
    evaluate_tracking_metrics,
    export_failure_examples,
    load_evaluation_config,
    save_evaluation_config,
)
from generate_dwell_events import EventRecord
from helpers import make_sequence, make_video
from track_people import FrameTracks, TrackDetection


class TestEvaluationDataclasses(unittest.TestCase):
    """Test validation and dictionary serialization of Phase 9 dataclasses."""

    def test_evaluation_config_defaults_and_validation(self) -> None:
        cfg = EvaluationConfig()
        self.assertEqual(cfg.iou_threshold, 0.5)
        self.assertEqual(cfg.mot17_max_frames, 300)
        self.assertEqual(cfg.expected_dwell_events_count, 10)
        self.assertEqual(cfg.expected_track_lost_in_zone_count, 3)

        with self.assertRaises(ValueError):
            EvaluationConfig(iou_threshold=0.0)
        with self.assertRaises(ValueError):
            EvaluationConfig(min_gt_visibility=1.5)
        with self.assertRaises(ValueError):
            EvaluationConfig(mot17_max_frames=0)
        with self.assertRaises(ValueError):
            EvaluationConfig(dwell_tolerance_seconds=-0.1)

    def test_detection_metrics_validation_and_to_dict(self) -> None:
        m = DetectionMetrics(
            precision=0.8287,
            recall=0.5670,
            f1=0.6733,
            tp=2018,
            fp=417,
            fn=1541,
            num_gt=3559,
            num_predictions=2435,
            mean_iou=0.8587,
        )
        d = m.to_dict()
        self.assertEqual(d["tp"], 2018)
        self.assertEqual(d["precision"], 0.8287)
        with self.assertRaises(ValueError):
            DetectionMetrics(
                precision=1.2,
                recall=0.5,
                f1=0.5,
                tp=1,
                fp=0,
                fn=0,
                num_gt=1,
                num_predictions=1,
                mean_iou=0.8,
            )

    def test_tracking_metrics_validation_and_to_dict(self) -> None:
        tm = TrackingMetrics(
            tracker_name="bytetrack",
            mota=0.4484,
            motp=0.8587,
            idf1=0.6026,
            precision=0.8287,
            recall=0.5670,
            id_switches=5,
            tp=2018,
            fp=417,
            fn=1541,
            num_gt_boxes=3559,
            num_pred_boxes=2435,
            num_gt_tracks=75,
            num_pred_tracks=43,
            mostly_tracked=20,
            partially_tracked=18,
            mostly_lost=37,
            fragmentation_count=35,
        )
        self.assertEqual(tm.to_dict()["tracker_name"], "bytetrack")
        self.assertEqual(tm.to_dict()["mostly_tracked"], 20)
        with self.assertRaises(ValueError):
            TrackingMetrics(
                tracker_name="",
                mota=0.5,
                motp=0.8,
                idf1=0.6,
                precision=0.8,
                recall=0.6,
                id_switches=0,
                tp=1,
                fp=0,
                fn=0,
                num_gt_boxes=1,
                num_pred_boxes=1,
                num_gt_tracks=1,
                num_pred_tracks=1,
                mostly_tracked=1,
                partially_tracked=0,
                mostly_lost=0,
                fragmentation_count=0,
            )

    def test_qualitative_spot_check_stage_fps_and_failure_case(self) -> None:
        qc = QualitativeSpotCheck(
            total_events=48,
            zone_entry_accuracy=1.0,
            zone_exit_accuracy=1.0,
            line_crossing_accuracy=1.0,
            dwell_count_matches=True,
            track_lost_in_zone_matches=True,
            max_dwell_duration_error_seconds=0.0,
            dwell_within_tolerance=True,
            zone_entry_comparison={
                "entrance_area": {"expected": 7, "actual": 7, "match": True}
            },
            zone_exit_comparison={
                "entrance_area": {"expected": 7, "actual": 7, "match": True}
            },
            line_crossing_comparison={
                "doorway_line:B->A": {"expected": 7, "actual": 7, "match": True}
            },
        )
        self.assertTrue(qc.to_dict()["dwell_within_tolerance"])

        fps_rec = StageFpsRecord(
            stage_id="02_person_detection",
            stage_name="Person Detection (YOLO11m)",
            device="cuda:0",
            frames_processed=596,
            elapsed_seconds=32.446,
            fps=18.37,
            ms_per_frame=54.44,
            runtime_share_pct=40.2,
        )
        self.assertEqual(fps_rec.to_dict()["fps"], 18.37)

        fc = FailureCaseRecord(
            case_id="failure_01",
            category="tracking_id_switch",
            source_name="MOT17-11-FRCNN",
            frame_id=23,
            title="Occlusion Re-entry ID Switch",
            symptom="GT ID 28 switches from pred ID 7 to 12",
            root_cause="4-frame full occlusion by foreground pedestrian",
            impact_and_mitigation="Splits track into two IDs; tune track_buffer",
            image_path="failure_examples/failure_01.jpg",
        )
        self.assertEqual(fc.to_dict()["frame_id"], 23)


class TestEvaluationConfigYaml(unittest.TestCase):
    """Test load_evaluation_config and save_evaluation_config YAML helpers."""

    def test_yaml_round_trip_and_error_cases(self) -> None:
        self.assertEqual(load_evaluation_config(None), EvaluationConfig())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = EvaluationConfig(
                iou_threshold=0.6,
                mot17_max_frames=150,
                expected_dwell_events_count=8,
            )
            yaml_path = save_evaluation_config(cfg, root / "evaluation.yaml")
            loaded = load_evaluation_config(yaml_path)
            self.assertEqual(loaded, cfg)

            empty_yaml = root / "empty.yaml"
            empty_yaml.write_text("", encoding="utf-8")
            self.assertEqual(load_evaluation_config(empty_yaml), EvaluationConfig())

            invalid_yaml = root / "invalid.yaml"
            invalid_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                load_evaluation_config(invalid_yaml)

            with self.assertRaises(FileNotFoundError):
                load_evaluation_config(root / "missing.yaml")


class TestEvaluateDetectionPrecisionRecall(unittest.TestCase):
    """Test evaluate_detection_precision_recall against synthetic ground truth."""

    def test_detection_precision_recall_tp_fp_fn(self) -> None:
        gt = {
            0: [
                TrackDetection(0, 1, 10.0, 10.0, 50.0, 50.0, 1.0),
                TrackDetection(0, 2, 100.0, 100.0, 140.0, 140.0, 1.0),
            ],
            1: [
                TrackDetection(1, 1, 12.0, 12.0, 52.0, 52.0, 1.0),
            ],
        }
        preds = [
            FrameTracks(
                frame_id=0,
                tracks=(
                    TrackDetection(0, 10, 10.0, 10.0, 50.0, 50.0, 0.9),  # TP (IoU=1.0)
                    TrackDetection(
                        0, 11, 200.0, 200.0, 240.0, 240.0, 0.8
                    ),  # FP (no overlap)
                ),
            ),
            FrameTracks(
                frame_id=1,
                tracks=(
                    TrackDetection(1, 10, 12.0, 12.0, 52.0, 52.0, 0.95),  # TP (IoU=1.0)
                ),
            ),
        ]
        res = evaluate_detection_precision_recall(gt, preds, iou_threshold=0.5)
        self.assertEqual(res.tp, 2)
        self.assertEqual(res.fp, 1)
        self.assertEqual(res.fn, 1)
        self.assertEqual(res.num_gt, 3)
        self.assertEqual(res.num_predictions, 3)
        self.assertAlmostEqual(res.precision, 0.6667, places=4)
        self.assertAlmostEqual(res.recall, 0.6667, places=4)
        self.assertAlmostEqual(res.f1, 0.6667, places=4)
        self.assertAlmostEqual(res.mean_iou, 1.0, places=4)

        with self.assertRaises(ValueError):
            evaluate_detection_precision_recall(gt, preds, iou_threshold=0.0)


class TestEvaluateTrackingMetrics(unittest.TestCase):
    """Test evaluate_tracking_metrics with MT/PT/ML lifespan coverage and ID switches."""

    def test_tracking_metrics_mt_pt_ml_and_idsw(self) -> None:
        # GT track 1: present in all 5 frames (0..4) -> matched in all 5 -> MT (100%)
        # GT track 2: present in 5 frames -> matched in 2 frames -> PT (40%)
        # GT track 3: present in 5 frames -> matched in 0 frames -> ML (0%)
        gt: dict[int, list[TrackDetection]] = {}
        preds: list[FrameTracks] = []
        for fid in range(5):
            gt[fid] = [
                TrackDetection(fid, 1, 10.0, 10.0, 50.0, 50.0, 1.0),
                TrackDetection(fid, 2, 100.0, 100.0, 140.0, 140.0, 1.0),
                TrackDetection(fid, 3, 200.0, 200.0, 240.0, 240.0, 1.0),
            ]
            # Pred for GT 1 switches ID from 10 to 99 at frame 3
            pred_id_for_1 = 10 if fid < 3 else 99
            frame_preds = [
                TrackDetection(fid, pred_id_for_1, 10.0, 10.0, 50.0, 50.0, 0.9),
            ]
            # Pred 20 for GT 2 appears at frame 0 and frame 4 (fragmentation gap of 3 frames)
            if fid in (0, 4):
                frame_preds.append(
                    TrackDetection(fid, 20, 100.0, 100.0, 140.0, 140.0, 0.85)
                )
            preds.append(FrameTracks(frame_id=fid, tracks=tuple(frame_preds)))

        tm = evaluate_tracking_metrics("bytetrack", gt, preds, iou_threshold=0.5)
        self.assertEqual(tm.tracker_name, "bytetrack")
        self.assertEqual(tm.num_gt_tracks, 3)
        self.assertEqual(tm.num_pred_tracks, 3)  # IDs 10, 99, 20
        self.assertEqual(tm.mostly_tracked, 1)
        self.assertEqual(tm.partially_tracked, 1)
        self.assertEqual(tm.mostly_lost, 1)
        self.assertEqual(tm.id_switches, 1)
        self.assertEqual(tm.fragmentation_count, 1)
        self.assertEqual(len(tm.id_switch_events), 1)


class TestEvaluateQualitativeEvents(unittest.TestCase):
    """Test evaluate_qualitative_events against manual playback audit expectations."""

    def test_qualitative_events_spot_check(self) -> None:
        cfg = EvaluationConfig(
            expected_zone_entries={"z1": 1},
            expected_zone_exits={"z1": 1},
            expected_line_crossings={"l1:A->B": 1},
            expected_dwell_events_count=1,
            expected_track_lost_in_zone_count=0,
        )
        events = [
            EventRecord("evt_0001", "zone_entry", 1, "z1", 1.0, 1.0, 0.0),
            EventRecord("evt_0002", "dwell_time_event", 1, "z1", 1.0, 3.0, 2.0),
            EventRecord("evt_0003", "zone_exit", 1, "z1", 3.1, 3.1, 0.0),
            EventRecord("evt_0004", "line_crossing", 1, "l1:A->B", 3.2, 3.2, 0.0),
        ]
        stay_traces = [
            {
                "track_id": 1,
                "zone_id": "z1",
                "enter_time": 1.0,
                "duration_seconds": 2.0,
                "peak_state": "triggered",
                "track_lost_in_zone": False,
            }
        ]
        res = evaluate_qualitative_events(events, cfg, stay_traces=stay_traces)
        self.assertEqual(res.total_events, 4)
        self.assertEqual(res.zone_entry_accuracy, 1.0)
        self.assertEqual(res.zone_exit_accuracy, 1.0)
        self.assertEqual(res.line_crossing_accuracy, 1.0)
        self.assertTrue(res.dwell_count_matches)
        self.assertTrue(res.track_lost_in_zone_matches)
        self.assertTrue(res.dwell_within_tolerance)


class TestCollectStageFpsTable(unittest.TestCase):
    """Test collect_stage_fps_table runtime share normalization and defaults."""

    def test_collect_stage_fps_table_with_overrides_and_defaults(self) -> None:
        overrides = [
            StageFpsRecord(
                "01_test",
                "Stage 1",
                "cuda:0",
                100,
                2.0,
                50.0,
                20.0,
            ),
            StageFpsRecord(
                "02_test",
                "Stage 2",
                "cpu",
                100,
                8.0,
                12.5,
                80.0,
            ),
        ]
        table = collect_stage_fps_table(stage_overrides=overrides)
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(table[0].runtime_share_pct, 20.0, places=2)
        self.assertAlmostEqual(table[1].runtime_share_pct, 80.0, places=2)

        with tempfile.TemporaryDirectory() as tmp:
            default_table = collect_stage_fps_table(project_root=Path(tmp))
            self.assertEqual(len(default_table), 8)
            total_share = sum(r.runtime_share_pct for r in default_table)
            self.assertAlmostEqual(total_share, 100.0, delta=0.1)


class TestExportFailureExamples(unittest.TestCase):
    """Test export_failure_examples annotated image creation."""

    def test_export_failure_examples_writes_four_annotated_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq_dir = make_sequence(root, name="mot_seq", count=5)
            vid_path = make_video(root, name="surv.avi", count=5)
            gt = {23: [TrackDetection(23, 28, 10.0, 10.0, 40.0, 40.0, 1.0)]}
            pred = {23: [TrackDetection(23, 12, 11.0, 11.0, 41.0, 41.0, 0.9)]}
            cases = export_failure_examples(
                root / "out",
                mot17_source=seq_dir,
                surveillance_source=vid_path,
                gt_by_frame=gt,
                pred_by_frame=pred,
            )
            self.assertEqual(len(cases), 4)
            for c in cases:
                img_file = root / "out" / c.image_path
                self.assertTrue(img_file.is_file())
                self.assertGreater(img_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
