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
    load_evaluation_config,
    save_evaluation_config,
)


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


if __name__ == "__main__":
    unittest.main()
