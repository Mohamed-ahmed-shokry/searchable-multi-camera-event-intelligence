"""Tests for Phase 2 person detection data structures, CSV serialization,
and schema validation.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from detect_people import (
    DETECTIONS_CSV_HEADER,
    Detection,
    FrameDetections,
    PersonDetector,
    create_video_writer,
    draw_detections,
    load_config,
    main,
    read_detections_csv,
    run_detection,
    select_default_device,
    write_detections_csv,
)
from helpers import make_sequence, make_video


class DetectionSchemaTests(unittest.TestCase):
    """Test Detection dataclass, FrameDetections, and serialization logic."""

    def test_detection_to_dict_rounds_and_sets_fields(self) -> None:
        det = Detection(
            x1=10.1234,
            y1=20.5678,
            x2=100.999,
            y2=200.001,
            confidence=0.87654,
            class_name="person",
        )
        data = det.to_dict(frame_id=42)
        self.assertEqual(data["frame_id"], 42)
        self.assertEqual(data["x1"], 10.12)
        self.assertEqual(data["y1"], 20.57)
        self.assertEqual(data["x2"], 101.0)
        self.assertEqual(data["y2"], 200.0)
        self.assertEqual(data["confidence"], 0.8765)
        self.assertEqual(data["class"], "person")

    def test_frame_with_no_boxes_emits_empty_row_accounting_for_frame(self) -> None:
        fd = FrameDetections(frame_id=7, detections=[])
        rows = fd.to_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame_id"], 7)
        self.assertEqual(rows[0]["x1"], "")
        self.assertEqual(rows[0]["confidence"], "")
        self.assertEqual(rows[0]["class"], "")

    def test_frame_with_multiple_boxes_emits_row_per_box(self) -> None:
        d1 = Detection(0, 0, 10, 10, 0.9)
        d2 = Detection(20, 20, 30, 30, 0.85)
        fd = FrameDetections(frame_id=3, detections=[d1, d2])
        rows = fd.to_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["frame_id"], 3)
        self.assertEqual(rows[1]["frame_id"], 3)
        self.assertEqual(rows[0]["confidence"], 0.9)
        self.assertEqual(rows[1]["confidence"], 0.85)

    def test_csv_header_conforms_to_schema(self) -> None:
        expected = ["frame_id", "x1", "y1", "x2", "y2", "confidence", "class"]
        self.assertEqual(DETECTIONS_CSV_HEADER, expected)

    def test_write_and_read_detections_csv_round_trip(self) -> None:
        records = [
            FrameDetections(
                0,
                [Detection(12.5, 20.0, 50.0, 100.0, 0.92, "person")],
            ),
            FrameDetections(1, []),
            FrameDetections(
                2,
                [
                    Detection(10.0, 15.0, 40.0, 80.0, 0.75, "person"),
                    Detection(60.0, 70.0, 90.0, 150.0, 0.81, "person"),
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "detections.csv"
            count = write_detections_csv(records, csv_path)
            self.assertEqual(count, 3)

            # verify raw CSV content
            with open(csv_path, "r", encoding="utf-8") as f:
                raw_rows = list(csv.DictReader(f))
            self.assertEqual(len(raw_rows), 4)  # frame 0: 1, frame 1: 1, frame 2: 2

            # verify parsed content
            parsed = read_detections_csv(csv_path)
            self.assertEqual(len(parsed), 4)
            self.assertEqual(parsed[0]["frame_id"], 0)
            self.assertEqual(parsed[0]["x1"], 12.5)
            self.assertIsNone(parsed[1]["x1"])
            self.assertEqual(parsed[2]["frame_id"], 2)
            self.assertEqual(parsed[3]["frame_id"], 2)

    def test_all_video_frames_accounted_for_in_csv(self) -> None:
        # Even when frames have 0 detections, every frame from 0..N-1 is present
        total_frames = 15
        records = [FrameDetections(i, []) for i in range(total_frames)]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "detections.csv"
            write_detections_csv(records, csv_path)
            parsed = read_detections_csv(csv_path)
            unique_frames = {r["frame_id"] for r in parsed}
            self.assertEqual(len(unique_frames), total_frames)
            self.assertEqual(unique_frames, set(range(total_frames)))

    def test_read_csv_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_detections_csv("nonexistent_detections.csv")

    def test_read_csv_invalid_header_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("wrong,header,here\n1,2,3\n")
            with self.assertRaises(ValueError):
                read_detections_csv(csv_path)


class PersonDetectorFilteringTests(unittest.TestCase):
    """Test person-only filtering and confidence gating in PersonDetector."""

    def test_default_device_selection_returns_valid_string(self) -> None:
        dev = select_default_device()
        self.assertIn(dev, ("cuda:0", "cpu"))

    def test_detector_filters_non_person_classes_and_low_confidence(self) -> None:
        class MockBox:
            def __init__(self, cls_id: int, conf: float, coords: list[float]):
                self.cls = [cls_id]
                self.conf = [conf]
                self.xyxy = [coords]

        class MockResult:
            def __init__(self, boxes: list[MockBox]):
                self.boxes = boxes

        class MockYOLO:
            def __init__(self) -> None:
                self.names = {0: "person", 2: "car", 24: "backpack"}

            def predict(self, *args, **kwargs):
                return [
                    MockResult(
                        [
                            MockBox(
                                0, 0.85, [10.0, 20.0, 50.0, 100.0]
                            ),  # person, high conf -> keep
                            MockBox(
                                0, 0.35, [15.0, 25.0, 55.0, 105.0]
                            ),  # person, low conf -> reject
                            MockBox(
                                2, 0.95, [100.0, 100.0, 200.0, 200.0]
                            ),  # car, high conf -> reject
                            MockBox(
                                24, 0.90, [30.0, 40.0, 60.0, 80.0]
                            ),  # backpack -> reject
                        ]
                    )
                ]

        detector = PersonDetector(
            confidence_threshold=0.4,
            model=MockYOLO(),
        )

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(detections[0].confidence, 0.85)
        self.assertEqual((detections[0].x1, detections[0].y1), (10.0, 20.0))

    def test_detector_handles_empty_results_gracefully(self) -> None:
        class EmptyYOLO:
            def __init__(self) -> None:
                self.names = {0: "person"}

            def predict(self, *args, **kwargs):
                return [SimpleNamespace(boxes=[])]

        detector = PersonDetector(model=EmptyYOLO())
        detections = detector.detect(np.zeros((50, 50, 3), dtype=np.uint8))
        self.assertEqual(detections, [])


class AnnotationTests(unittest.TestCase):
    """Test frame bounding-box drawing and video writer creation."""

    def test_draw_detections_returns_annotated_frame(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        det = Detection(10.0, 10.0, 50.0, 60.0, 0.88, "person")
        annotated = draw_detections(frame, [det], box_color=(0, 255, 0), thickness=2)

        self.assertEqual(annotated.shape, frame.shape)
        # Bounding box pixels should no longer be purely black
        self.assertTrue(np.any(annotated > 0))

    def test_draw_detections_handles_out_of_bounds_gracefully(self) -> None:
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        # Coordinates extending past image dimensions
        det = Detection(-10.0, -10.0, 100.0, 100.0, 0.75, "person")
        annotated = draw_detections(frame, [det])
        self.assertEqual(annotated.shape, (50, 50, 3))

    def test_create_video_writer_opens_and_writes_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_out.mp4"
            writer = create_video_writer(out_path, fps=10.0, width=64, height=48)
            self.assertTrue(writer.isOpened())

            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            writer.write(frame)
            writer.write(frame)
            writer.release()

            self.assertTrue(out_path.is_file())
            self.assertGreater(out_path.stat().st_size, 0)


class DetectionPipelineIntegrationTests(unittest.TestCase):
    """Test end-to-end detection execution against video files and frame sequences."""

    def test_load_config_reads_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / "test_cfg.yaml"
            cfg_path.write_text(
                "model: yolo11s.pt\nconfidence_threshold: 0.5\ndevice: cpu\nsave_video: false\n",
                encoding="utf-8",
            )
            cfg = load_config(cfg_path)
            self.assertEqual(cfg["model"], "yolo11s.pt")
            self.assertEqual(cfg["confidence_threshold"], 0.5)
            self.assertEqual(cfg["device"], "cpu")
            self.assertFalse(cfg["save_video"])

    def test_run_detection_on_synthetic_video_generates_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = make_video(tmp, name="test_clip.avi", count=6)
            out_dir = tmp / "output"

            class MockDetector:
                model_name = "mock_model.pt"
                confidence_threshold = 0.4
                device = "cpu"

                def detect(self, frame: np.ndarray) -> list[Detection]:
                    # Return 1 detection on even frames, 0 on odd frames
                    return [Detection(10.0, 10.0, 30.0, 30.0, 0.85, "person")]

            summary = run_detection(
                video_path,
                output_dir=out_dir,
                detector=MockDetector(),
                save_video=True,
                save_sample_frames=True,
                sample_frame_interval=2,
            )

            self.assertEqual(summary["frame_count"], 6)
            self.assertEqual(summary["total_detections"], 6)
            self.assertTrue((out_dir / "detections.csv").is_file())
            self.assertTrue((out_dir / "annotated_video.mp4").is_file())
            self.assertTrue((out_dir / "detection_summary.json").is_file())

            # Read back CSV and verify exact frame match
            rows = read_detections_csv(out_dir / "detections.csv")
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["frame_id"] for r in rows], list(range(6)))

    def test_run_detection_on_synthetic_frame_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seq_dir = make_sequence(tmp, name="test_seq", count=5)
            out_dir = tmp / "seq_output"

            class ZeroDetector:
                model_name = "zero.pt"
                confidence_threshold = 0.4
                device = "cpu"

                def detect(self, frame: np.ndarray) -> list[Detection]:
                    return []

            summary = run_detection(
                seq_dir,
                output_dir=out_dir,
                detector=ZeroDetector(),
                save_video=False,
            )

            self.assertEqual(summary["frame_count"], 5)
            self.assertEqual(summary["total_detections"], 0)
            self.assertEqual(summary["frames_with_detections"], 0)

            # Every frame must still be present in the CSV
            rows = read_detections_csv(out_dir / "detections.csv")
            self.assertEqual(len(rows), 5)
            self.assertEqual([r["frame_id"] for r in rows], list(range(5)))
            for r in rows:
                self.assertIsNone(r["x1"])
                self.assertIsNone(r["confidence"])

    def test_cli_exits_cleanly_on_nonexistent_source(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["nonexistent_clip.mp4"])
        self.assertEqual(ctx.exception.code, 2)
