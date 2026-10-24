"""Tests for Phase 2 person detection data structures, CSV serialization,
and schema validation.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from detect_people import (
    DETECTIONS_CSV_HEADER,
    Detection,
    FrameDetections,
    read_detections_csv,
    write_detections_csv,
)


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
