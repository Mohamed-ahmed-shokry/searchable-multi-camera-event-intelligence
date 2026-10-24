"""Tests for Phase 3 person tracking data structures, CSV serialization,
track summaries, anomaly detection, and MOT17 evaluation.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import helpers
import numpy as np
from evaluate_mot import box_iou, compute_iou_matrix, evaluate_tracking, load_mot17_gt
from track_people import (
    TRACK_SUMMARY_CSV_HEADER,
    TRACKS_CSV_HEADER,
    FrameTracks,
    PersonTracker,
    TrackDetection,
    analyze_track_anomalies,
    compute_track_summaries,
    draw_tracks,
    load_config,
    main,
    read_track_summary_csv,
    read_tracks_csv,
    run_tracking,
    track_color,
    write_track_summary_csv,
    write_tracks_csv,
)


class TrackSchemaTests(unittest.TestCase):
    def test_valid_track_detection_and_to_row(self):
        td = TrackDetection(
            frame_id=3,
            track_id=7,
            x1=10.5,
            y1=20.25,
            x2=110.0,
            y2=220.75,
            confidence=0.875,
        )
        self.assertEqual(
            td.to_row(),
            ["3", "7", "10.50", "20.25", "110.00", "220.75", "0.8750"],
        )

    def test_invalid_track_detection_raises(self):
        with self.assertRaises(ValueError):
            TrackDetection(
                frame_id=-1, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=0.5
            )
        with self.assertRaises(ValueError):
            TrackDetection(
                frame_id=0, track_id=-2, x1=0, y1=0, x2=10, y2=10, confidence=0.5
            )
        with self.assertRaises(ValueError):
            TrackDetection(
                frame_id=0, track_id=1, x1=20, y1=0, x2=10, y2=10, confidence=0.5
            )
        with self.assertRaises(ValueError):
            TrackDetection(
                frame_id=0, track_id=1, x1=0, y1=0, x2=10, y2=10, confidence=1.5
            )

    def test_tracks_csv_round_trip_preserves_empty_frames(self):
        frames = [
            FrameTracks(
                frame_id=0,
                tracks=[
                    TrackDetection(
                        frame_id=0,
                        track_id=1,
                        x1=12.0,
                        y1=14.0,
                        x2=52.0,
                        y2=94.0,
                        confidence=0.91,
                    )
                ],
            ),
            FrameTracks(frame_id=1, tracks=[]),
            FrameTracks(
                frame_id=2,
                tracks=[
                    TrackDetection(
                        frame_id=2,
                        track_id=1,
                        x1=15.0,
                        y1=14.0,
                        x2=55.0,
                        y2=94.0,
                        confidence=0.88,
                    ),
                    TrackDetection(
                        frame_id=2,
                        track_id=2,
                        x1=100.0,
                        y1=40.0,
                        x2=140.0,
                        y2=130.0,
                        confidence=0.76,
                    ),
                ],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "tracks.csv"
            write_tracks_csv(csv_path, frames)
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], TRACKS_CSV_HEADER)
            self.assertEqual(rows[2], ["1", "", "", "", "", "", ""])

            loaded = read_tracks_csv(csv_path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(len(loaded[0].tracks), 1)
            self.assertEqual(len(loaded[1].tracks), 0)
            self.assertEqual(len(loaded[2].tracks), 2)
            self.assertEqual(loaded[2].tracks[1].track_id, 2)


class TrackSummaryTests(unittest.TestCase):
    def test_compute_and_serialize_track_summaries(self):
        frames = [
            FrameTracks(
                frame_id=0,
                tracks=[
                    TrackDetection(0, 10, 0, 0, 20, 40, 0.9),
                    TrackDetection(0, 20, 50, 0, 70, 40, 0.8),
                ],
            ),
            FrameTracks(
                frame_id=1, tracks=[TrackDetection(1, 20, 52, 0, 72, 40, 0.85)]
            ),
            FrameTracks(
                frame_id=2,
                tracks=[
                    TrackDetection(2, 10, 4, 0, 24, 40, 0.88),
                    TrackDetection(2, 20, 54, 0, 74, 40, 0.82),
                ],
            ),
        ]
        summaries = compute_track_summaries(frames, fps=10.0)
        self.assertEqual(len(summaries), 2)

        # Track 10: seen on frames 0 and 2 (gap at frame 1) -> duration_frames=3, observed=2
        s10 = summaries[0]
        self.assertEqual(s10.track_id, 10)
        self.assertEqual(s10.start_frame, 0)
        self.assertEqual(s10.end_frame, 2)
        self.assertEqual(s10.duration_frames, 3)
        self.assertAlmostEqual(s10.duration_seconds, 0.3)
        self.assertEqual(s10.observed_frames, 2)

        # Track 20: seen on frames 0, 1, 2 -> duration_frames=3, observed=3
        s20 = summaries[1]
        self.assertEqual(s20.track_id, 20)
        self.assertEqual(s20.observed_frames, 3)

        with tempfile.TemporaryDirectory() as tmp:
            summary_csv = Path(tmp) / "track_summary.csv"
            write_track_summary_csv(summary_csv, summaries)
            with summary_csv.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], TRACK_SUMMARY_CSV_HEADER)
            loaded = read_track_summary_csv(summary_csv)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].track_id, 10)
            self.assertEqual(loaded[0].observed_frames, 2)


class TrackAnomalyTests(unittest.TestCase):
    def test_detects_fragmentation_and_spatial_handoff(self):
        frames = [
            FrameTracks(
                frame_id=0,
                tracks=[
                    TrackDetection(0, 1, 10, 10, 50, 90, 0.9),
                    TrackDetection(0, 5, 200, 100, 240, 180, 0.85),
                ],
            ),
            FrameTracks(frame_id=1, tracks=[]),
            FrameTracks(
                frame_id=2,
                tracks=[
                    # Track 1 reappears after 1-frame gap at frame 1
                    TrackDetection(2, 1, 14, 10, 54, 90, 0.88),
                    # Track 6 starts right where Track 5 ended at frame 0
                    TrackDetection(2, 6, 204, 102, 244, 182, 0.86),
                ],
            ),
        ]
        anomalies = analyze_track_anomalies(frames)
        self.assertEqual(len(anomalies["fragmentations"]), 1)
        frag = anomalies["fragmentations"][0]
        self.assertEqual(frag["track_id"], 1)
        self.assertEqual(frag["lost_after_frame"], 0)
        self.assertEqual(frag["reappeared_at_frame"], 2)
        self.assertEqual(frag["gap_frames"], 1)

        self.assertEqual(len(anomalies["potential_handoffs"]), 1)
        handoff = anomalies["potential_handoffs"][0]
        self.assertEqual(handoff["old_track_id"], 5)
        self.assertEqual(handoff["new_track_id"], 6)


class PersonTrackerUnitTests(unittest.TestCase):
    def test_filters_non_person_low_conf_and_untracked_boxes(self):
        boxes = [
            # Valid tracked person (cls=0, id=4, conf=0.82)
            SimpleNamespace(
                id=[4],
                cls=[0],
                conf=[0.82],
                xyxy=[[10.0, 20.0, 50.0, 120.0]],
            ),
            # Non-person car (cls=2, id=9, conf=0.95) -> filtered out
            SimpleNamespace(
                id=[9],
                cls=[2],
                conf=[0.95],
                xyxy=[[100.0, 100.0, 200.0, 200.0]],
            ),
            # Untracked person box (id=None) -> filtered out
            SimpleNamespace(
                id=None,
                cls=[0],
                conf=[0.90],
                xyxy=[[30.0, 30.0, 60.0, 110.0]],
            ),
            # Low confidence person (conf=0.25 < 0.4) -> filtered out
            SimpleNamespace(
                id=[11],
                cls=[0],
                conf=[0.25],
                xyxy=[[70.0, 20.0, 90.0, 80.0]],
            ),
        ]
        fake_model = SimpleNamespace(
            track=lambda *_a, **_kw: [SimpleNamespace(boxes=boxes)]
        )
        tracker = PersonTracker(
            tracker_config="bytetrack.yaml",
            confidence_threshold=0.4,
            device="cpu",
            model=fake_model,
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        res = tracker.track(frame, frame_id=5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].frame_id, 5)
        self.assertEqual(res[0].track_id, 4)
        self.assertAlmostEqual(res[0].confidence, 0.82)


class DrawTracksTests(unittest.TestCase):
    def test_track_color_and_draw_tracks(self):
        c1 = track_color(1)
        c2 = track_color(2)
        self.assertNotEqual(c1, c2)
        self.assertEqual(track_color(1), track_color(11))

        blank = np.zeros((100, 120, 3), dtype=np.uint8)
        tracks = [TrackDetection(0, 1, 10, 15, 60, 85, 0.92)]
        annotated = draw_tracks(blank, tracks)
        self.assertEqual(annotated.shape, blank.shape)
        self.assertEqual(int(blank.sum()), 0)
        self.assertGreater(int(annotated.sum()), 0)


class RunTrackingPipelineTests(unittest.TestCase):
    def test_run_tracking_end_to_end_on_synthetic_sequence(self):
        call_idx = 0

        def fake_track(*_args, **_kwargs):
            nonlocal call_idx
            idx = call_idx
            call_idx += 1
            if idx == 1:
                return [SimpleNamespace(boxes=[])]
            return [
                SimpleNamespace(
                    boxes=[
                        SimpleNamespace(
                            id=[1],
                            cls=[0],
                            conf=[0.89],
                            xyxy=[[5.0 + idx, 5.0, 25.0 + idx, 40.0]],
                        )
                    ]
                )
            ]

        fake_tracker = PersonTracker(
            model_name="yolo11m.pt",
            tracker_config="bytetrack.yaml",
            device="cpu",
            model=SimpleNamespace(track=fake_track),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq_dir = helpers.make_sequence(root, count=4)
            out_dir = root / "outputs"
            summary = run_tracking(
                seq_dir,
                output_dir=out_dir,
                tracker=fake_tracker,
                save_video=True,
            )
            self.assertEqual(summary["frame_count"], 4)
            self.assertEqual(summary["num_unique_tracks"], 1)
            self.assertEqual(summary["total_track_observations"], 3)
            self.assertTrue((out_dir / "tracks.csv").is_file())
            self.assertTrue((out_dir / "track_summary.csv").is_file())
            self.assertTrue((out_dir / "annotated_tracking_video.mp4").is_file())

    def test_load_config_and_cli_missing_source_exits_2(self):
        cfg_path = (
            helpers.PROJECT_ROOT
            / "cv_projects"
            / "03_person_tracking"
            / "configs"
            / "tracking.yaml"
        )
        cfg = load_config(cfg_path)
        self.assertEqual(cfg["model"], "yolo11m.pt")
        self.assertEqual(cfg["tracker"], "bytetrack.yaml")

        with self.assertRaises(SystemExit) as ctx:
            main(["nonexistent_source_path_xyz.mp4", "--no-video"])
        self.assertEqual(ctx.exception.code, 2)


class EvaluateMotParserTests(unittest.TestCase):
    def test_load_mot17_gt_and_iou_matrix(self):
        gt_lines = (
            "1,1,100,200,50,100,1,1,1.0\n"
            "1,2,300,200,50,100,0,1,1.0\n"
            "1,3,400,200,50,100,1,3,1.0\n"
            "2,1,110,200,50,100,1,1,0.8\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            gt_file = Path(tmp) / "gt.txt"
            gt_file.write_text(gt_lines, encoding="utf-8")
            gt_by_frame = load_mot17_gt(gt_file)

        self.assertEqual(sorted(gt_by_frame.keys()), [0, 1])
        self.assertEqual(len(gt_by_frame[0]), 1)
        box0 = gt_by_frame[0][0]
        self.assertEqual(box0.track_id, 1)
        self.assertEqual(
            (box0.x1, box0.y1, box0.x2, box0.y2), (100.0, 200.0, 150.0, 300.0)
        )

        # Identical box -> IoU 1.0; disjoint box -> IoU 0.0
        disjoint = TrackDetection(0, 9, 500.0, 500.0, 550.0, 600.0, 0.9)
        self.assertAlmostEqual(box_iou(box0, box0), 1.0)
        self.assertAlmostEqual(box_iou(box0, disjoint), 0.0)
        mat = compute_iou_matrix([box0], [box0, disjoint])
        self.assertEqual(mat.shape, (1, 2))
        self.assertAlmostEqual(float(mat[0, 0]), 1.0)
        self.assertAlmostEqual(float(mat[0, 1]), 0.0)

    def test_evaluate_tracking_perfect_and_id_switch(self):
        gt = {
            0: [TrackDetection(0, 1, 10, 10, 50, 90, 1.0)],
            1: [TrackDetection(1, 1, 12, 10, 52, 90, 1.0)],
            2: [TrackDetection(2, 1, 14, 10, 54, 90, 1.0)],
        }
        perfect_pred = [
            FrameTracks(0, [TrackDetection(0, 42, 10, 10, 50, 90, 0.9)]),
            FrameTracks(1, [TrackDetection(1, 42, 12, 10, 52, 90, 0.9)]),
            FrameTracks(2, [TrackDetection(2, 42, 14, 10, 54, 90, 0.9)]),
        ]
        m_perfect = evaluate_tracking(gt, perfect_pred)
        self.assertAlmostEqual(m_perfect["mota"], 1.0)
        self.assertAlmostEqual(m_perfect["idf1"], 1.0)
        self.assertEqual(m_perfect["id_switches"], 0)

        # ID switch on frame 2 (42 -> 99) + 1 false positive on frame 2
        switched_pred = [
            FrameTracks(0, [TrackDetection(0, 42, 10, 10, 50, 90, 0.9)]),
            FrameTracks(1, [TrackDetection(1, 42, 12, 10, 52, 90, 0.9)]),
            FrameTracks(
                2,
                [
                    TrackDetection(2, 99, 14, 10, 54, 90, 0.9),
                    TrackDetection(2, 100, 300, 300, 340, 380, 0.8),
                ],
            ),
        ]
        m_switched = evaluate_tracking(gt, switched_pred)
        self.assertEqual(m_switched["id_switches"], 1)
        self.assertEqual(m_switched["fp"], 1)
        self.assertEqual(m_switched["fn"], 0)
        # MOTA = 1 - (0 + 1 + 1) / 3 = 1/3 = 0.3333
        self.assertAlmostEqual(m_switched["mota"], 0.3333, places=4)
        # IDTP = 2 (for pred 42 matched on frames 0 & 1), denom = 3 GT + 4 pred = 7 -> IDF1 = 4/7 = 0.5714
        self.assertAlmostEqual(m_switched["idf1"], 0.5714, places=4)


if __name__ == "__main__":
    unittest.main()
