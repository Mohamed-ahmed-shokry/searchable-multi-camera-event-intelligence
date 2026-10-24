"""Tests for Phase 3 person tracking data structures, CSV serialization,
track summaries, anomaly detection, and MOT17 evaluation.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from track_people import (
    TRACKS_CSV_HEADER,
    FrameTracks,
    TrackDetection,
    read_tracks_csv,
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


if __name__ == "__main__":
    unittest.main()
