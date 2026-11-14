"""Integration tests for event_intelligence.search CLI multi-camera flags (Phase 11c)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from event_intelligence.insights import InsightRecord
from event_intelligence.search import main as search_main
from event_intelligence.storage import (
    CameraMetadata,
    SearchConfig,
    ingest_multi_camera_intelligence_db,
    insert_insights,
)


class TestSearchMultiCameraCli(unittest.TestCase):
    """Test CLI execution of --camera and --camera-summary against multi-camera intelligence.db."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        self.db_path = self.tmp / "multicam_intelligence.db"

        # Setup Camera A
        tracks_a = self.tmp / "tracks_a.csv"
        tracks_a.write_text(
            "frame_id,track_id,x1,y1,x2,y2,confidence\n"
            "0,1,10.0,10.0,20.0,30.0,0.9\n"
            "1,1,12.0,10.0,22.0,30.0,0.9\n",
            encoding="utf-8",
        )
        events_a = self.tmp / "events_a.csv"
        events_a.write_text(
            "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
            "ev_a1,dwell_time_event,1,lobby,0.0,3.0,3.0,snap_a.jpg,crop_a.jpg\n",
            encoding="utf-8",
        )
        cfg_a = SearchConfig(
            source_name="lobby_cam.mp4",
            camera_id="lobby",
            tracks_csv=str(tracks_a),
            events_csv=str(events_a),
            detections_csv="",
            tracking_summary_csv="",
            trajectories_csv="",
            trajectory_summary_csv="",
            zone_counts_csv="",
            line_counts_csv="",
        )

        # Setup Camera B
        tracks_b = self.tmp / "tracks_b.csv"
        tracks_b.write_text(
            "frame_id,track_id,x1,y1,x2,y2,confidence\n"
            "0,1,40.0,40.0,50.0,60.0,0.85\n"
            "1,1,42.0,40.0,52.0,60.0,0.85\n",
            encoding="utf-8",
        )
        events_b = self.tmp / "events_b.csv"
        events_b.write_text(
            "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
            "ev_b1,dwell_time_event,1,corridor,1.0,5.0,4.0,snap_b.jpg,crop_b.jpg\n",
            encoding="utf-8",
        )
        cfg_b = SearchConfig(
            source_name="corridor_cam.mp4",
            camera_id="corridor",
            tracks_csv=str(tracks_b),
            events_csv=str(events_b),
            detections_csv="",
            tracking_summary_csv="",
            trajectories_csv="",
            trajectory_summary_csv="",
            zone_counts_csv="",
            line_counts_csv="",
        )

        ingest_multi_camera_intelligence_db(
            self.db_path,
            [cfg_a, cfg_b],
            cameras_metadata=[
                CameraMetadata("lobby", "lobby_cam.mp4", 15.0, 640, 480, 100),
                CameraMetadata("corridor", "corridor_cam.mp4", 15.0, 640, 480, 100),
            ],
        )

        insert_insights(
            self.db_path,
            [
                InsightRecord(
                    insight_id="ins_lob_01",
                    camera_id="lobby",
                    insight_type="loitering",
                    severity="high",
                    track_id=1,
                    zone_id="lobby",
                    start_time=1.0,
                    end_time=15.0,
                    duration=14.0,
                    description="Loitering detected in lobby",
                ),
                InsightRecord(
                    insight_id="ins_cor_01",
                    camera_id="corridor",
                    insight_type="wrong_way",
                    severity="medium",
                    track_id=1,
                    zone_id="corridor_line:B->A",
                    start_time=20.0,
                    end_time=20.0,
                    duration=0.0,
                    description="Wrong way line crossing in corridor",
                ),
            ],
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_cli_camera_summary_json(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(["--db", str(self.db_path), "--camera-summary", "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 2)
        cam_ids = {item["camera_id"] for item in data}
        self.assertEqual(cam_ids, {"lobby", "corridor"})

    def test_cli_camera_summary_text(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(["--db", str(self.db_path), "--camera-summary"])
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("=== Camera Activity Summary (2 cameras) ===", output)
        self.assertIn("Camera [lobby]", output)
        self.assertIn("Camera [corridor]", output)

    def test_cli_filter_events_by_camera(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(["--db", str(self.db_path), "--camera", "lobby", "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["camera_id"], "lobby")
        self.assertEqual(data[0]["event_id"], "ev_a1")

    def test_cli_filter_tracks_by_camera(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(
                [
                    "--db",
                    str(self.db_path),
                    "--camera",
                    "corridor",
                    "--tracks",
                    "--json",
                ]
            )
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["camera_id"], "corridor")

    def test_cli_query_insights_json(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(["--db", str(self.db_path), "--insights", "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 2)
        types = {item["insight_type"] for item in data}
        self.assertEqual(types, {"loitering", "wrong_way"})

    def test_cli_filter_insights_by_type_and_severity(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(
                [
                    "--db",
                    str(self.db_path),
                    "--insight-type",
                    "loitering",
                    "--severity",
                    "high",
                    "--json",
                ]
            )
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["insight_id"], "ins_lob_01")
        self.assertEqual(data[0]["camera_id"], "lobby")

    def test_cli_query_insights_text_formatting(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(["--db", str(self.db_path), "--insights"])
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("Matched 2 insight(s):", output)
        self.assertIn("[ins_lob_01] cam=lobby type=loitering sev=high", output)
        self.assertIn("[ins_cor_01] cam=corridor type=wrong_way sev=medium", output)

    def test_cli_filter_events_by_clips_flag(self) -> None:
        events_clip = self.tmp / "events_clip.csv"
        events_clip.write_text(
            "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,"
            "snapshot_path,crop_path,video_clip_path,storyboard_path\n"
            "ev_clip_01,dwell_time_event,1,lobby,0.0,3.0,3.0,snap_c.jpg,crop_c.jpg,clips/c01.mp4,storyboards/c01_sb.jpg\n"
            "ev_no_clip,zone_entry,2,lobby,1.0,1.0,0.0,snap_d.jpg,crop_d.jpg,,\n",
            encoding="utf-8",
        )
        clip_db = self.tmp / "clips_test.db"
        cfg = SearchConfig(
            source_name="lobby_cam.mp4",
            camera_id="lobby",
            events_csv=str(events_clip),
            tracks_csv="",
            detections_csv="",
            tracking_summary_csv="",
            trajectories_csv="",
            trajectory_summary_csv="",
            zone_counts_csv="",
            line_counts_csv="",
        )
        from event_intelligence.storage import ingest_intelligence_db

        ingest_intelligence_db(clip_db, cfg, project_root=self.tmp)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(["--db", str(clip_db), "--clips", "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["event_id"], "ev_clip_01")
        self.assertEqual(data[0]["video_clip_path"], "clips/c01.mp4")
        self.assertEqual(data[0]["storyboard_path"], "storyboards/c01_sb.jpg")

        # Test text output shows clip label
        buf_text = io.StringIO()
        with redirect_stdout(buf_text):
            rc = search_main(["--db", str(clip_db), "--clips"])
        self.assertEqual(rc, 0)
        text_out = buf_text.getvalue()
        self.assertIn("Matched 1 event(s):", text_out)
        self.assertIn("clip=clips/c01.mp4", text_out)
        self.assertIn("sb=storyboards/c01_sb.jpg", text_out)


if __name__ == "__main__":
    unittest.main()
