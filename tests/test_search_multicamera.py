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

    def test_cli_filter_insights_by_clips_and_storyboards_flags(self) -> None:
        insert_insights(
            self.db_path,
            [
                {
                    "insight_id": "ins_clip_01",
                    "camera_id": "lobby",
                    "insight_type": "loitering",
                    "severity": "critical",
                    "track_id": 1,
                    "zone_id": "lobby",
                    "start_time": 0.0,
                    "end_time": 12.0,
                    "duration": 12.0,
                    "description": "Loitering with video clip",
                    "video_clip_path": "clips/ins_clip_01.mp4",
                    "storyboard_path": "storyboards/ins_clip_01_sb.jpg",
                }
            ],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = search_main(
                ["--db", str(self.db_path), "--insights", "--clips", "--json"]
            )
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["insight_id"], "ins_clip_01")
        self.assertEqual(data[0]["video_clip_path"], "clips/ins_clip_01.mp4")

        buf_text = io.StringIO()
        with redirect_stdout(buf_text):
            rc_t = search_main(
                ["--db", str(self.db_path), "--insights", "--storyboards"]
            )
        self.assertEqual(rc_t, 0)
        text_out = buf_text.getvalue()
        self.assertIn("Matched 1 insight(s):", text_out)
        self.assertIn("clip=clips/ins_clip_01.mp4", text_out)
        self.assertIn("sb=storyboards/ins_clip_01_sb.jpg", text_out)

    def test_cli_trajectories_bbox_zone_line_stats_and_track_timeline(self) -> None:
        from event_intelligence.storage import ingest_intelligence_db

        tracks_f = self.tmp / "tracks_full.csv"
        tracks_f.write_text(
            "frame_id,track_id,x1,y1,x2,y2,confidence\n"
            "0,1,10,10,30,50,0.9\n"
            "10,1,40,50,60,90,0.95\n",
            encoding="utf-8",
        )
        traj_f = self.tmp / "traj_full.csv"
        traj_f.write_text(
            "track_id,frame_id,timestamp,x,y\n"
            "1,0,0.0,20.0,50.0\n"
            "1,10,1.0,50.0,90.0\n"
            "2,5,0.5,250.0,300.0\n",
            encoding="utf-8",
        )
        events_f = self.tmp / "events_full.csv"
        events_f.write_text(
            "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
            "ev_f1,dwell_time_event,1,lobby,0.0,1.0,1.0,snap.jpg,crop.jpg\n",
            encoding="utf-8",
        )
        zc_f = self.tmp / "zc_full.csv"
        zc_f.write_text(
            "zone_id,entry_count,exit_count,total_dwell_seconds,peak_occupancy\n"
            "lobby,3,2,9.5,2\n",
            encoding="utf-8",
        )
        lc_f = self.tmp / "lc_full.csv"
        lc_f.write_text(
            "line_id,crossings_ab,crossings_ba,total\ngate_a,4,1,5\n",
            encoding="utf-8",
        )
        full_db = self.tmp / "full_search.db"
        cfg = SearchConfig(
            source_name="lobby_cam.mp4",
            camera_id="lobby",
            tracks_csv=str(tracks_f),
            trajectories_csv=str(traj_f),
            events_csv=str(events_f),
            zone_counts_csv=str(zc_f),
            line_counts_csv=str(lc_f),
            detections_csv="",
            tracking_summary_csv="",
            trajectory_summary_csv="",
        )
        ingest_intelligence_db(full_db, cfg, project_root=self.tmp)

        # 1. --trajectories & --bbox
        buf_traj = io.StringIO()
        with redirect_stdout(buf_traj):
            rc = search_main(["--db", str(full_db), "--bbox", "0,0,100,100", "--json"])
        self.assertEqual(rc, 0)
        pts = json.loads(buf_traj.getvalue())
        self.assertEqual(len(pts), 2)
        self.assertEqual(pts[0]["track_id"], 1)

        # 2. --zone-stats
        buf_zs = io.StringIO()
        with redirect_stdout(buf_zs):
            rc_zs = search_main(
                ["--db", str(full_db), "--zone-stats", "--zone", "lobby", "--json"]
            )
        self.assertEqual(rc_zs, 0)
        z_rows = json.loads(buf_zs.getvalue())
        self.assertEqual(len(z_rows), 1)
        self.assertEqual(z_rows[0]["peak_occupancy"], 2)

        # 3. --line-stats
        buf_ls = io.StringIO()
        with redirect_stdout(buf_ls):
            rc_ls = search_main(
                ["--db", str(full_db), "--line-stats", "--line", "gate_a", "--json"]
            )
        self.assertEqual(rc_ls, 0)
        l_rows = json.loads(buf_ls.getvalue())
        self.assertEqual(len(l_rows), 1)
        self.assertEqual(l_rows[0]["total"], 5)

        # 4. --track-timeline (JSON & text)
        buf_tt = io.StringIO()
        with redirect_stdout(buf_tt):
            rc_tt = search_main(
                ["--db", str(full_db), "--track-timeline", "1", "--json"]
            )
        self.assertEqual(rc_tt, 0)
        dossiers = json.loads(buf_tt.getvalue())
        self.assertEqual(len(dossiers), 1)
        self.assertEqual(dossiers[0]["track_id"], 1)
        self.assertAlmostEqual(dossiers[0]["mean_speed_px_s"], 50.0)

        buf_tt_txt = io.StringIO()
        with redirect_stdout(buf_tt_txt):
            rc_txt = search_main(["--db", str(full_db), "--track-timeline", "1"])
        self.assertEqual(rc_txt, 0)
        self.assertIn("Track Timeline Dossier: track_id=1", buf_tt_txt.getvalue())

    def test_cli_polygon_activity_buckets_and_export_incident(self) -> None:
        from event_intelligence.storage.search_events import ingest_intelligence_db

        (self.tmp / "snap.jpg").write_bytes(b"jpg-data")
        (self.tmp / "crop.jpg").write_bytes(b"crop-data")
        tracks_f = self.tmp / "tracks_11g.csv"
        tracks_f.write_text(
            "frame_id,track_id,x1,y1,x2,y2,confidence\n"
            "0,1,10,10,30,50,0.9\n"
            "10,1,40,50,60,90,0.95\n",
            encoding="utf-8",
        )
        traj_f = self.tmp / "traj_11g.csv"
        traj_f.write_text(
            "track_id,frame_id,timestamp,x,y\n1,0,1.0,20.0,20.0\n1,10,6.5,80.0,80.0\n",
            encoding="utf-8",
        )
        events_f = self.tmp / "events_11g.csv"
        events_f.write_text(
            "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
            "ev_11g,dwell_time_event,1,lobby,1.0,6.5,5.5,snap.jpg,crop.jpg\n",
            encoding="utf-8",
        )
        db_11g = self.tmp / "search_11g.db"
        cfg = SearchConfig(
            source_name="cam_11g.mp4",
            camera_id="cam_11g",
            tracks_csv=str(tracks_f),
            trajectories_csv=str(traj_f),
            events_csv=str(events_f),
            detections_csv="",
            tracking_summary_csv="",
            trajectory_summary_csv="",
            zone_counts_csv="",
            line_counts_csv="",
        )
        ingest_intelligence_db(db_11g, cfg, project_root=self.tmp)

        # 1. --polygon (triangle covers (20,20) but excludes (80,80))
        buf_poly = io.StringIO()
        with redirect_stdout(buf_poly):
            rc_poly = search_main(
                [
                    "--db",
                    str(db_11g),
                    "--polygon",
                    "0,0;100,0;0,100",
                    "--json",
                ]
            )
        self.assertEqual(rc_poly, 0)
        poly_rows = json.loads(buf_poly.getvalue())
        self.assertEqual(len(poly_rows), 1)
        self.assertEqual((poly_rows[0]["x"], poly_rows[0]["y"]), (20.0, 20.0))

        # 2. --activity-buckets (JSON & text)
        buf_act = io.StringIO()
        with redirect_stdout(buf_act):
            rc_act = search_main(
                ["--db", str(db_11g), "--activity-buckets", "5.0", "--json"]
            )
        self.assertEqual(rc_act, 0)
        act_rows = json.loads(buf_act.getvalue())
        self.assertEqual(len(act_rows), 2)
        self.assertEqual(act_rows[0]["dwell_events_count"], 1)

        buf_act_txt = io.StringIO()
        with redirect_stdout(buf_act_txt):
            rc_act_txt = search_main(["--db", str(db_11g), "--activity-buckets", "5.0"])
        self.assertEqual(rc_act_txt, 0)
        self.assertIn("Temporal Activity Histogram", buf_act_txt.getvalue())

        # 3. --export-incident & --bundle-dir
        bundle_out = self.tmp / "exported_case"
        buf_exp = io.StringIO()
        with redirect_stdout(buf_exp):
            rc_exp = search_main(
                [
                    "--db",
                    str(db_11g),
                    "--export-incident",
                    "ev_11g",
                    "--bundle-dir",
                    str(bundle_out),
                    "--evidence-root",
                    str(self.tmp),
                    "--json",
                ]
            )
        self.assertEqual(rc_exp, 0)
        manifest = json.loads(buf_exp.getvalue())
        self.assertEqual(manifest["incident_id"], "ev_11g")
        self.assertTrue((bundle_out / "manifest.json").is_file())
        self.assertTrue((bundle_out / "incident_report.md").is_file())
        self.assertTrue((bundle_out / "trajectories.csv").is_file())
        self.assertIn("snapshot", manifest["copied_media"])

    def test_cli_cameras_flag_json_and_text(self) -> None:
        buf_json = io.StringIO()
        with redirect_stdout(buf_json):
            rc = search_main(["--db", str(self.db_path), "--cameras", "--json"])
        self.assertEqual(rc, 0)
        cams = json.loads(buf_json.getvalue())
        self.assertEqual([c["camera_id"] for c in cams], ["corridor", "lobby"])

        buf_txt = io.StringIO()
        with redirect_stdout(buf_txt):
            rc_txt = search_main(
                ["--db", str(self.db_path), "--cameras", "--camera", "lobby"]
            )
        self.assertEqual(rc_txt, 0)
        self.assertIn("Registered Cameras (1)", buf_txt.getvalue())
        self.assertIn("[lobby]", buf_txt.getvalue())


if __name__ == "__main__":
    unittest.main()
