"""Unit tests for multi-camera SQLite storage and cross-camera queries (Phase 11c)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from event_intelligence.insights import InsightRecord, InsightType, Severity
from event_intelligence.storage import (
    CameraMetadata,
    InsightSearchResult,
    SearchConfig,
    ingest_intelligence_db,
    ingest_multi_camera_intelligence_db,
    init_intelligence_db,
    insert_insights,
    lookup_event_evidence,
    query_camera_summary,
    query_events,
    query_insights,
    query_tracks,
)


class TestMultiCameraStorage(unittest.TestCase):
    """Test multi-camera intelligence.db schema, ingestion, and cross-camera queries."""

    def test_single_camera_ingestion_registers_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            tracks_csv = tmp / "tracks.csv"
            tracks_csv.write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n"
                "0,1,10.0,10.0,20.0,30.0,0.95\n"
                "1,1,12.0,10.0,22.0,30.0,0.94\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="camera_north.mp4",
                camera_id="cam_north",
                tracks_csv=str(tracks_csv),
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                events_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            meta = CameraMetadata(
                camera_id="cam_north",
                source_name="camera_north.mp4",
                fps=25.0,
                width=1280,
                height=720,
                total_frames=100,
            )
            summary = ingest_intelligence_db(db_path, cfg, camera_metadata=meta)
            self.assertEqual(summary.tracks_count, 1)
            self.assertEqual(summary.cameras_count, 1)

            cam_summaries = query_camera_summary(db_path)
            self.assertEqual(len(cam_summaries), 1)
            self.assertEqual(cam_summaries[0].camera_id, "cam_north")
            self.assertEqual(cam_summaries[0].total_tracks, 1)

    def test_multi_camera_ingestion_preserves_independent_tracks_and_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"

            # Camera 1 files
            cam1_tracks = tmp / "cam1_tracks.csv"
            cam1_tracks.write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n"
                "0,1,10.0,10.0,20.0,30.0,0.9\n"
                "1,1,12.0,10.0,22.0,30.0,0.9\n",
                encoding="utf-8",
            )
            cam1_events = tmp / "cam1_events.csv"
            cam1_events.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "ev_001,dwell_time_event,1,zone_a,0.0,2.5,2.5,snap1.jpg,crop1.jpg\n",
                encoding="utf-8",
            )
            cfg1 = SearchConfig(
                source_name="cam_entrance.mp4",
                camera_id="entrance",
                tracks_csv=str(cam1_tracks),
                events_csv=str(cam1_events),
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )

            # Camera 2 files (uses the same track_id=1 and event_id=ev_001 to verify no primary key collisions)
            cam2_tracks = tmp / "cam2_tracks.csv"
            cam2_tracks.write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n"
                "0,1,50.0,50.0,60.0,70.0,0.85\n"
                "1,1,52.0,50.0,62.0,70.0,0.88\n",
                encoding="utf-8",
            )
            cam2_events = tmp / "cam2_events.csv"
            cam2_events.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "ev_001,dwell_time_event,1,zone_b,1.0,4.0,3.0,snap2.jpg,crop2.jpg\n",
                encoding="utf-8",
            )
            cfg2 = SearchConfig(
                source_name="cam_concourse.mp4",
                camera_id="concourse",
                tracks_csv=str(cam2_tracks),
                events_csv=str(cam2_events),
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )

            summary = ingest_multi_camera_intelligence_db(
                db_path,
                [cfg1, cfg2],
                cameras_metadata=[
                    CameraMetadata("entrance", "cam_entrance.mp4", 10.0, 640, 480, 50),
                    CameraMetadata(
                        "concourse", "cam_concourse.mp4", 10.0, 640, 480, 50
                    ),
                ],
            )
            self.assertEqual(summary.cameras_count, 2)
            self.assertEqual(summary.tracks_count, 2)
            self.assertEqual(summary.events_count, 2)

            # Query all events
            all_events = query_events(db_path)
            self.assertEqual(len(all_events), 2)

            # Filter by camera_id
            entrance_events = query_events(db_path, camera_id="entrance")
            self.assertEqual(len(entrance_events), 1)
            self.assertEqual(entrance_events[0].camera_id, "entrance")
            self.assertEqual(entrance_events[0].zone_id, "zone_a")

            concourse_events = query_events(db_path, camera_id="concourse")
            self.assertEqual(len(concourse_events), 1)
            self.assertEqual(concourse_events[0].camera_id, "concourse")
            self.assertEqual(concourse_events[0].zone_id, "zone_b")

            # Query tracks by camera
            entrance_tracks = query_tracks(db_path, camera_id="entrance")
            self.assertEqual(len(entrance_tracks), 1)
            self.assertEqual(entrance_tracks[0].camera_id, "entrance")

            # Activity summary across both cameras
            act_summaries = query_camera_summary(db_path)
            self.assertEqual(len(act_summaries), 2)
            summary_by_id = {s.camera_id: s for s in act_summaries}
            self.assertEqual(summary_by_id["entrance"].total_events, 1)
            self.assertEqual(summary_by_id["entrance"].total_dwell_seconds, 2.5)
            self.assertEqual(summary_by_id["concourse"].total_events, 1)
            self.assertEqual(summary_by_id["concourse"].total_dwell_seconds, 3.0)

    def test_insights_table_insertion_and_query_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "intelligence.db"
            init_intelligence_db(db_path)

            records = [
                InsightRecord(
                    insight_id="ins_001",
                    camera_id="cam_east",
                    insight_type=InsightType.LOITERING.value,
                    severity=Severity.HIGH.value,
                    track_id=1,
                    zone_id="lobby",
                    start_time=2.0,
                    end_time=15.0,
                    duration=13.0,
                    description="East lobby loitering",
                    evidence_snapshot_path="snapshots/s1.jpg",
                    evidence_crop_path="crops/c1.jpg",
                    details={"loiter_seconds": 13.0},
                ),
                InsightRecord(
                    insight_id="ins_002",
                    camera_id="cam_west",
                    insight_type=InsightType.WRONG_WAY.value,
                    severity=Severity.MEDIUM.value,
                    track_id=2,
                    zone_id="exit_corridor:B->A",
                    start_time=20.0,
                    end_time=20.0,
                    duration=0.0,
                    description="West wrong way crossing",
                    evidence_snapshot_path=None,
                    evidence_crop_path=None,
                    details={"line": "exit_corridor"},
                ),
            ]

            inserted = insert_insights(db_path, records)
            self.assertEqual(inserted, 2)

            # Query all insights
            all_ins = query_insights(db_path)
            self.assertEqual(len(all_ins), 2)
            self.assertIsInstance(all_ins[0], InsightSearchResult)

            # Filter by camera_id
            east_ins = query_insights(db_path, camera_id="cam_east")
            self.assertEqual(len(east_ins), 1)
            self.assertEqual(east_ins[0].insight_id, "ins_001")
            self.assertEqual(east_ins[0].details["loiter_seconds"], 13.0)

            # Filter by insight_type
            ww_ins = query_insights(db_path, insight_type="wrong_way")
            self.assertEqual(len(ww_ins), 1)
            self.assertEqual(ww_ins[0].insight_id, "ins_002")

            # Filter by severity
            high_ins = query_insights(db_path, severity="high")
            self.assertEqual(len(high_ins), 1)
            self.assertEqual(high_ins[0].severity, "high")

            # Filter by time range
            time_ins = query_insights(db_path, min_time=18.0)
            self.assertEqual(len(time_ins), 1)
            self.assertEqual(time_ins[0].insight_id, "ins_002")

    def test_ingest_intelligence_db_with_insights_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            tracks_csv = tmp / "tracks.csv"
            tracks_csv.write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n0,1,10.0,10.0,20.0,30.0,0.95\n",
                encoding="utf-8",
            )
            ins_json = tmp / "insights.json"
            ins_payload = [
                {
                    "insight_id": "ins_json_01",
                    "camera_id": "cam_gate",
                    "insight_type": "high_speed",
                    "severity": "critical",
                    "track_id": 1,
                    "zone_id": None,
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "duration": 2.0,
                    "description": "High speed running",
                    "details": {"max_speed": 250.0},
                }
            ]
            ins_json.write_text(json.dumps(ins_payload), encoding="utf-8")

            cfg = SearchConfig(
                source_name="gate.mp4",
                camera_id="cam_gate",
                tracks_csv=str(tracks_csv),
                insights_json=str(ins_json),
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                events_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            summary = ingest_intelligence_db(db_path, cfg, project_root=tmp)
            self.assertEqual(summary.insights_count, 1)

            res = query_insights(db_path, insight_type="high_speed")
            self.assertEqual(len(res), 1)
            self.assertEqual(res[0].insight_id, "ins_json_01")
            self.assertEqual(res[0].severity, "critical")

    def test_schema_migration_adds_video_clip_and_storyboard_columns(self) -> None:
        """Verify older databases without clip columns are safely migrated."""
        import sqlite3
        from contextlib import closing

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"
            # Create a bare legacy table without video_clip_path or storyboard_path
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE events (event_id TEXT PRIMARY KEY, event_type TEXT, track_id INTEGER, "
                    "zone_id TEXT, start_time REAL, end_time REAL, duration_seconds REAL, "
                    "snapshot_path TEXT, crop_path TEXT)"
                )
                conn.execute(
                    "INSERT INTO events VALUES ('evt_legacy', 'dwell', 1, 'zone_a', 0.0, 1.0, 1.0, 's.jpg', 'c.jpg')"
                )
                conn.commit()

            # init_intelligence_db must migrate the columns non-destructively
            init_intelligence_db(db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                cursor = conn.execute("PRAGMA table_info(events)")
                cols = [row[1] for row in cursor.fetchall()]
                self.assertIn("video_clip_path", cols)
                self.assertIn("storyboard_path", cols)
                row = conn.execute(
                    "SELECT video_clip_path, storyboard_path FROM events WHERE event_id = 'evt_legacy'"
                ).fetchone()
                self.assertEqual(row, ("", ""))

    def test_ingest_and_query_events_with_clips_and_storyboards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            events_csv = tmp / "events.csv"
            events_csv.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,"
                "snapshot_path,crop_path,video_clip_path,storyboard_path\n"
                "evt_clip_01,dwell_time_event,3,zone_east,1.0,4.0,3.0,snapshots/e01.jpg,crops/e01.jpg,"
                "clips/e01.mp4,storyboards/e01_sb.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_east.mp4",
                camera_id="cam_east",
                events_csv=str(events_csv),
                tracks_csv="",
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            summary = ingest_intelligence_db(db_path, cfg, project_root=tmp)
            self.assertEqual(summary.events_count, 1)

            events = query_events(db_path, event_type="dwell_time_event")
            self.assertEqual(len(events), 1)
            ev = events[0]
            self.assertEqual(ev.video_clip_path, "clips/e01.mp4")
            self.assertEqual(ev.storyboard_path, "storyboards/e01_sb.jpg")
            self.assertEqual(ev.to_dict()["video_clip_path"], "clips/e01.mp4")

    def test_lookup_event_evidence_resolves_clips_and_storyboards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            events_csv = tmp / "events.csv"

            # Create physical evidence files
            (tmp / "snapshots").mkdir(parents=True)
            (tmp / "crops").mkdir(parents=True)
            (tmp / "clips").mkdir(parents=True)
            (tmp / "storyboards").mkdir(parents=True)
            (tmp / "snapshots" / "e02.jpg").write_bytes(b"jpeg")
            (tmp / "crops" / "e02.jpg").write_bytes(b"crop")
            (tmp / "clips" / "e02.mp4").write_bytes(b"mp4")
            (tmp / "storyboards" / "e02_sb.jpg").write_bytes(b"storyboard")

            events_csv.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,"
                "snapshot_path,crop_path,video_clip_path,storyboard_path\n"
                "evt_clip_02,zone_entry,5,zone_west,2.0,2.0,0.0,snapshots/e02.jpg,crops/e02.jpg,"
                "clips/e02.mp4,storyboards/e02_sb.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_west.mp4",
                camera_id="cam_west",
                events_csv=str(events_csv),
                tracks_csv="",
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)

            lookup = lookup_event_evidence(db_path, "evt_clip_02", evidence_root=tmp)
            self.assertTrue(lookup.is_valid)
            self.assertTrue(lookup.video_clip_exists)
            self.assertTrue(lookup.storyboard_exists)
            self.assertEqual(lookup.video_clip_size_bytes, 3)
            self.assertEqual(lookup.storyboard_size_bytes, 10)
            self.assertIn("resolved_video_clip_path", lookup.to_dict())

    def test_query_trajectories_spatial_and_temporal_filtering(self) -> None:
        from event_intelligence.storage.search_events import query_trajectories

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            traj_csv = tmp / "trajectories.csv"
            traj_csv.write_text(
                "track_id,frame_id,timestamp,x,y\n"
                "1,0,0.0,10.0,20.0\n"
                "1,1,1.0,50.0,60.0\n"
                "2,2,2.0,200.0,300.0\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_traj.mp4",
                camera_id="cam_traj",
                trajectories_csv=str(traj_csv),
                events_csv="",
                tracks_csv="",
                detections_csv="",
                tracking_summary_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)

            all_pts = query_trajectories(db_path, camera_id="cam_traj")
            self.assertEqual(len(all_pts), 3)

            bbox_pts = query_trajectories(
                db_path,
                bbox=(0.0, 0.0, 100.0, 100.0),
                min_time=0.5,
            )
            self.assertEqual(len(bbox_pts), 1)
            self.assertEqual(bbox_pts[0].track_id, 1)
            self.assertEqual(bbox_pts[0].frame_id, 1)
            self.assertEqual(bbox_pts[0].to_dict()["x"], 50.0)

            with self.assertRaises(ValueError):
                query_trajectories(db_path, bbox=(100.0, 0.0, 50.0, 100.0))

    def test_query_zone_counts_and_line_counts(self) -> None:
        from event_intelligence.storage.search_events import (
            query_line_counts,
            query_zone_counts,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            zc_csv = tmp / "zone_counts.csv"
            zc_csv.write_text(
                "zone_id,entry_count,exit_count,total_dwell_seconds,peak_occupancy\n"
                "lobby,4,3,18.5,2\n",
                encoding="utf-8",
            )
            lc_csv = tmp / "line_counts.csv"
            lc_csv.write_text(
                "line_id,crossings_ab,crossings_ba,total\ngate,5,2,7\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_stats.mp4",
                camera_id="cam_stats",
                zone_counts_csv=str(zc_csv),
                line_counts_csv=str(lc_csv),
                events_csv="",
                tracks_csv="",
                detections_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)

            z_res = query_zone_counts(db_path, camera_id="cam_stats", zone_id="lobby")
            self.assertEqual(len(z_res), 1)
            self.assertEqual(z_res[0].peak_occupancy, 2)
            self.assertEqual(z_res[0].to_dict()["total_dwell_seconds"], 18.5)

            l_res = query_line_counts(db_path, camera_id="cam_stats", line_id="gate")
            self.assertEqual(len(l_res), 1)
            self.assertEqual(l_res[0].total, 7)
            self.assertEqual(l_res[0].to_dict()["crossings_ab"], 5)

    def test_query_track_timeline_builds_complete_dossier(self) -> None:
        from event_intelligence.storage.search_events import query_track_timeline

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            tracks_csv = tmp / "tracks.csv"
            tracks_csv.write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n"
                "0,1,10,10,30,50,0.9\n"
                "10,1,40,50,60,90,0.95\n",
                encoding="utf-8",
            )
            traj_csv = tmp / "trajectories.csv"
            traj_csv.write_text(
                "track_id,frame_id,timestamp,x,y\n1,0,0.0,20.0,50.0\n1,10,1.0,50.0,90.0\n",
                encoding="utf-8",
            )
            events_csv = tmp / "events.csv"
            events_csv.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_t1,dwell_time_event,1,lobby,0.0,1.0,1.0,snap.jpg,crop.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_dossier.mp4",
                camera_id="cam_dossier",
                tracks_csv=str(tracks_csv),
                trajectories_csv=str(traj_csv),
                events_csv=str(events_csv),
                detections_csv="",
                tracking_summary_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)
            insert_insights(
                db_path,
                [
                    {
                        "insight_id": "ins_t1",
                        "camera_id": "cam_dossier",
                        "insight_type": "loitering",
                        "severity": "high",
                        "track_id": 1,
                        "zone_id": "lobby",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "duration": 1.0,
                        "description": "Track 1 loitered",
                    }
                ],
            )

            dossiers = query_track_timeline(db_path, 1, camera_id="cam_dossier")
            self.assertEqual(len(dossiers), 1)
            d = dossiers[0]
            self.assertEqual(d.track_id, 1)
            self.assertEqual(d.trajectory_points_count, 2)
            # 30-40-50 right triangle over 1.0s -> 50 px/s
            self.assertAlmostEqual(d.mean_speed_px_s, 50.0)
            self.assertEqual(len(d.events), 1)
            self.assertEqual(len(d.insights), 1)
            self.assertEqual(d.to_dict()["spatial_bounds"]["max_x"], 50.0)

            with self.assertRaises(KeyError):
                query_track_timeline(db_path, 999)

    def test_query_temporal_activity_buckets_from_sqlite(self) -> None:
        from event_intelligence.storage.search_events import query_temporal_activity

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            traj_csv = tmp / "trajectories.csv"
            traj_csv.write_text(
                "track_id,frame_id,timestamp,x,y\n"
                "1,0,1.0,20.0,20.0\n"
                "2,1,2.5,30.0,30.0\n"
                "1,2,7.0,40.0,40.0\n",
                encoding="utf-8",
            )
            events_csv = tmp / "events.csv"
            events_csv.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_b1,dwell_time_event,1,lobby,1.0,3.0,2.0,snap.jpg,crop.jpg\n"
                "evt_b2,line_crossing,1,gate:A->B,7.5,7.5,0.0,snap.jpg,crop.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_act.mp4",
                camera_id="cam_act",
                trajectories_csv=str(traj_csv),
                events_csv=str(events_csv),
                tracks_csv="",
                detections_csv="",
                tracking_summary_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)
            insert_insights(
                db_path,
                [
                    {
                        "insight_id": "ins_act1",
                        "camera_id": "cam_act",
                        "insight_type": "loitering",
                        "severity": "medium",
                        "track_id": 1,
                        "zone_id": "lobby",
                        "start_time": 6.5,
                        "end_time": 8.0,
                        "duration": 1.5,
                        "description": "Loitering",
                    }
                ],
            )

            buckets = query_temporal_activity(
                db_path, bucket_seconds=5.0, camera_id="cam_act"
            )
            self.assertEqual(len(buckets), 2)
            self.assertEqual(buckets[0].active_tracks_count, 2)
            self.assertEqual(buckets[0].dwell_events_count, 1)
            self.assertEqual(buckets[0].insights_count, 0)
            self.assertEqual(buckets[1].active_tracks_count, 1)
            self.assertEqual(buckets[1].line_crossings_count, 1)
            self.assertEqual(buckets[1].insights_count, 1)

            windowed = query_temporal_activity(
                db_path,
                bucket_seconds=5.0,
                camera_id="cam_act",
                min_time=5.0,
                max_time=10.0,
            )
            self.assertEqual(len(windowed), 1)
            self.assertEqual(windowed[0].start_time, 5.0)

    def test_query_trajectories_polygon_spatial_filtering(self) -> None:
        from event_intelligence.storage.search_events import query_trajectories

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            traj_csv = tmp / "trajectories.csv"
            # Triangle (0,0) - (100,0) - (0,100) has bbox [0,100]x[0,100].
            # (20,20) is strictly inside; (80,80) is inside bbox but OUTSIDE triangle;
            # (50,50) lies on hypotenuse (covered).
            traj_csv.write_text(
                "track_id,frame_id,timestamp,x,y\n"
                "1,0,0.0,20.0,20.0\n"
                "1,1,0.5,80.0,80.0\n"
                "2,2,1.0,50.0,50.0\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_poly.mp4",
                camera_id="cam_poly",
                trajectories_csv=str(traj_csv),
                tracks_csv="",
                detections_csv="",
                tracking_summary_csv="",
                trajectory_summary_csv="",
                events_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)

            triangle = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]
            matched = query_trajectories(
                db_path, camera_id="cam_poly", polygon=triangle
            )
            self.assertEqual(len(matched), 2)
            self.assertEqual(
                [(p.x, p.y) for p in matched], [(20.0, 20.0), (50.0, 50.0)]
            )

            limited = query_trajectories(
                db_path, camera_id="cam_poly", polygon=triangle, limit=1
            )
            self.assertEqual(len(limited), 1)
            self.assertEqual((limited[0].x, limited[0].y), (20.0, 20.0))

            with self.assertRaises(ValueError):
                query_trajectories(db_path, polygon=[(0.0, 0.0), (10.0, 10.0)])

    def test_export_incident_bundle_for_event_and_insight(self) -> None:
        from event_intelligence.storage.search_events import export_incident_bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            (tmp / "snapshots").mkdir()
            (tmp / "crops").mkdir()
            (tmp / "clips").mkdir()
            (tmp / "snapshots" / "e1.jpg").write_bytes(b"snap-bytes")
            (tmp / "crops" / "e1.jpg").write_bytes(b"crop-bytes")
            (tmp / "clips" / "ins1.mp4").write_bytes(b"clip-bytes")

            tracks_csv = tmp / "tracks.csv"
            tracks_csv.write_text(
                "frame_id,track_id,x1,y1,x2,y2,confidence\n"
                "0,1,10,10,30,40,0.9\n"
                "5,1,20,10,40,40,0.92\n",
                encoding="utf-8",
            )
            traj_csv = tmp / "trajectories.csv"
            traj_csv.write_text(
                "track_id,frame_id,timestamp,x,y\n"
                "1,0,0.0,20.0,40.0\n"
                "1,5,1.0,30.0,40.0\n",
                encoding="utf-8",
            )
            events_csv = tmp / "events.csv"
            events_csv.write_text(
                "event_id,event_type,track_id,zone_id,start_time,end_time,duration_seconds,snapshot_path,crop_path\n"
                "evt_b01,dwell_time_event,1,lobby,0.0,1.0,1.0,snapshots/e1.jpg,crops/e1.jpg\n",
                encoding="utf-8",
            )
            cfg = SearchConfig(
                source_name="cam_bundle.mp4",
                camera_id="cam_bundle",
                tracks_csv=str(tracks_csv),
                trajectories_csv=str(traj_csv),
                events_csv=str(events_csv),
                detections_csv="",
                tracking_summary_csv="",
                trajectory_summary_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_intelligence_db(db_path, cfg, project_root=tmp)
            insert_insights(
                db_path,
                [
                    {
                        "insight_id": "ins_b01",
                        "camera_id": "cam_bundle",
                        "insight_type": "loitering",
                        "severity": "high",
                        "track_id": 1,
                        "zone_id": "lobby",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "duration": 1.0,
                        "description": "Loitering alert",
                        "evidence_snapshot_path": "snapshots/e1.jpg",
                        "video_clip_path": "clips/ins1.mp4",
                        "storyboard_path": "storyboards/missing.jpg",
                    }
                ],
            )

            ev_bundle_dir = tmp / "bundle_event"
            ev_manifest = export_incident_bundle(
                db_path, "evt_b01", ev_bundle_dir, evidence_root=tmp
            )
            self.assertEqual(ev_manifest.record_kind, "event")
            self.assertEqual(ev_manifest.involved_track_ids, [1])
            self.assertIn("snapshot", ev_manifest.copied_media)
            self.assertIn("crop", ev_manifest.copied_media)
            self.assertTrue((ev_bundle_dir / "manifest.json").is_file())
            self.assertTrue((ev_bundle_dir / "incident_report.md").is_file())
            self.assertTrue((ev_bundle_dir / "trajectories.csv").is_file())
            self.assertTrue(
                (ev_bundle_dir / ev_manifest.copied_media["snapshot"]).is_file()
            )

            ins_bundle_dir = tmp / "bundle_insight"
            ins_manifest = export_incident_bundle(
                db_path, "ins_b01", ins_bundle_dir, evidence_root=tmp
            )
            self.assertEqual(ins_manifest.record_kind, "insight")
            self.assertIn("video_clip", ins_manifest.copied_media)
            self.assertEqual(
                ins_manifest.missing_media, ["storyboard:storyboards/missing.jpg"]
            )
            report_text = (ins_bundle_dir / "incident_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Incident Case Bundle", report_text)
            self.assertIn("Loitering alert", report_text)

            with self.assertRaises(KeyError):
                export_incident_bundle(db_path, "nonexistent_id", tmp / "bundle_bad")

    def test_query_cameras_returns_metadata_and_filters(self) -> None:
        from event_intelligence.storage.search_events import (
            CameraMetadata,
            ingest_multi_camera_intelligence_db,
            query_cameras,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "intelligence.db"
            cfg_a = SearchConfig(
                source_name="entrance.mp4",
                camera_id="entrance",
                detections_csv="",
                tracks_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                events_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            cfg_b = SearchConfig(
                source_name="lobby.mp4",
                camera_id="lobby",
                detections_csv="",
                tracks_csv="",
                tracking_summary_csv="",
                trajectories_csv="",
                trajectory_summary_csv="",
                events_csv="",
                zone_counts_csv="",
                line_counts_csv="",
            )
            ingest_multi_camera_intelligence_db(
                db_path,
                [cfg_a, cfg_b],
                project_root=tmp,
                cameras_metadata=[
                    CameraMetadata(
                        camera_id="entrance",
                        source_name="entrance.mp4",
                        fps=12.0,
                        width=768,
                        height=432,
                        total_frames=596,
                        created_at="2026-11-20T00:00:00+00:00",
                    ),
                    CameraMetadata(
                        camera_id="lobby",
                        source_name="lobby.mp4",
                        fps=25.0,
                        width=1280,
                        height=720,
                        total_frames=300,
                        created_at="2026-11-20T00:00:01+00:00",
                    ),
                ],
            )
            all_cams = query_cameras(db_path)
            self.assertEqual([c.camera_id for c in all_cams], ["entrance", "lobby"])
            self.assertEqual((all_cams[0].width, all_cams[0].height), (768, 432))
            self.assertEqual(all_cams[0].fps, 12.0)

            only_lobby = query_cameras(db_path, camera_id="lobby")
            self.assertEqual(len(only_lobby), 1)
            self.assertEqual(only_lobby[0].total_frames, 300)


if __name__ == "__main__":
    unittest.main()
