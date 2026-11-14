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


if __name__ == "__main__":
    unittest.main()
