"""Tests for Phase 5 Zone and Line Reasoning: scene config serialization,
point-in-polygon evaluation, directional line crossings, boundary flicker
analysis, and spatial video rendering.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from define_scene_geometry import (
    export_scene_geometry,
    extract_reference_frame,
    main as define_geometry_main,
    render_coordinate_grid,
    render_scene_preview,
)
from extract_trajectories import TrajectoryPoint, write_trajectories_csv
from helpers import PROJECT_ROOT, make_video
from reason_zones_lines import (
    ZONE_LINE_EVENTS_CSV_HEADER,
    LineSpec,
    SceneConfig,
    ZoneLineSignal,
    ZoneSpec,
    analyze_boundary_flicker,
    check_line_crossing,
    detect_line_signals,
    detect_zone_signals,
    draw_zones_and_lines,
    evaluate_scene_signals,
    load_scene_config,
    main,
    point_in_zone,
    read_zone_line_events_csv,
    run_zone_line_reasoning,
    save_scene_config,
    write_zone_line_events_csv,
)


class SpatialGeometrySpecsTests(unittest.TestCase):
    def test_zone_spec_validates_id_vertices_and_area(self):
        z = ZoneSpec(
            id="entrance",
            polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 80.0), (0.0, 80.0)),
        )
        self.assertAlmostEqual(z.shapely_polygon.area, 8000.0)

        with self.assertRaises(ValueError):
            ZoneSpec(id="", polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)))
        with self.assertRaises(ValueError):
            ZoneSpec(id="bad", polygon=((0.0, 0.0), (10.0, 0.0)))
        with self.assertRaises(ValueError):
            # Collinear points -> zero area
            ZoneSpec(id="collinear", polygon=((0.0, 0.0), (5.0, 0.0), (10.0, 0.0)))

    def test_line_spec_validates_id_endpoints_and_length(self):
        line = LineSpec(id="doorway", points=((0.0, 0.0), (30.0, 40.0)))
        self.assertAlmostEqual(line.shapely_line.length, 50.0)

        with self.assertRaises(ValueError):
            LineSpec(id=" ", points=((0.0, 0.0), (10.0, 10.0)))
        with self.assertRaises(ValueError):
            LineSpec(id="zero", points=((5.0, 5.0), (5.0, 5.0)))

        cfg = SceneConfig(
            zones=(z := ZoneSpec("z", ((0, 0), (1, 0), (0, 1))),),
            lines=(line,),
        )
        self.assertEqual(len(cfg.zones), 1)
        self.assertEqual(cfg.zones[0], z)

    def test_load_and_save_scene_config_round_trip(self):
        default_cfg_path = (
            PROJECT_ROOT
            / "cv_projects"
            / "05_zone_line_reasoning"
            / "configs"
            / "scene_config.yaml"
        )
        cfg = load_scene_config(default_cfg_path)
        self.assertEqual(len(cfg.zones), 3)
        self.assertEqual(len(cfg.lines), 2)
        self.assertEqual(cfg.zones[0].id, "entrance_area")
        self.assertEqual(cfg.lines[0].id, "doorway_line")

        with tempfile.TemporaryDirectory() as tmp:
            out_yaml = Path(tmp) / "roundtrip.yaml"
            save_scene_config(out_yaml, cfg)
            reloaded = load_scene_config(out_yaml)
            self.assertEqual(reloaded, cfg)

            dup_zone = Path(tmp) / "dup_zone.yaml"
            dup_zone.write_text(
                "zones:\n"
                "  - id: z1\n    polygon: [[0,0],[10,0],[10,10]]\n"
                "  - id: z1\n    polygon: [[0,0],[20,0],[20,20]]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_scene_config(dup_zone)


class ZoneLineSignalCsvTests(unittest.TestCase):
    def test_signal_validation_and_csv_round_trip(self):
        self.assertEqual(
            ZONE_LINE_EVENTS_CSV_HEADER,
            [
                "frame_id",
                "track_id",
                "signal_type",
                "zone_id",
                "line_id",
                "direction",
                "timestamp",
            ],
        )
        s_enter = ZoneLineSignal(
            frame_id=10,
            track_id=1,
            signal_type="zone_enter",
            zone_id="entrance_area",
            timestamp=0.8333,
        )
        s_cross = ZoneLineSignal(
            frame_id=12,
            track_id=1,
            signal_type="line_cross",
            line_id="doorway_line",
            direction="A->B",
            timestamp=1.0,
        )
        with self.assertRaises(ValueError):
            ZoneLineSignal(frame_id=0, track_id=1, signal_type="zone_enter")
        with self.assertRaises(ValueError):
            ZoneLineSignal(
                frame_id=0,
                track_id=1,
                signal_type="line_cross",
                line_id="doorway",
            )

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "zone_line_events.csv"
            write_zone_line_events_csv(csv_path, [s_cross, s_enter])
            with csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ZONE_LINE_EVENTS_CSV_HEADER)
            self.assertEqual(
                rows[1],
                ["10", "1", "zone_enter", "entrance_area", "", "", "0.8333"],
            )
            loaded = read_zone_line_events_csv(csv_path)
            self.assertEqual(loaded, [s_enter, s_cross])


class PointInZoneTests(unittest.TestCase):
    def test_point_in_zone_interior_boundary_and_exterior(self):
        zone = ZoneSpec(
            id="box",
            polygon=((10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)),
        )
        self.assertTrue(point_in_zone(30.0, 40.0, zone))
        self.assertTrue(point_in_zone(10.0, 20.0, zone))  # vertex on boundary
        self.assertTrue(point_in_zone(30.0, 60.0, zone))  # edge on boundary
        self.assertFalse(point_in_zone(9.9, 40.0, zone))
        self.assertFalse(point_in_zone(30.0, 60.1, zone))


class DetectZoneSignalsTests(unittest.TestCase):
    def test_detect_zone_signals_enter_and_exit_transitions(self):
        zone = ZoneSpec(
            id="lobby",
            polygon=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
        )
        trajs = {
            1: [
                TrajectoryPoint(1, 0, 0.0, 10.0, 50.0),  # outside
                TrajectoryPoint(1, 1, 0.1, 30.0, 50.0),  # enter
                TrajectoryPoint(1, 2, 0.2, 60.0, 50.0),  # stay inside
                TrajectoryPoint(1, 3, 0.3, 90.0, 50.0),  # exit
            ]
        }
        sigs = detect_zone_signals(trajs, [zone])
        self.assertEqual(len(sigs), 2)
        self.assertEqual(sigs[0].signal_type, "zone_enter")
        self.assertEqual(sigs[0].frame_id, 1)
        self.assertEqual(sigs[1].signal_type, "zone_exit")
        self.assertEqual(sigs[1].frame_id, 3)


class CheckLineCrossingTests(unittest.TestCase):
    def test_check_line_crossing_directions_and_finite_segment_bounds(self):
        # Horizontal line from (10, 50) to (100, 50)
        line = LineSpec(id="gate", points=((10.0, 50.0), (100.0, 50.0)))
        # Moving from y=40 (s < 0, Side A) to y=60 (s > 0, Side B) at x=50
        self.assertEqual(
            check_line_crossing((50.0, 40.0), (50.0, 60.0), line),
            "A->B",
        )
        # Moving from y=60 (Side B) to y=40 (Side A) at x=50
        self.assertEqual(
            check_line_crossing((50.0, 60.0), (50.0, 40.0), line),
            "B->A",
        )
        # Passing outside the finite segment at x=200 -> None
        self.assertIsNone(check_line_crossing((200.0, 40.0), (200.0, 60.0), line))
        # Staying on the same side -> None
        self.assertIsNone(check_line_crossing((50.0, 30.0), (50.0, 40.0), line))


class EvaluateSceneSignalsTests(unittest.TestCase):
    def test_evaluate_scene_signals_combines_zones_and_lines_in_frame_order(self):
        cfg = SceneConfig(
            zones=(
                ZoneSpec(
                    id="zone_a",
                    polygon=((0.0, 0.0), (60.0, 0.0), (60.0, 100.0), (0.0, 100.0)),
                ),
            ),
            lines=(LineSpec(id="mid_line", points=((60.0, 0.0), (60.0, 100.0))),),
        )
        trajs = {
            2: [
                TrajectoryPoint(2, 0, 0.0, 30.0, 50.0),  # enter zone_a
                TrajectoryPoint(2, 1, 0.1, 80.0, 50.0),  # exit zone_a + cross mid_line
            ]
        }
        line_only = detect_line_signals(trajs, cfg.lines)
        self.assertEqual(len(line_only), 1)
        self.assertEqual(line_only[0].signal_type, "line_cross")
        self.assertEqual(line_only[0].line_id, "mid_line")

        combined = evaluate_scene_signals(trajs, cfg)
        self.assertEqual(
            [(s.frame_id, s.signal_type) for s in combined],
            [
                (0, "zone_enter"),
                (1, "line_cross"),
                (1, "zone_exit"),
            ],
        )


class BoundaryFlickerTests(unittest.TestCase):
    def test_analyze_boundary_flicker_flags_rapid_zone_and_line_oscillations(self):
        signals = [
            ZoneLineSignal(10, 1, "zone_enter", zone_id="z1", timestamp=1.0),
            ZoneLineSignal(12, 1, "zone_exit", zone_id="z1", timestamp=1.2),
            ZoneLineSignal(14, 1, "zone_enter", zone_id="z1", timestamp=1.4),
            ZoneLineSignal(40, 1, "zone_exit", zone_id="z1", timestamp=4.0),
            ZoneLineSignal(
                11,
                1,
                "line_cross",
                line_id="l1",
                direction="A->B",
                timestamp=1.1,
            ),
            ZoneLineSignal(
                13,
                1,
                "line_cross",
                line_id="l1",
                direction="B->A",
                timestamp=1.3,
            ),
        ]
        report = analyze_boundary_flicker(signals, max_flicker_frames=5)
        self.assertEqual(report["zone_flicker_count"], 2)
        self.assertEqual(report["line_jitter_count"], 1)
        self.assertEqual(report["zone_flicker_cases"][0]["gap_frames"], 2)
        self.assertEqual(report["line_jitter_cases"][0]["from_direction"], "A->B")
        self.assertEqual(report["line_jitter_cases"][0]["to_direction"], "B->A")

        with self.assertRaises(ValueError):
            analyze_boundary_flicker(signals, max_flicker_frames=0)


class DrawZonesAndLinesTests(unittest.TestCase):
    def test_draw_zones_and_lines_modifies_frame_and_preserves_shape(self):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        cfg = SceneConfig(
            zones=(
                ZoneSpec(
                    id="zone_a",
                    polygon=(
                        (20.0, 20.0),
                        (120.0, 20.0),
                        (120.0, 100.0),
                        (20.0, 100.0),
                    ),
                ),
            ),
            lines=(LineSpec(id="line_a", points=((140.0, 20.0), (140.0, 140.0))),),
        )
        rendered = draw_zones_and_lines(frame, cfg)
        self.assertEqual(rendered.shape, frame.shape)
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertGreater(int(rendered.sum()), 0)

        active = [
            ZoneLineSignal(0, 1, "zone_enter", zone_id="zone_a", timestamp=0.0),
            ZoneLineSignal(
                0,
                1,
                "line_cross",
                line_id="line_a",
                direction="A->B",
                timestamp=0.0,
            ),
        ]
        highlighted = draw_zones_and_lines(frame, cfg, active_signals=active)
        self.assertFalse(np.array_equal(rendered, highlighted))


class RunZoneLineReasoningPipelineTests(unittest.TestCase):
    def test_end_to_end_pipeline_and_cli_write_events_csv_and_annotated_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = make_video(root, count=4)
            trajs = {
                1: [
                    TrajectoryPoint(1, 0, 0.0, 5.0, 24.0),
                    TrajectoryPoint(1, 1, 0.1, 20.0, 24.0),
                    TrajectoryPoint(1, 2, 0.2, 45.0, 24.0),
                    TrajectoryPoint(1, 3, 0.3, 55.0, 24.0),
                ]
            }
            traj_csv = write_trajectories_csv(root / "trajectories.csv", trajs)
            cfg = SceneConfig(
                zones=(
                    ZoneSpec(
                        id="mid_zone",
                        polygon=(
                            (10.0, 10.0),
                            (40.0, 10.0),
                            (40.0, 40.0),
                            (10.0, 40.0),
                        ),
                    ),
                ),
                lines=(
                    LineSpec(
                        id="exit_line",
                        points=((40.0, 5.0), (40.0, 45.0)),
                    ),
                ),
            )
            cfg_path = save_scene_config(root / "scene_config.yaml", cfg)
            out_dir = root / "outputs"

            report = run_zone_line_reasoning(
                video_path,
                out_dir,
                trajectories_csv=traj_csv,
                scene_config_path=cfg_path,
                save_video=True,
            )
            self.assertEqual(report["processed_frames"], 4)
            self.assertEqual(report["total_signals"], 3)
            self.assertTrue((out_dir / "zone_line_events.csv").exists())
            self.assertTrue((out_dir / "scene_config.yaml").exists())
            self.assertTrue((out_dir / "scene_layout_preview.png").exists())
            self.assertTrue((out_dir / "annotated_spatial_video.mp4").exists())
            self.assertTrue((out_dir / "spatial_reasoning_summary.json").exists())

            cli_out = root / "cli_outputs"
            rc = main(
                [
                    str(video_path),
                    "--trajectories-csv",
                    str(traj_csv),
                    "--scene-config",
                    str(cfg_path),
                    "-o",
                    str(cli_out),
                    "--no-video",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((cli_out / "zone_line_events.csv").exists())


class DefineSceneGeometryTests(unittest.TestCase):
    def test_extract_frame_render_grid_preview_and_cli_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = make_video(root, count=3)
            frame0 = extract_reference_frame(video_path, frame_index=0)
            self.assertEqual(frame0.shape, (48, 64, 3))

            with self.assertRaises(ValueError):
                extract_reference_frame(video_path, frame_index=-1)
            with self.assertRaises(ValueError):
                render_coordinate_grid(frame0, step=5)

            grid = render_coordinate_grid(frame0, step=20)
            self.assertEqual(grid.shape, frame0.shape)
            self.assertFalse(np.array_equal(frame0, grid))

            cfg = SceneConfig(
                zones=(
                    ZoneSpec(
                        id="z1",
                        polygon=((5.0, 5.0), (30.0, 5.0), (30.0, 30.0), (5.0, 30.0)),
                    ),
                ),
                lines=(LineSpec(id="l1", points=((35.0, 5.0), (35.0, 40.0))),),
            )
            preview = render_scene_preview(frame0, cfg, show_grid=True, grid_step=20)
            self.assertEqual(preview.shape, frame0.shape)

            cfg_yaml = save_scene_config(root / "in_scene.yaml", cfg)
            out_dir = root / "geom_out"
            report = export_scene_geometry(
                video_path,
                out_dir,
                scene_config_path=cfg_yaml,
                frame_index=1,
                grid_step=20,
            )
            self.assertEqual(report["num_zones"], 1)
            self.assertEqual(report["num_lines"], 1)
            self.assertTrue((out_dir / "reference_frame_grid.png").exists())
            self.assertTrue((out_dir / "scene_layout_preview.png").exists())
            self.assertTrue((out_dir / "scene_config.yaml").exists())

            cli_dir = root / "geom_cli"
            rc = define_geometry_main(
                [
                    str(video_path),
                    "--scene-config",
                    str(cfg_yaml),
                    "-o",
                    str(cli_dir),
                    "--grid-step",
                    "20",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((cli_dir / "scene_layout_preview.png").exists())


if __name__ == "__main__":
    unittest.main()
