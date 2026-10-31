"""Phase 5: Visual scene geometry authoring helper.

Provides:
- Reference frame extraction with coordinate ruler grid (`render_coordinate_grid`)
- Annotated scene preview with vertex/endpoint coordinates (`render_scene_preview`)
- OpenCV mouse-callback interactive zone/line collector (`collect_geometry_interactively`)
- CLI to preview or interactively author `scene_config.yaml` on a sample frame
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from reason_zones_lines import (
        LineSpec,
        SceneConfig,
        VideoSource,
        ZoneSpec,
        draw_zones_and_lines,
        load_scene_config,
        save_scene_config,
    )
except ImportError:
    _subproject_5_scripts = Path(__file__).resolve().parent
    if str(_subproject_5_scripts) not in sys.path:
        sys.path.insert(0, str(_subproject_5_scripts))
    from reason_zones_lines import (
        LineSpec,
        SceneConfig,
        VideoSource,
        ZoneSpec,
        draw_zones_and_lines,
        load_scene_config,
        save_scene_config,
    )


def extract_reference_frame(
    video_path: Path | str,
    frame_index: int = 0,
) -> np.ndarray:
    """Extract a single reference frame at `frame_index` from a video or sequence."""
    if frame_index < 0:
        raise ValueError(f"frame_index must be >= 0, got {frame_index}")
    with VideoSource(video_path) as src:
        last_frame: np.ndarray | None = None
        for idx, frame in src.iter_frames(stop=frame_index + 1):
            last_frame = frame
            if idx == frame_index:
                return frame.copy()
        if last_frame is not None:
            return last_frame.copy()
        meta = src.metadata
        return np.zeros((meta.height, meta.width, 3), dtype=np.uint8)


def render_coordinate_grid(
    frame: np.ndarray,
    *,
    step: int = 50,
) -> np.ndarray:
    """Overlay a pixel-coordinate ruler grid every `step` pixels for visual authoring."""
    if step < 10:
        raise ValueError(f"step must be >= 10, got {step}")
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    grid_color = (90, 90, 90)
    major_color = (140, 140, 140)

    for x in range(0, width, step):
        is_major = (x % (step * 2)) == 0
        color = major_color if is_major else grid_color
        cv2.line(canvas, (x, 0), (x, height - 1), color, 1)
        if x > 0:
            cv2.putText(
                canvas,
                str(x),
                (x + 2, 12),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    for y in range(0, height, step):
        is_major = (y % (step * 2)) == 0
        color = major_color if is_major else grid_color
        cv2.line(canvas, (0, y), (width - 1, y), color, 1)
        if y > 0:
            cv2.putText(
                canvas,
                str(y),
                (2, y - 3),
                cv2.FONT_HERSHEY_PLAIN,
                0.8,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

    return canvas


def render_scene_preview(
    frame: np.ndarray,
    scene_config: SceneConfig,
    *,
    show_grid: bool = True,
    grid_step: int = 50,
) -> np.ndarray:
    """Render coordinate grid, zones, lines, and vertex coordinates onto a sample frame."""
    base = render_coordinate_grid(frame, step=grid_step) if show_grid else frame.copy()
    rendered = draw_zones_and_lines(base, scene_config, alpha=0.25)

    for zone in scene_config.zones:
        for vx, vy in zone.polygon:
            pt = (round(vx), round(vy))
            cv2.circle(rendered, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                rendered,
                f"({pt[0]},{pt[1]})",
                (pt[0] + 4, max(12, pt[1] - 4)),
                cv2.FONT_HERSHEY_PLAIN,
                0.75,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    for line in scene_config.lines:
        for lx, ly in line.points:
            pt = (round(lx), round(ly))
            cv2.putText(
                rendered,
                f"({pt[0]},{pt[1]})",
                (pt[0] + 4, min(rendered.shape[0] - 4, pt[1] + 12)),
                cv2.FONT_HERSHEY_PLAIN,
                0.75,
                (255, 255, 180),
                1,
                cv2.LINE_AA,
            )

    return rendered


def collect_geometry_interactively(
    reference_frame: np.ndarray,
    initial_config: SceneConfig | None = None,
) -> SceneConfig:
    """OpenCV mouse-callback session to click zone polygons and tripwire lines.

    Controls:
      - Left click: add vertex (x, y) to current buffer
      - 'z': commit current buffer (>=3 pts) as a new ZoneSpec
      - 'l': commit last 2 points in buffer as a new LineSpec
      - 'c': clear current point buffer
      - 'q' or ESC: finish and return SceneConfig
    """
    zones = list(initial_config.zones) if initial_config else []
    lines = list(initial_config.lines) if initial_config else []
    buffer_pts: list[tuple[float, float]] = []
    win_name = "Phase 5 Scene Geometry Authoring (z=zone, l=line, c=clear, q=save)"

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            buffer_pts.append((float(x), float(y)))

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _on_mouse)
    try:
        while True:
            cfg = SceneConfig(zones=tuple(zones), lines=tuple(lines))
            canvas = render_scene_preview(reference_frame, cfg, show_grid=True)
            for idx, (bx, by) in enumerate(buffer_pts):
                pt = (round(bx), round(by))
                cv2.circle(canvas, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
                if idx > 0:
                    prev = (
                        round(buffer_pts[idx - 1][0]),
                        round(buffer_pts[idx - 1][1]),
                    )
                    cv2.line(canvas, prev, pt, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("c"):
                buffer_pts.clear()
            elif key == ord("z") and len(buffer_pts) >= 3:
                zid = f"zone_{len(zones) + 1}"
                zones.append(ZoneSpec(id=zid, polygon=tuple(buffer_pts)))
                buffer_pts.clear()
            elif key == ord("l") and len(buffer_pts) >= 2:
                lid = f"line_{len(lines) + 1}"
                p0, p1 = buffer_pts[-2], buffer_pts[-1]
                lines.append(LineSpec(id=lid, points=(p0, p1)))
                buffer_pts.clear()
    finally:
        cv2.destroyWindow(win_name)

    return SceneConfig(zones=tuple(zones), lines=tuple(lines))


def export_scene_geometry(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    scene_config_path: Path | str | None = None,
    scene_config: SceneConfig | None = None,
    frame_index: int = 0,
    show_grid: bool = True,
    grid_step: int = 50,
    interactive: bool = False,
) -> dict[str, Any]:
    """Extract a sample frame, render coordinate grid + geometry preview, and save scene_config.yaml."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_frame = extract_reference_frame(video_path, frame_index=frame_index)
    if scene_config is not None:
        cfg = scene_config
    elif scene_config_path is not None and Path(scene_config_path).exists():
        cfg = load_scene_config(scene_config_path)
    else:
        cfg = SceneConfig()

    if interactive:
        cfg = collect_geometry_interactively(ref_frame, initial_config=cfg)

    grid_frame = render_coordinate_grid(ref_frame, step=grid_step)
    grid_path = out_dir / "reference_frame_grid.png"
    cv2.imwrite(str(grid_path), grid_frame)

    preview_frame = render_scene_preview(
        ref_frame,
        cfg,
        show_grid=show_grid,
        grid_step=grid_step,
    )
    preview_path = out_dir / "scene_layout_preview.png"
    cv2.imwrite(str(preview_path), preview_frame)

    yaml_out_path = save_scene_config(out_dir / "scene_config.yaml", cfg)
    height, width = ref_frame.shape[:2]

    return {
        "video_path": str(video_path),
        "frame_index": frame_index,
        "resolution": [width, height],
        "num_zones": len(cfg.zones),
        "num_lines": len(cfg.lines),
        "zones": [z.id for z in cfg.zones],
        "lines": [ln.id for ln in cfg.lines],
        "outputs": {
            "reference_frame_grid_png": str(grid_path),
            "scene_layout_preview_png": str(preview_path),
            "scene_config_yaml": str(yaml_out_path),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for define_scene_geometry.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5: Visually author or preview polygon zones and tripwire lines "
            "on a sample video frame and export scene_config.yaml."
        )
    )
    parser.add_argument(
        "video",
        help="Path to the input video file or MOT image-sequence directory.",
    )
    parser.add_argument(
        "--scene-config",
        default="cv_projects/05_zone_line_reasoning/configs/scene_config.yaml",
        help="Path to existing or target scene_config.yaml.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="cv_projects/05_zone_line_reasoning/outputs",
        help="Directory for reference_frame_grid.png, scene_layout_preview.png, and scene_config.yaml.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based frame index to extract from the video source.",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=50,
        help="Pixel spacing between coordinate ruler grid lines.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Omit coordinate grid lines from scene_layout_preview.png.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive OpenCV mouse-callback window to click zones and lines.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = export_scene_geometry(
        args.video,
        args.output_dir,
        scene_config_path=args.scene_config,
        frame_index=args.frame_index,
        show_grid=not args.no_grid,
        grid_step=args.grid_step,
        interactive=args.interactive,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
