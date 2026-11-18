"""Markdown and JSON report generators for pairwise run diffs and multi-run trends."""

from __future__ import annotations

import json
from pathlib import Path

from event_intelligence.compare.types import RunComparisonReport


def _signed_int(val: int) -> str:
    return f"+{val}" if val > 0 else str(val)


def _signed_float(val: float, digits: int = 2) -> str:
    return f"+{val:.{digits}f}" if val > 0 else f"{val:.{digits}f}"


def generate_comparison_markdown(report: RunComparisonReport) -> str:
    """Render an executive GitHub Flavored Markdown diff report between two pipeline runs."""
    base = report.baseline
    cand = report.candidate
    kpis = report.kpi_deltas

    lines: list[str] = [
        f"# Run Comparison Report — `{base.run_id}` vs `{cand.run_id}`",
        "",
        "## Executive KPI Delta",
        "",
        f"- **Baseline Run:** `{base.run_id}` (`{base.run_dir}`)",
        f"- **Candidate Run:** `{cand.run_id}` (`{cand.run_dir}`)",
    ]
    if report.camera_filter:
        lines.append(f"- **Camera Filter:** `{report.camera_filter}`")

    lines.extend(
        [
            "",
            "| Metric | Baseline | Candidate | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Detections | {base.detections_count} | {cand.detections_count} | `{_signed_int(int(kpis['detections_delta']))}` |",
            f"| Tracks | {base.tracks_count} | {cand.tracks_count} | `{_signed_int(int(kpis['tracks_delta']))}` |",
            f"| Trajectory Points | {base.trajectories_count} | {cand.trajectories_count} | `{_signed_int(int(kpis['trajectories_delta']))}` |",
            f"| Events | {base.events_count} | {cand.events_count} | `{_signed_int(int(kpis['events_delta']))}` |",
            f"| Behavioral Insights | {base.insights_count} | {cand.insights_count} | `{_signed_int(int(kpis['insights_delta']))}` |",
            f"| High/Critical Insights | {base.high_critical_insights_count} | {cand.high_critical_insights_count} | `{_signed_int(int(kpis['high_critical_insights_delta']))}` |",
            f"| Total Dwell (s) | {base.total_dwell_seconds:.2f} | {cand.total_dwell_seconds:.2f} | `{_signed_float(float(kpis['total_dwell_seconds_delta']))}` |",
            "",
        ]
    )

    if report.config_diffs:
        lines.extend(
            [
                "## Configuration Drift (`run_config.yaml`)",
                "",
                "| Parameter | Baseline | Candidate |",
                "| --- | --- | --- |",
            ]
        )
        for cd in report.config_diffs:
            lines.append(
                f"| `{cd.key}` | `{cd.baseline_value}` | `{cd.candidate_value}` |"
            )
        lines.append("")

    if report.zone_deltas:
        lines.extend(
            [
                "## Zone Activity Deltas",
                "",
                "| Camera | Zone | Entries (Base -> Cand / Delta) | Dwell s (Delta) | Peak Occupancy (Delta) |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for z in report.zone_deltas:
            cam_label = z.camera_id or "—"
            lines.append(
                f"| `{cam_label}` | `{z.zone_id}` | {z.baseline_entries} -> {z.candidate_entries} (`{_signed_int(z.delta_entries)}`) | "
                f"`{_signed_float(z.delta_dwell_seconds)}` | `{_signed_int(z.delta_peak_occupancy)}` |"
            )
        lines.append("")

    if report.line_deltas:
        lines.extend(
            [
                "## Line Crossing Deltas",
                "",
                "| Camera | Line | A->B Delta | B->A Delta | Total (Base -> Cand / Delta) |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for ln in report.line_deltas:
            cam_label = ln.camera_id or "—"
            lines.append(
                f"| `{cam_label}` | `{ln.line_id}` | `{_signed_int(ln.delta_ab)}` | `{_signed_int(ln.delta_ba)}` | "
                f"{ln.baseline_total} -> {ln.candidate_total} (`{_signed_int(ln.delta_total)}`) |"
            )
        lines.append("")

    if report.insight_deltas:
        lines.extend(
            [
                "## Behavioral Insight Deltas",
                "",
                "| Insight Type | Baseline | Candidate | Delta | High/Crit Delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for ins in report.insight_deltas:
            lines.append(
                f"| `{ins.insight_type}` | {ins.baseline_count} | {ins.candidate_count} | "
                f"`{_signed_int(ins.delta_count)}` | `{_signed_int(ins.delta_high_critical)}` |"
            )
        lines.append("")

    if report.temporal_deltas:
        lines.extend(
            [
                "## Temporal Activity Histogram Deltas",
                "",
                "| Window (s) | Tracks Delta | Events Delta | Insights Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for tb in report.temporal_deltas:
            lines.append(
                f"| `{tb.start_time:.1f}–{tb.end_time:.1f}` | `{_signed_int(tb.delta_active_tracks)}` | "
                f"`{_signed_int(tb.delta_events)}` | `{_signed_int(tb.delta_insights)}` |"
            )
        lines.append("")

    if report.spatial_shift is not None:
        sp = report.spatial_shift
        lines.extend(
            [
                "## Spatial Hotspot Shift",
                "",
                f"- **Grid / Resolution:** `{sp.grid_shape[0]}x{sp.grid_shape[1]}` over `{sp.frame_width}x{sp.frame_height}`",
                f"- **Mean Abs Density Shift:** `{sp.mean_abs_delta:.4f}`",
                f"- **Max Increase:** `{sp.max_increase:.4f}` at `{sp.hotspot_increase_xy}`",
                f"- **Max Decrease:** `{sp.max_decrease:.4f}` at `{sp.hotspot_decrease_xy}`",
            ]
        )
        if sp.diff_image_path:
            img_name = Path(sp.diff_image_path).name
            lines.append(f"- **Diff Heatmap:** [{img_name}]({img_name})")
        lines.append("")

    return "\n".join(lines)


def write_comparison_artifacts(
    report: RunComparisonReport,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write comparison_report.json and comparison_report.md into output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "comparison_report.json"
    md_path = out / "comparison_report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(
        generate_comparison_markdown(report).strip() + "\n", encoding="utf-8"
    )
    return json_path, md_path
