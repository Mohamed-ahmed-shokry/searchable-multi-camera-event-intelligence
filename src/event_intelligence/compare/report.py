"""Markdown and JSON report generators for pairwise run diffs and multi-run trends."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from event_intelligence.compare.types import MultiRunTrendReport, RunComparisonReport


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
        ]
    )
    if (
        base.cross_camera_transitions_count > 0
        or cand.cross_camera_transitions_count > 0
        or base.cross_camera_journeys_count > 0
        or cand.cross_camera_journeys_count > 0
    ):
        lines.extend(
            [
                f"| Cross-Camera Transitions | {base.cross_camera_transitions_count} | {cand.cross_camera_transitions_count} | `{_signed_int(int(kpis['cross_camera_transitions_delta']))}` |",
                f"| Cross-Camera Journeys | {base.cross_camera_journeys_count} | {cand.cross_camera_journeys_count} | `{_signed_int(int(kpis['cross_camera_journeys_delta']))}` |",
                f"| Mean Transit (s) | {base.mean_transit_seconds:.2f} | {cand.mean_transit_seconds:.2f} | `{_signed_float(float(kpis['mean_transit_seconds_delta']))}` |",
                f"| Mean Transition Confidence | {base.mean_transition_confidence:.3f} | {cand.mean_transition_confidence:.3f} | `{_signed_float(float(kpis['mean_transition_confidence_delta']), 3)}` |",
            ]
        )
    lines.append("")

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

    if report.pair_flow_deltas:
        lines.extend(
            [
                "## Cross-Camera Pair Flow Deltas",
                "",
                "| From Camera | To Camera | Transitions (Base -> Cand / Delta) | Avg Transit s (Delta) | Avg Confidence (Delta) |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for pf in report.pair_flow_deltas:
            lines.append(
                f"| `{pf.from_camera}` | `{pf.to_camera}` | "
                f"{pf.baseline_transitions} -> {pf.candidate_transitions} "
                f"(`{_signed_int(pf.delta_transitions)}`) | "
                f"{pf.baseline_avg_transit_seconds:.2f} -> {pf.candidate_avg_transit_seconds:.2f} "
                f"(`{_signed_float(pf.delta_avg_transit_seconds)}`) | "
                f"{pf.baseline_avg_confidence:.3f} -> {pf.candidate_avg_confidence:.3f} "
                f"(`{_signed_float(pf.delta_avg_confidence, 3)}`) |"
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

    if report.regression_gate is not None:
        gate = report.regression_gate
        status_badge = "PASS" if gate.passed else "FAIL"
        lines.extend(
            [
                "## Automated Regression Gate",
                "",
                (
                    f"- **Gate Status:** **`{status_badge}`** "
                    f"(`{gate.checks_evaluated}` check(s) evaluated, "
                    f"`{gate.error_count}` error(s), `{gate.warning_count}` warning(s))"
                ),
                "",
            ]
        )
        if gate.violations:
            lines.extend(
                [
                    "| Rule | Severity | Metric | Baseline | Candidate | Threshold | Message |",
                    "| --- | --- | --- | ---: | ---: | ---: | --- |",
                ]
            )
            for v in gate.violations:
                clean_msg = v.message.replace("|", "/")
                lines.append(
                    f"| `{v.rule_name}` | **`{v.severity}`** | `{v.metric_name}` | "
                    f"{v.baseline_value:.4f} | {v.candidate_value:.4f} | "
                    f"{v.threshold:.4f} | {clean_msg} |"
                )
            lines.append("")

    return "\n".join(lines)


def write_comparison_artifacts(
    report: RunComparisonReport,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write comparison_report.json, comparison_report.md, and delta CSV tables into output_dir."""
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

    with (out / "zone_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "camera_id",
                "zone_id",
                "baseline_entries",
                "candidate_entries",
                "delta_entries",
                "baseline_exits",
                "candidate_exits",
                "delta_exits",
                "baseline_dwell_seconds",
                "candidate_dwell_seconds",
                "delta_dwell_seconds",
                "baseline_peak_occupancy",
                "candidate_peak_occupancy",
                "delta_peak_occupancy",
            ],
        )
        writer.writeheader()
        for z in report.zone_deltas:
            writer.writerow(z.to_dict())

    with (out / "line_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "camera_id",
                "line_id",
                "baseline_ab",
                "candidate_ab",
                "delta_ab",
                "baseline_ba",
                "candidate_ba",
                "delta_ba",
                "baseline_total",
                "candidate_total",
                "delta_total",
            ],
        )
        writer.writeheader()
        for ln in report.line_deltas:
            writer.writerow(ln.to_dict())

    with (out / "temporal_deltas.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bucket_index",
                "start_time",
                "end_time",
                "baseline_active_tracks",
                "candidate_active_tracks",
                "delta_active_tracks",
                "baseline_events",
                "candidate_events",
                "delta_events",
                "baseline_insights",
                "candidate_insights",
                "delta_insights",
            ],
        )
        writer.writeheader()
        for tb in report.temporal_deltas:
            writer.writerow(tb.to_dict())

    with (out / "pair_flow_deltas.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "from_camera",
                "to_camera",
                "baseline_transitions",
                "candidate_transitions",
                "delta_transitions",
                "baseline_avg_transit_seconds",
                "candidate_avg_transit_seconds",
                "delta_avg_transit_seconds",
                "baseline_avg_confidence",
                "candidate_avg_confidence",
                "delta_avg_confidence",
            ],
        )
        writer.writeheader()
        for pf in report.pair_flow_deltas:
            writer.writerow(pf.to_dict())

    return json_path, md_path


def generate_trends_markdown(report: MultiRunTrendReport) -> str:
    """Render a GitHub Flavored Markdown multi-run baseline catalog and anomaly report."""
    lines: list[str] = [
        f"# Multi-Run Baseline & Trend Analytics ({len(report.runs)} Runs)",
        "",
        f"- **Z-Score Anomaly Threshold:** `|z| >= {report.z_score_threshold:.2f}`",
        f"- **Flagged Anomalies:** **`{len(report.anomalies)}`**",
    ]
    if report.camera_filter:
        lines.append(f"- **Camera Filter:** `{report.camera_filter}`")
    lines.append("")

    if report.anomalies:
        lines.extend(
            [
                "> [!WARNING]",
                f"> **{len(report.anomalies)} Cross-Run Baseline Anomalies** detected:",
                ">",
            ]
        )
        for an in report.anomalies:
            lines.append(
                f"> - **`{an.run_id}`** `{an.metric_name}` = `{an.value:.2f}` "
                f"(baseline `{an.baseline_mean:.2f} ± {an.baseline_std:.2f}`, "
                f"`z={an.z_score:+.2f}`, `{an.direction}`)"
            )
        lines.append("")

    lines.extend(
        [
            "## Run Catalog",
            "",
            "| Run ID | Cameras | Detections | Tracks | Events | Insights | High/Crit | Total Dwell (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in report.runs:
        lines.append(
            f"| `{r.run_id}` | {r.cameras_count} | {r.detections_count} | {r.tracks_count} | "
            f"{r.events_count} | {r.insights_count} | {r.high_critical_insights_count} | "
            f"{r.total_dwell_seconds:.2f} |"
        )
    lines.append("")

    if report.metric_baselines:
        lines.extend(
            [
                "## Cross-Run Baseline Statistics",
                "",
                "| Metric | Mean | Std | Min | Max |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric_name, stats in report.metric_baselines.items():
            lines.append(
                f"| `{metric_name}` | {stats['mean']:.2f} | {stats['std']:.2f} | "
                f"{stats['min']:.2f} | {stats['max']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_trends_artifacts(
    report: MultiRunTrendReport,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write trends_report.json, trends_report.md, runs_catalog.csv, and baseline_anomalies.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "trends_report.json"
    md_path = out / "trends_report.md"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(
        generate_trends_markdown(report).strip() + "\n", encoding="utf-8"
    )

    with (out / "runs_catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "run_dir",
                "cameras_count",
                "detections_count",
                "tracks_count",
                "trajectories_count",
                "events_count",
                "insights_count",
                "high_critical_insights_count",
                "total_dwell_seconds",
                "mean_track_duration_seconds",
                "mean_path_length_px",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in report.runs:
            writer.writerow(r.to_dict())

    with (out / "baseline_anomalies.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "metric_name",
                "value",
                "baseline_mean",
                "baseline_std",
                "z_score",
                "direction",
            ],
        )
        writer.writeheader()
        for an in report.anomalies:
            writer.writerow(an.to_dict())

    return json_path, md_path
