"""Markdown executive reporting and JSON/CSV artifact writers for multi-camera correlation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from event_intelligence.multicam.types import MultiCameraCorrelationReport


def generate_correlation_markdown(report: MultiCameraCorrelationReport) -> str:
    """Render a concise executive Markdown report for multi-camera correlation."""
    lines: list[str] = [
        "# Multi-Camera Spatial-Temporal Correlation Report",
        "",
        f"- **Database:** `{report.db_path}`",
        f"- **Topology:** `{report.topology_name}`",
        f"- **Cameras ({len(report.cameras)}):** "
        + (", ".join(f"`{c}`" for c in report.cameras) if report.cameras else "None"),
        f"- **Track Endpoints:** `{report.track_endpoints_count}`",
        f"- **Correlated Transitions:** `{len(report.transitions)}`",
        f"- **Stitched Journeys:** `{len(report.journeys)}`",
        f"- **Peak Simultaneous Cameras:** `{report.peak_simultaneous_cameras}`",
        f"- **Peak Simultaneous Active Tracks:** `{report.peak_simultaneous_tracks}`",
        "",
    ]

    if report.summary_text:
        lines.extend([f"> {report.summary_text}", ""])

    lines.append("## Camera-to-Camera Transition Flow Matrix")
    lines.append("")
    if report.pair_flows:
        lines.extend(
            [
                "| From Camera | To Camera | Transitions | Avg Transit (s) | Min / Max (s) | Avg Confidence |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for flow in report.pair_flows:
            lines.append(
                f"| `{flow.from_camera}` | `{flow.to_camera}` | {flow.transition_count} | "
                f"{flow.avg_transit_seconds:.2f} | "
                f"{flow.min_transit_seconds:.2f} / {flow.max_transit_seconds:.2f} | "
                f"{flow.avg_confidence:.2f} |"
            )
    else:
        lines.append(
            "_No cross-camera transitions detected under the active topology._"
        )
    lines.append("")

    lines.append("## Cross-Camera Transitions")
    lines.append("")
    if report.transitions:
        lines.extend(
            [
                "| ID | Source Track | Target Track | Exit (s) | Entry (s) | Transit (s) | Confidence | Rule |",
                "|---|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for tr in report.transitions[:25]:
            lines.append(
                f"| `{tr.transition_id}` | `{tr.from_camera}#{tr.from_track_id}` | "
                f"`{tr.to_camera}#{tr.to_track_id}` | {tr.exit_time:.2f} | "
                f"{tr.entry_time:.2f} | {tr.transit_seconds:.2f} | "
                f"{tr.confidence:.2f} | {tr.rule_label} |"
            )
    else:
        lines.append("_No transitions to display._")
    lines.append("")

    lines.append("## Stitched Multi-Hop Journeys")
    lines.append("")
    if report.journeys:
        lines.extend(
            [
                "| Journey ID | Path | Hops | Span (s) | Avg Confidence |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for jr in report.journeys[:20]:
            path_str = " -> ".join(jr.cameras_visited)
            lines.append(
                f"| `{jr.journey_id}` | `{path_str}` | {len(jr.hops)} | "
                f"{jr.total_span_seconds:.2f} | {jr.avg_confidence:.2f} |"
            )
    else:
        lines.append("_No multi-camera journeys stitched._")
    lines.append("")

    lines.append("## Synchronized Co-Activity Windows")
    lines.append("")
    surges = [w for w in report.co_activity_windows if w.is_simultaneous_surge]
    lines.append(
        f"Total windows: **{len(report.co_activity_windows)}** | "
        f"Simultaneous surge windows: **{len(surges)}**"
    )
    lines.append("")
    if report.co_activity_windows:
        lines.extend(
            [
                "| Window (s) | Active Cameras | Total Tracks | Total Events | Surge |",
                "|---|---|---:|---:|---|",
            ]
        )
        for win in report.co_activity_windows[:20]:
            cams_str = ", ".join(win.active_cameras) if win.active_cameras else "-"
            surge_flag = "YES" if win.is_simultaneous_surge else "no"
            lines.append(
                f"| {win.start_time:.1f}-{win.end_time:.1f} | {cams_str} | "
                f"{win.total_active_tracks} | {win.total_events} | {surge_flag} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_correlation_artifacts(
    report: MultiCameraCorrelationReport,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write JSON, Markdown, and CSV artifacts for a MultiCameraCorrelationReport."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "multicam_correlation_report.json"
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )

    md_path = out_dir / "multicam_correlation_report.md"
    md_path.write_text(
        generate_correlation_markdown(report),
        encoding="utf-8",
    )

    transitions_csv = out_dir / "cross_camera_transitions.csv"
    with transitions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transition_id",
                "from_camera",
                "from_track_id",
                "to_camera",
                "to_track_id",
                "exit_time",
                "entry_time",
                "transit_seconds",
                "confidence",
                "rule_label",
                "from_zone",
                "to_zone",
                "evidence_reason",
            ],
        )
        writer.writeheader()
        for tr in report.transitions:
            writer.writerow(tr.to_dict())

    flows_csv = out_dir / "camera_pair_flows.csv"
    with flows_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "from_camera",
                "to_camera",
                "transition_count",
                "avg_transit_seconds",
                "min_transit_seconds",
                "max_transit_seconds",
                "avg_confidence",
            ],
        )
        writer.writeheader()
        for flow in report.pair_flows:
            writer.writerow(flow.to_dict())

    windows_csv = out_dir / "co_activity_windows.csv"
    with windows_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bucket_index",
                "start_time",
                "end_time",
                "active_camera_count",
                "active_cameras",
                "total_active_tracks",
                "total_events",
                "is_simultaneous_surge",
            ],
        )
        writer.writeheader()
        for win in report.co_activity_windows:
            writer.writerow(
                {
                    "bucket_index": win.bucket_index,
                    "start_time": round(win.start_time, 3),
                    "end_time": round(win.end_time, 3),
                    "active_camera_count": len(win.active_cameras),
                    "active_cameras": ";".join(win.active_cameras),
                    "total_active_tracks": win.total_active_tracks,
                    "total_events": win.total_events,
                    "is_simultaneous_surge": win.is_simultaneous_surge,
                }
            )

    return {
        "json": json_path,
        "markdown": md_path,
        "transitions_csv": transitions_csv,
        "pair_flows_csv": flows_csv,
        "co_activity_windows_csv": windows_csv,
    }
