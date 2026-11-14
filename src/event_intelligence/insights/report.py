"""Surveillance executive intelligence report and artifact generation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from event_intelligence.insights.types import InsightRecord, Severity


def generate_intelligence_report_markdown(
    insights: Sequence[InsightRecord],
    *,
    run_id: str = "default_run",
    source_info: Mapping[str, Any] | None = None,
    camera_summaries: Sequence[Any] | None = None,
    zone_counts: Sequence[Any] | None = None,
    line_counts: Sequence[Any] | None = None,
) -> str:
    """Generate a GitHub Flavored Markdown executive intelligence report."""
    src = dict(source_info or {})
    total_insights = len(insights)

    sev_counts: dict[str, int] = {s.value: 0 for s in Severity}
    for ins in insights:
        sev_counts[ins.severity] = sev_counts.get(ins.severity, 0) + 1

    crit_or_high = [
        ins
        for ins in insights
        if ins.severity in (Severity.CRITICAL.value, Severity.HIGH.value)
    ]

    lines: list[str] = [
        f"# Surveillance Event Intelligence Report — `{run_id}`",
        "",
        "## Executive Summary",
        "",
        f"- **Run ID:** `{run_id}`",
    ]
    if "source_path" in src:
        lines.append(f"- **Source:** `{src['source_path']}`")
    if "total_frames" in src:
        lines.append(f"- **Processed Frames:** `{src['total_frames']}`")
    if "elapsed_seconds" in src:
        lines.append(f"- **Processing Time:** `{src['elapsed_seconds']:.2f}s`")

    lines.extend(
        [
            f"- **Total Detected Insights / Anomalies:** **`{total_insights}`**",
            (
                f"- **Severity Breakdown:** `Critical: {sev_counts.get('critical', 0)}` | "
                f"`High: {sev_counts.get('high', 0)}` | `Medium: {sev_counts.get('medium', 0)}` | "
                f"`Low: {sev_counts.get('low', 0)}` | `Info: {sev_counts.get('info', 0)}`"
            ),
            "",
        ]
    )

    if crit_or_high:
        lines.extend(
            [
                "> [!WARNING]",
                f"> **{len(crit_or_high)} High/Critical Priority Security Alerts** require operator attention:",
                "",
            ]
        )
        for al in crit_or_high:
            links = []
            if al.evidence_snapshot_path:
                links.append(f"[Snapshot]({al.evidence_snapshot_path})")
            if al.video_clip_path:
                links.append(f"[Clip]({al.video_clip_path})")
            if al.storyboard_path:
                links.append(f"[Storyboard]({al.storyboard_path})")
            ev_link = f" ({', '.join(links)})" if links else ""
            lines.append(
                f"> - **[{al.severity.upper()}]** `[{al.camera_id}]` {al.description}{ev_link}"
            )
        lines.append("")

    lines.extend(
        [
            "## Detected Behavioral Insights & Anomalies",
            "",
            "| ID | Camera | Type | Severity | Track | Zone / Target | Time | Description | Evidence |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    if not insights:
        lines.append("| — | — | — | — | — | — | — | No anomalous events detected | — |")
    else:
        for ins in insights:
            ev_parts = []
            if ins.evidence_snapshot_path:
                ev_parts.append(f"[Snap]({ins.evidence_snapshot_path})")
            if ins.evidence_crop_path:
                ev_parts.append(f"[Crop]({ins.evidence_crop_path})")
            if ins.video_clip_path:
                ev_parts.append(f"[Clip]({ins.video_clip_path})")
            if ins.storyboard_path:
                ev_parts.append(f"[Storyboard]({ins.storyboard_path})")
            ev_str = " / ".join(ev_parts) if ev_parts else "None"

            t_str = (
                f"{ins.start_time:.1f}s"
                if ins.start_time == ins.end_time
                else f"{ins.start_time:.1f}s–{ins.end_time:.1f}s ({ins.duration:.1f}s)"
            )
            track_str = str(ins.track_id) if ins.track_id is not None else "—"
            zone_str = ins.zone_id if ins.zone_id else "—"
            clean_desc = ins.description.replace("|", "/")
            lines.append(
                f"| `{ins.insight_id}` | `{ins.camera_id}` | `{ins.insight_type}` | "
                f"**`{ins.severity}`** | `{track_str}` | `{zone_str}` | {t_str} | {clean_desc} | {ev_str} |"
            )

    lines.append("")

    if camera_summaries:
        lines.extend(
            [
                "## Camera Activity Breakdown",
                "",
                "| Camera ID | Source Name | Detections | Tracks | Events | Total Dwell (s) | Active Zones | Active Lines |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for cs in camera_summaries:
            cid = getattr(cs, "camera_id", "default")
            sname = getattr(cs, "source_name", cid)
            dets = getattr(cs, "total_detections", 0)
            trks = getattr(cs, "total_tracks", 0)
            evs = getattr(cs, "total_events", 0)
            dwell = getattr(cs, "total_dwell_seconds", 0.0)
            zactive = getattr(cs, "zones_active", 0)
            lactive = getattr(cs, "lines_active", 0)
            lines.append(
                f"| `{cid}` | `{sname}` | {dets} | {trks} | {evs} | {dwell:.1f} | {zactive} | {lactive} |"
            )
        lines.append("")

    if zone_counts:
        lines.extend(
            [
                "## Zone Activity & Occupancy",
                "",
                "| Zone ID | Entries | Exits | Total Dwell (s) | Peak Occupancy |",
                "|---|---|---|---|---|",
            ]
        )
        for z in zone_counts:
            zid = getattr(z, "zone_id", "")
            ent = getattr(z, "entry_count", 0)
            ex = getattr(z, "exit_count", 0)
            dwell = getattr(z, "total_dwell_seconds", 0.0)
            peak = getattr(z, "peak_occupancy", 0)
            lines.append(f"| `{zid}` | {ent} | {ex} | {dwell:.1f} | {peak} |")
        lines.append("")

    if line_counts:
        lines.extend(
            [
                "## Line Crossing Statistics",
                "",
                "| Line ID | Direction A->B | Direction B->A | Total Crossings |",
                "|---|---|---|---|",
            ]
        )
        for l in line_counts:
            lid = getattr(l, "line_id", "")
            ab = getattr(l, "crossings_ab", 0)
            ba = getattr(l, "crossings_ba", 0)
            tot = getattr(l, "total", 0)
            lines.append(f"| `{lid}` | {ab} | {ba} | {tot} |")
        lines.append("")

    return "\n".join(lines)


def write_intelligence_report(
    report_md: str,
    output_path: Path | str,
) -> Path:
    """Write markdown intelligence report to disk."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(report_md.strip() + "\n", encoding="utf-8")
    return out_p


def write_insights_json(
    insights: Sequence[InsightRecord],
    output_path: Path | str,
) -> Path:
    """Write insights records as structured JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = [ins.to_dict() for ins in insights]
    out_p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_p


def write_insights_csv(
    insights: Sequence[InsightRecord],
    output_path: Path | str,
) -> Path:
    """Write insights records as structured CSV."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "insight_id",
        "camera_id",
        "insight_type",
        "severity",
        "track_id",
        "zone_id",
        "start_time",
        "end_time",
        "duration",
        "description",
        "evidence_snapshot_path",
        "evidence_crop_path",
        "video_clip_path",
        "storyboard_path",
    ]
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for ins in insights:
            row = ins.to_dict()
            writer.writerow(row)
    return out_p
