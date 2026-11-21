"""CLI entrypoint for multi-run comparison and cross-run baseline analytics: ``python -m event_intelligence.compare``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from event_intelligence.compare.engine import (
    analyze_multi_run_trends,
    compare_runs,
    discover_runs,
    load_regression_policy,
)
from event_intelligence.compare.report import write_trends_artifacts
from event_intelligence.compare.types import RegressionPolicy


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for python -m event_intelligence.compare."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare two event_intelligence pipeline runs (KPI, config, zone, line, "
            "insight, temporal, and spatial heatmap diff) or compute cross-run baseline "
            "trends and z-score anomalies across multiple saved runs."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline run directory or intelligence.db for pairwise comparison.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help="Path to candidate run directory or intelligence.db for pairwise comparison.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory containing saved pipeline run subdirectories for multi-run trends.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit list of run directories or intelligence.db files for multi-run trends.",
    )
    parser.add_argument(
        "-c",
        "--camera",
        dest="camera_id",
        type=str,
        default=None,
        help="Optional camera_id filter for single-camera focus within multi-camera DBs.",
    )
    parser.add_argument(
        "--bucket-seconds",
        type=float,
        default=5.0,
        help="Time bucket duration in seconds for temporal histogram comparison (default: 5.0).",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=1.5,
        help="Z-score anomaly threshold for multi-run baseline outlier flagging (default: 1.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write comparison/trends JSON, Markdown report, and heatmap_diff.png.",
    )
    parser.add_argument(
        "--no-heatmap-diff",
        action="store_true",
        help="Skip rendering diverging spatial heatmap_diff.png during pairwise comparison.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Optional YAML RegressionPolicy file to evaluate automated CV regression gates.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Evaluate regression gate (using --policy or defaults) and exit with code 2 on violation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured report as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for python -m event_intelligence.compare."""
    parser = build_parser()
    args = parser.parse_args(argv)

    has_pairwise = args.baseline is not None or args.candidate is not None
    has_trends = args.runs_root is not None or args.runs is not None

    if has_pairwise and (args.baseline is None or args.candidate is None):
        parser.error("Pairwise comparison requires both --baseline and --candidate.")

    if not has_pairwise and not has_trends:
        parser.error(
            "Specify either (--baseline and --candidate) or (--runs-root / --runs)."
        )

    if has_pairwise:
        assert args.baseline is not None and args.candidate is not None
        active_policy: RegressionPolicy | None = None
        if args.policy is not None:
            active_policy = load_regression_policy(args.policy)
        elif args.fail_on_regression:
            active_policy = RegressionPolicy()

        report = compare_runs(
            args.baseline,
            args.candidate,
            camera_id=args.camera_id,
            bucket_seconds=args.bucket_seconds,
            output_dir=args.output_dir,
            render_heatmap_diff=not args.no_heatmap_diff,
            policy=active_policy,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            kpis = report.kpi_deltas
            print(
                f"=== Run Comparison: {report.baseline.run_id} vs {report.candidate.run_id} ===\n"
                f"  Tracks: {report.baseline.tracks_count} -> {report.candidate.tracks_count} "
                f"({kpis['tracks_delta']:+d})\n"
                f"  Events: {report.baseline.events_count} -> {report.candidate.events_count} "
                f"({kpis['events_delta']:+d})\n"
                f"  Insights: {report.baseline.insights_count} -> {report.candidate.insights_count} "
                f"({kpis['insights_delta']:+d})\n"
                f"  Total Dwell: {report.baseline.total_dwell_seconds:.2f}s -> "
                f"{report.candidate.total_dwell_seconds:.2f}s "
                f"({kpis['total_dwell_seconds_delta']:+.2f}s)\n"
                f"  Config Diffs: {len(report.config_diffs)} | Zone Deltas: {len(report.zone_deltas)} | "
                f"Line Deltas: {len(report.line_deltas)}"
            )
            if report.regression_gate is not None:
                gate = report.regression_gate
                status_str = "PASS" if gate.passed else "FAIL"
                print(
                    f"  Regression Gate: {status_str} "
                    f"({gate.checks_evaluated} checks, {gate.error_count} errors)"
                )
                for v in gate.violations:
                    print(f"    - [{v.severity.upper()}] {v.rule_name}: {v.message}")

        if (
            args.fail_on_regression
            and report.regression_gate is not None
            and not report.regression_gate.passed
        ):
            return 2
        return 0

    run_paths: list[Path] = []
    if args.runs_root is not None:
        run_paths.extend(discover_runs(args.runs_root))
    if args.runs is not None:
        run_paths.extend(args.runs)

    trends = analyze_multi_run_trends(
        run_paths,
        camera_id=args.camera_id,
        z_score_threshold=args.z_threshold,
    )
    if args.output_dir is not None:
        write_trends_artifacts(trends, args.output_dir)

    if args.json:
        print(json.dumps(trends.to_dict(), indent=2))
    else:
        print(
            f"=== Multi-Run Baseline Trends ({len(trends.runs)} run(s), "
            f"|z| >= {trends.z_score_threshold:.2f}) ==="
        )
        for snap in trends.runs:
            print(
                f"  {snap.run_id}: tracks={snap.tracks_count}, events={snap.events_count}, "
                f"insights={snap.insights_count}, dwell={snap.total_dwell_seconds:.2f}s"
            )
        if trends.anomalies:
            print(f"  Flagged Anomalies ({len(trends.anomalies)}):")
            for an in trends.anomalies:
                print(
                    f"    - {an.run_id} [{an.metric_name}]={an.value:.2f} "
                    f"(mean={an.baseline_mean:.2f}, z={an.z_score:+.2f}, {an.direction})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
