"""CLI entrypoint for multi-camera spatial-temporal correlation: ``python -m event_intelligence.multicam``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from event_intelligence.multicam.engine import (
    analyze_multi_camera_correlation,
    apply_calibrated_offsets,
    estimate_camera_clock_offsets,
    infer_default_topology,
    load_camera_topology,
    save_camera_topology,
    validate_camera_topology,
)
from event_intelligence.multicam.report import (
    generate_correlation_markdown,
    write_correlation_artifacts,
)
from event_intelligence.storage.search_events import insert_cross_camera_correlation


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for python -m event_intelligence.multicam."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze multi-camera spatial-temporal track transitions, stitch "
            "multi-hop journeys, compute camera-pair flow matrices, and detect "
            "simultaneous multi-camera co-activity surges from intelligence.db."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to SQLite intelligence.db (or a run directory containing it).",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help=(
            "Optional camera_topology.yaml defining directed transition rules and "
            "camera clock offsets. Auto-infers all-pairs topology if omitted."
        ),
    )
    parser.add_argument(
        "--bucket-seconds",
        type=float,
        default=5.0,
        help="Time bucket duration in seconds for co-activity windows (default: 5.0).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Minimum transition confidence score in [0.0, 1.0] (default: 0.25).",
    )
    parser.add_argument(
        "--surge-min-cameras",
        type=int,
        default=2,
        help="Minimum active cameras to flag a simultaneous surge window (default: 2).",
    )
    parser.add_argument(
        "--surge-min-tracks",
        type=int,
        default=2,
        help="Minimum active tracks to flag a simultaneous surge window (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write JSON, Markdown, and CSV correlation artifacts.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist correlated transitions and journeys back into intelligence.db "
            "(for querying via python -m event_intelligence.search --transitions/--journeys)."
        ),
    )
    parser.add_argument(
        "--validate-topology",
        action="store_true",
        help=(
            "Validate --topology structurally and against cameras/zones in --db; "
            "exits with code 2 if validation errors are found."
        ),
    )
    parser.add_argument(
        "--calibrate-clocks",
        action="store_true",
        help=(
            "Estimate inter-camera clock-skew offsets via temporal cross-correlation "
            "across intelligence.db."
        ),
    )
    parser.add_argument(
        "--reference-camera",
        type=str,
        default=None,
        help="Optional reference camera_id (offset=0.0s) for --calibrate-clocks.",
    )
    parser.add_argument(
        "--max-lag",
        type=float,
        default=15.0,
        help="Maximum clock-skew search radius in seconds for --calibrate-clocks (default: 15.0).",
    )
    parser.add_argument(
        "--write-topology",
        type=Path,
        default=None,
        help="Optional output YAML path to save calibrated camera_time_offsets.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit structured JSON report to stdout instead of Markdown text.",
    )
    return parser


def _resolve_db_path(raw_path: Path) -> Path:
    if raw_path.is_dir():
        return raw_path / "intelligence.db"
    return raw_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the multi-camera correlation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    db_path = _resolve_db_path(args.db)
    if not db_path.exists():
        parser.error(f"SQLite database not found: {db_path}")

    topology = load_camera_topology(args.topology) if args.topology else None

    if args.validate_topology:
        if topology is None:
            parser.error("--validate-topology requires --topology <yaml>")
        val_report = validate_camera_topology(topology, db_path)
        if args.as_json:
            print(json.dumps(val_report.to_dict(), indent=2))
        else:
            status = "VALID" if val_report.valid else "INVALID"
            print(
                f"Topology Validation [{val_report.topology_name}]: {status} "
                f"({val_report.error_count} error(s), {val_report.warning_count} warning(s))"
            )
            for issue in val_report.issues:
                print(f"  - [{issue.severity.upper()}] {issue.code}: {issue.message}")
        return 0 if val_report.valid else 2

    if args.calibrate_clocks:
        cal_report = estimate_camera_clock_offsets(
            db_path,
            reference_camera=args.reference_camera,
            max_lag_seconds=args.max_lag,
        )
        written_topo_path: str | None = None
        if args.write_topology is not None:
            base_topo = topology or infer_default_topology(
                list(cal_report.recommended_offsets.keys())
            )
            calibrated_topo = apply_calibrated_offsets(base_topo, cal_report)
            written_topo_path = str(
                save_camera_topology(calibrated_topo, args.write_topology)
            )
        if args.as_json:
            payload = cal_report.to_dict()
            if written_topo_path is not None:
                payload["written_topology_path"] = written_topo_path
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"Clock Calibration (reference={cal_report.reference_camera}, "
                f"max_lag=±{cal_report.max_lag_seconds:.1f}s):"
            )
            for est in cal_report.estimates:
                print(
                    f"  - {est.camera_id}: offset={est.estimated_offset_seconds:+.2f}s "
                    f"(corr={est.correlation_score:.3f}, pairs={est.matched_pairs_count})"
                )
            if written_topo_path is not None:
                print(f"Calibrated topology written to: {written_topo_path}")
        return 0

    report = analyze_multi_camera_correlation(
        db_path,
        topology=topology,
        bucket_seconds=args.bucket_seconds,
        min_confidence=args.min_confidence,
        surge_min_cameras=args.surge_min_cameras,
        surge_min_tracks=args.surge_min_tracks,
    )

    persisted: dict[str, int] | None = None
    if args.persist:
        tr_saved, jr_saved = insert_cross_camera_correlation(db_path, report)
        persisted = {"transitions": tr_saved, "journeys": jr_saved}

    written: dict[str, str] = {}
    if args.output_dir is not None:
        artifacts = write_correlation_artifacts(report, args.output_dir)
        written = {key: str(path) for key, path in artifacts.items()}

    if args.as_json:
        payload = report.to_dict()
        if persisted is not None:
            payload["persisted"] = persisted
        if written:
            payload["artifacts"] = written
        print(json.dumps(payload, indent=2))
    else:
        print(generate_correlation_markdown(report))
        if persisted is not None:
            print(
                f"Persisted to {db_path}: {persisted['transitions']} transition(s), "
                f"{persisted['journeys']} journey(s)."
            )
        if written:
            print("Artifacts written:")
            for key, path_str in written.items():
                print(f"  - {key}: {path_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
