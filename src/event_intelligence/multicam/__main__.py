"""CLI entrypoint for multi-camera spatial-temporal correlation: ``python -m event_intelligence.multicam``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from event_intelligence.multicam.engine import (
    analyze_multi_camera_correlation,
    load_camera_topology,
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
