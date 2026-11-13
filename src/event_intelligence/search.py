"""Search CLI entrypoint: ``python -m event_intelligence.search``."""

from __future__ import annotations

from event_intelligence.storage.search_events import (
    SCHEMA_DDL,
    CameraActivitySummary,
    CameraMetadata,
    DatabaseSummary,
    EventSearchResult,
    EvidenceLookupResult,
    InsightSearchResult,
    SearchConfig,
    TrackSearchResult,
    build_parser,
    generate_schema_markdown,
    generate_search_examples_markdown,
    ingest_intelligence_db,
    ingest_multi_camera_intelligence_db,
    init_intelligence_db,
    insert_insights,
    load_search_config,
    lookup_event_evidence,
    main,
    query_camera_summary,
    query_events,
    query_insights,
    query_tracks,
    run_searchable_outputs,
    save_search_config,
)

__all__ = [
    "SCHEMA_DDL",
    "CameraActivitySummary",
    "CameraMetadata",
    "DatabaseSummary",
    "EventSearchResult",
    "EvidenceLookupResult",
    "InsightSearchResult",
    "SearchConfig",
    "TrackSearchResult",
    "build_parser",
    "generate_schema_markdown",
    "generate_search_examples_markdown",
    "ingest_intelligence_db",
    "ingest_multi_camera_intelligence_db",
    "init_intelligence_db",
    "insert_insights",
    "load_search_config",
    "lookup_event_evidence",
    "main",
    "query_camera_summary",
    "query_events",
    "query_insights",
    "query_tracks",
    "run_searchable_outputs",
    "save_search_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
