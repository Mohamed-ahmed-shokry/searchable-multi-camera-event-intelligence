"""Phase 10 promotion: SQLite intelligence.db ingestion and CLI/programmatic search."""

from __future__ import annotations

from event_intelligence.storage.search_events import (
    SCHEMA_DDL,
    DatabaseSummary,
    EventSearchResult,
    EvidenceLookupResult,
    SearchConfig,
    TrackSearchResult,
    build_parser,
    generate_schema_markdown,
    generate_search_examples_markdown,
    get_database_summary,
    ingest_intelligence_db,
    init_intelligence_db,
    load_search_config,
    lookup_event_evidence,
    main,
    query_events,
    query_tracks,
    run_searchable_outputs,
    save_search_config,
)

__all__ = [
    "SCHEMA_DDL",
    "DatabaseSummary",
    "EventSearchResult",
    "EvidenceLookupResult",
    "SearchConfig",
    "TrackSearchResult",
    "build_parser",
    "generate_schema_markdown",
    "generate_search_examples_markdown",
    "get_database_summary",
    "ingest_intelligence_db",
    "init_intelligence_db",
    "load_search_config",
    "lookup_event_evidence",
    "main",
    "query_events",
    "query_tracks",
    "run_searchable_outputs",
    "save_search_config",
]
