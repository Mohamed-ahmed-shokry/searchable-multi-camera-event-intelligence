"""Phase 10 promotion: SQLite intelligence.db ingestion and CLI/programmatic search."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "cv_projects"
    / "10_searchable_outputs"
    / "scripts"
)
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from search_events import (
    SCHEMA_DDL,
    DatabaseSummary,
    EventSearchResult,
    EvidenceLookupResult,
    SearchConfig,
    TrackSearchResult,
    generate_schema_markdown,
    generate_search_examples_markdown,
    ingest_intelligence_db,
    init_intelligence_db,
    load_search_config,
    lookup_event_evidence,
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
    "generate_schema_markdown",
    "generate_search_examples_markdown",
    "ingest_intelligence_db",
    "init_intelligence_db",
    "load_search_config",
    "lookup_event_evidence",
    "query_events",
    "query_tracks",
    "run_searchable_outputs",
    "save_search_config",
]
