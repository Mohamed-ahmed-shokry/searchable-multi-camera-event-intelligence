"""Pytest bootstrap: expose experiment script modules to test collection.

Previously ``import helpers`` worked only because unittest discover and pytest's
alphabetical collection imported ``helpers`` (which inserts the Phase 1 scripts
directory into ``sys.path``) before modules that import ``inspect_video`` et al.
Importing helpers here makes every test module order-independent.
"""

from helpers import (  # noqa: F401
    ANALYTICS_SCRIPTS,
    DETECTION_SCRIPTS,
    EVALUATION_SCRIPTS,
    EVENTS_SCRIPTS,
    EVIDENCE_SCRIPTS,
    INSPECTION_SCRIPTS,
    PROJECT_ROOT,
    SEARCH_SCRIPTS,
    SPATIAL_SCRIPTS,
    TRACKING_SCRIPTS,
    TRAJECTORY_SCRIPTS,
)
