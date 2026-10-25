"""Pytest bootstrap: expose experiment script modules to test collection.

Previously ``import helpers`` worked only because unittest discover and pytest's
alphabetical collection imported ``helpers`` (which inserts the Phase 1 scripts
directory into ``sys.path``) before modules that import ``inspect_video`` et al.
Importing helpers here makes every test module order-independent.
"""

from helpers import (  # noqa: F401
    DETECTION_SCRIPTS,
    INSPECTION_SCRIPTS,
    PROJECT_ROOT,
    TRACKING_SCRIPTS,
    TRAJECTORY_SCRIPTS,
)
