"""Package CLI entrypoint: python -m event_intelligence."""

from __future__ import annotations

from event_intelligence.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
