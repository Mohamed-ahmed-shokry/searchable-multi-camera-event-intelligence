"""Package CLI entrypoint: python -m event_intelligence.evaluation."""

from __future__ import annotations

from event_intelligence.evaluation.evaluate_pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
