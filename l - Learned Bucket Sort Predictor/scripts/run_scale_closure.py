"""Run the Part 3.8 scale-closure benchmark from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learned_bucket_sort.scale_closure import main


if __name__ == "__main__":
    raise SystemExit(main())
