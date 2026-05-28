"""Generate local synthetic scenario datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learned_bucket_sort.scenarios import (
    DEFAULT_DATASET_DIR,
    SUPPORTED_SCENARIOS,
    generate_scenario_dataset_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local synthetic scenario datasets.")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("all", *SUPPORTED_SCENARIOS),
        help="Scenario dataset to generate, or 'all'.",
    )
    parser.add_argument("--n", type=_non_negative_int, default=10_000, help="Number of values.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=Path, default=DEFAULT_DATASET_DIR, help="Output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_scenario_dataset_files(
        scenario=args.scenario,
        n=args.n,
        seed=args.seed,
        out_dir=args.out,
    )

    for path in result.data_files:
        print(f"wrote dataset: {path}")
    print(f"wrote manifest: {result.manifest_path}")
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
