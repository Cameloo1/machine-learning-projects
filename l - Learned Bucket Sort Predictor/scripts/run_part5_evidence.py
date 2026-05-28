"""Run the Part 5 evidence workflow from a checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learned_bucket_sort.part5_evidence import run_part5_evidence
from learned_bucket_sort.scenarios import DEFAULT_DATASET_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate Part 5 evidence artifacts and promoted plots.")
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts") / "evidence")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = run_part5_evidence(
        evidence_root=args.evidence_root,
        assets_dir=args.assets_dir,
        dataset_dir=args.dataset_dir,
    )

    print(f"evidence dir: {run.evidence_dir}")
    print(f"scale artifact: {run.scale_artifact}")
    print(f"benchmark artifact: {run.benchmark_artifact}")
    print(f"amortized artifact: {run.amortized_artifact}")
    for name, path in run.plot_paths.promoted.items():
        print(f"promoted plot: {name} -> {path}")
    print(f"manifest: {run.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
