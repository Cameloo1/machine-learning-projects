from __future__ import annotations

import argparse
from pathlib import Path

from backtest_lab.sensitivity import run_sensitivity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost/slippage sensitivity for a config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output-dir", help="Override output directory for artifacts.")
    parser.add_argument("--run-id", help="Override base run id for sensitivity runs.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_sensitivity(
        Path(args.config),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
