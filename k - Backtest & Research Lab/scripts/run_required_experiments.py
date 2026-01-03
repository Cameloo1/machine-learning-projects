from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtest_lab.run import run_from_config_path
from backtest_lab.sensitivity import run_sensitivity


CONFIGS = [
    "configs/sma_spy.yaml",
    "configs/rsi_spy.yaml",
    "configs/ew_baseline_multi.yaml",
    "configs/vol_target_multi.yaml",
    "configs/ml_gated_spy.yaml",
    "configs/walkforward_sma_multi.yaml",
    "configs/walkforward_rsi_spy.yaml",
    "configs/walkforward_ew_multi.yaml",
]


def main() -> int:
    summary_rows = []
    for cfg_path in CONFIGS:
        artifacts_dir = run_from_config_path(Path(cfg_path))
        metrics_path = artifacts_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        summary_rows.append(
            {
                "config": str(cfg_path),
                "run_id": artifacts_dir.name,
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "turnover_avg": metrics.get("turnover_avg"),
                "costs_avg": metrics.get("costs_avg"),
            }
        )

    run_sensitivity(Path("configs/vol_target_multi.yaml"), run_id="vol_target_multi_sensitivity")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_path = Path("artifacts") / "required_experiments_summary.csv"
        summary_df.to_csv(out_path, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
