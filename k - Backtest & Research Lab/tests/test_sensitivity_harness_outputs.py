from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from backtest_lab.sensitivity import run_sensitivity


def _write_prices(path: Path) -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=6, freq="D"),
            "symbol": ["AAA"] * 6,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500],
        }
    )
    df.to_csv(path, index=False)


def test_sensitivity_harness_outputs(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    _write_prices(prices_path)

    cfg = {
        "run_id": "sens",
        "output_dir": str(tmp_path / "artifacts"),
        "data": {"mode": "csv", "prices_path": str(prices_path), "cache_dir": str(tmp_path)},
        "universe": {"symbols": ["AAA"], "min_history_days": 3, "missing_data_policy": "keep_gaps"},
        "features": {"sma_fast": 2, "sma_slow": 3, "rsi_window": 2, "rsi_low": 30, "rsi_high": 70},
        "strategy": {"name": "sma_trend", "params": {}},
        "execution": {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = run_sensitivity(config_path)

    summary_path = result["summary_path"]
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    assert len(summary) == 3
    for suffix in ["low", "mid", "high"]:
        assert (Path(cfg["output_dir"]) / f"sens_{suffix}").exists()
