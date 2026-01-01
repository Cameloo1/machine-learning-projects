from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from backtest_lab.run import run_from_config_path


def _write_prices(path: Path) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        }
    )
    df.to_csv(path, index=False)


def test_e2e_pipeline_outputs_exist(tmp_path: Path) -> None:
    prices_path = tmp_path / "SPY.csv"
    _write_prices(prices_path)

    cfg = {
        "run_id": "e2e_test",
        "output_dir": "artifacts",
        "data": {"mode": "csv", "prices_path": "SPY.csv", "cache_dir": "data/raw"},
        "universe": {"symbols": ["SPY"], "min_history_days": 3, "missing_data_policy": "keep_gaps"},
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

    artifacts_dir = run_from_config_path(config_path)

    expected_files = [
        "config.json",
        "run_metadata.json",
        "diagnostics.json",
        "returns.csv",
        "weights.csv",
        "trades.csv",
        "metrics.csv",
        "metrics.json",
        "report.html",
    ]
    for name in expected_files:
        assert (artifacts_dir / name).exists()
        assert (artifacts_dir / name).stat().st_size > 0

    diagnostics = json.loads((artifacts_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert set(diagnostics.keys()) == {"validate", "universe", "alignment", "warmup"}
