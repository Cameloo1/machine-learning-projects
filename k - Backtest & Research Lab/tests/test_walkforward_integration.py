from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from backtest_lab.run import run_from_config_path


def _hash_assets(assets: list[str]) -> str:
    payload = "\n".join(assets).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_prices(path: Path, symbols: list[str], n_days: int) -> None:
    rows = []
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    for symbol in symbols:
        for idx, ts in enumerate(dates):
            px = 100 + idx
            rows.append(
                {
                    "Date": ts,
                    "Symbol": symbol,
                    "Open": px,
                    "High": px,
                    "Low": px,
                    "Close": px,
                    "Volume": 1000,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_walkforward_no_leakage_and_universe_lock(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    symbols = ["AAA", "BBB", "CCC"]
    _write_prices(prices_path, symbols, n_days=40)

    cfg = {
        "run_id": "wf_test",
        "output_dir": "artifacts",
        "data": {"mode": "csv", "prices_path": "prices.csv", "cache_dir": "data/raw"},
        "universe": {"symbols": symbols, "min_history_days": 5, "missing_data_policy": "keep_gaps"},
        "features": {"sma_fast": 2, "sma_slow": 3, "rsi_window": 2, "rsi_low": 30, "rsi_high": 70},
        "strategy": {"name": "sma_trend", "params": {}},
        "execution": {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
        "walkforward": {"enabled": True, "train_days": 10, "test_days": 5, "step_days": 5},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    artifacts_dir = run_from_config_path(config_path)
    diagnostics = json.loads((artifacts_dir / "diagnostics.json").read_text(encoding="utf-8"))
    wf = diagnostics["walkforward"]

    assert wf["window_count"] >= 1
    assert wf["window_universe_diagnostics"]
    for window in wf["windows"]:
        win = window["window"]
        universe = window["universe"]
        assert win["train_end"] < win["test_start"]
        assert universe["final_assets"]
        assert universe["asset_hash"] == _hash_assets(sorted(universe["final_assets"]))
    for universe in wf["window_universe_diagnostics"]:
        assert universe["asset_hash"] == _hash_assets(sorted(universe["final_assets"]))
