from __future__ import annotations

import pandas as pd

from backtest_lab.walkforward.engine import run_walkforward


def _make_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for ts in dates:
        rows.append(
            {
                "ts": ts,
                "symbol": "AAA",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000,
            }
        )
    for ts in dates:
        open_val = None if ts >= dates[3] else 200.0
        rows.append(
            {
                "ts": ts,
                "symbol": "BBB",
                "open": open_val,
                "high": 200.0,
                "low": 200.0,
                "close": 200.0,
                "volume": 1000,
            }
        )
    return pd.DataFrame(rows)


def _base_cfg() -> dict:
    return {
        "data": {"mode": "csv", "prices_path": "unused", "cache_dir": "data/raw"},
        "universe": {"symbols": ["AAA", "BBB"], "min_history_days": 3, "missing_data_policy": "drop_symbol"},
        "features": {"sma_fast": 2, "sma_slow": 3, "rsi_window": 2, "rsi_low": 30, "rsi_high": 70},
        "strategy": {"name": "equal_weight", "params": {"rebalance_frequency": "daily"}},
        "execution": {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
        "walkforward": {"enabled": True, "train_days": 3, "test_days": 2, "step_days": 2},
    }


def test_walkforward_universe_train_only_keeps_missing_test_symbol() -> None:
    prices = _make_prices()
    cfg = _base_cfg()
    cfg["universe_selection_mode"] = "train_only"

    _returns, _weights, _trades, diagnostics = run_walkforward(prices, cfg)
    universe = diagnostics["window_universe_diagnostics"][0]
    assert "BBB" in universe["final_assets"]


def test_walkforward_universe_window_full_drops_missing_test_symbol() -> None:
    prices = _make_prices()
    cfg = _base_cfg()
    cfg["universe_selection_mode"] = "window_full"

    _returns, _weights, _trades, diagnostics = run_walkforward(prices, cfg)
    universe = diagnostics["window_universe_diagnostics"][0]
    assert "BBB" not in universe["final_assets"]
