from __future__ import annotations

import pandas as pd

from backtest_lab.walkforward.engine import run_walkforward


def _make_prices(symbols: list[str], n_days: int) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    for symbol in symbols:
        for idx, ts in enumerate(dates):
            px = 100 + idx
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1000,
                }
            )
    return pd.DataFrame(rows)


def test_walkforward_validation_uses_only_val_window() -> None:
    prices = _make_prices(["AAA"], n_days=25)
    cfg = {
        "data": {"mode": "csv", "prices_path": "unused", "cache_dir": "data/raw"},
        "universe": {"symbols": ["AAA"], "min_history_days": 3, "missing_data_policy": "keep_gaps"},
        "features": {"sma_fast": 2, "sma_slow": 4, "rsi_window": 2, "rsi_low": 30, "rsi_high": 70},
        "strategy": {"name": "sma_trend", "params": {}},
        "execution": {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
        "walkforward": {"enabled": True, "train_days": 8, "val_days": 4, "test_days": 4, "step_days": 4},
    }

    _returns, _weights, _trades, diagnostics = run_walkforward(prices, cfg)
    for window_diag in diagnostics["windows"]:
        window = window_diag["window"]
        validation = window_diag["validation"]
        if not validation.get("enabled"):
            continue
        val_start = pd.Timestamp(window["val_start"])
        val_end = pd.Timestamp(window["val_end"])
        for candidate in validation.get("candidates", []):
            if candidate["decision_start"] is None:
                continue
            decision_start = pd.Timestamp(candidate["decision_start"])
            decision_end = pd.Timestamp(candidate["decision_end"])
            assert decision_start >= val_start
            assert decision_end <= val_end
