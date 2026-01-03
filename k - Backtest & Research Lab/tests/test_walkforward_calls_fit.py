from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest_lab.walkforward.engine import run_walkforward


class _CountingStrategy:
    def __init__(self, counter):
        self._counter = counter

    def fit(self, prices, features, returns, cfg, *, diagnostics=None):
        self._counter[0] += 1
        if diagnostics is not None:
            diagnostics["called"] = True

    def predict_weights(self, prices, features, cfg, *, diagnostics=None):
        df = prices[["ts", "symbol"]].copy()
        counts = df.groupby("ts", sort=False)["symbol"].transform("count")
        df["weight"] = 1.0 / counts
        return df


def _write_prices(path: Path, symbols: list[str], n_days: int) -> pd.DataFrame:
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
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def test_walkforward_calls_fit_once_per_window(tmp_path, monkeypatch) -> None:
    prices_path = tmp_path / "prices.csv"
    prices = _write_prices(prices_path, ["AAA", "BBB"], n_days=20)

    counter = [0]

    def _create_stub(_cfg):
        return _CountingStrategy(counter)

    monkeypatch.setattr("backtest_lab.strategies.factory.create", _create_stub)

    cfg = {
        "data": {"mode": "csv", "prices_path": str(prices_path), "cache_dir": str(tmp_path)},
        "universe": {"symbols": ["AAA", "BBB"], "min_history_days": 3, "missing_data_policy": "keep_gaps"},
        "features": {"sma_fast": 2, "sma_slow": 3, "rsi_window": 2, "rsi_low": 30, "rsi_high": 70},
        "strategy": {"name": "sma_trend", "params": {}},
        "execution": {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
        "walkforward": {"enabled": True, "train_days": 5, "test_days": 3, "step_days": 3},
    }

    _returns, _weights, _trades, diagnostics = run_walkforward(prices, cfg)
    assert counter[0] == diagnostics["window_count"]
