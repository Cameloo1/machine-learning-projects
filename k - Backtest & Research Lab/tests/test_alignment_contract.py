from __future__ import annotations

import pandas as pd

from backtest_lab.execution.accounting import run_backtest


def test_alignment_contract_weights_apply_next_day() -> None:
    prices = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "open": [100.0, 110.0, 121.0],
            "high": [100.0, 110.0, 121.0],
            "low": [100.0, 110.0, 121.0],
            "close": [100.0, 110.0, 121.0],
            "volume": [100, 110, 120],
        }
    )

    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "symbol": ["AAA", "AAA"],
            "weight": [1.0, 0.5],
        }
    )

    returns_df, _, _ = run_backtest(
        prices,
        weights,
        {"cost_bps": 0.0, "slippage_model": "none", "max_leverage": 1.0, "max_weight_per_asset": 1.0},
        diagnostics={},
    )

    returns_df = returns_df.set_index("ts")
    assert round(returns_df.loc[pd.Timestamp("2020-01-02"), "gross"], 6) == 0.10
    assert round(returns_df.loc[pd.Timestamp("2020-01-03"), "gross"], 6) == 0.05
    assert pd.Timestamp("2020-01-01") not in returns_df.index
    assert returns_df.loc[pd.Timestamp("2020-01-02"), "decision_ts"] == pd.Timestamp("2020-01-01")


def test_alignment_forced_zero_on_missing_next_return() -> None:
    prices = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-03"]
            ),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "open": [100.0, 110.0, 200.0, 210.0],
            "high": [100.0, 110.0, 200.0, 210.0],
            "low": [100.0, 110.0, 200.0, 210.0],
            "close": [100.0, 110.0, 200.0, 210.0],
            "volume": [100, 110, 200, 210],
        }
    )

    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
        }
    )

    diagnostics = {}
    returns_df, _, _ = run_backtest(
        prices,
        weights,
        {"cost_bps": 0.0, "slippage_model": "none", "max_leverage": 1.0, "max_weight_per_asset": 1.0},
        diagnostics=diagnostics,
    )

    returns_df = returns_df.set_index("ts")
    assert round(returns_df.loc[pd.Timestamp("2020-01-02"), "gross"], 6) == 0.05
    assert diagnostics["alignment_forced_zero_count"] == 1
