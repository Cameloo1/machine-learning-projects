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
    assert round(returns_df.loc[pd.Timestamp("2020-01-01"), "gross"], 6) == 0.10
    assert round(returns_df.loc[pd.Timestamp("2020-01-02"), "gross"], 6) == 0.05
    assert round(returns_df.loc[pd.Timestamp("2020-01-03"), "gross"], 6) == 0.0
