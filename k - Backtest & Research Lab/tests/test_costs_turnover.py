from __future__ import annotations

import pandas as pd

from backtest_lab.execution.accounting import run_backtest


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2),
            "symbol": ["AAA"] * 3 + ["BBB"] * 3,
            "open": [1, 1, 1, 1, 1, 1],
            "high": [1, 1, 1, 1, 1, 1],
            "low": [1, 1, 1, 1, 1, 1],
            "close": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "volume": [100, 100, 100, 100, 100, 100],
        }
    )


def test_turnover_costs_applied_next_day() -> None:
    prices = _prices()
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.5, 0.5, 1.0, 0.0],
        }
    )

    returns_df, _, _ = run_backtest(
        prices,
        weights,
        {
            "cost_bps": 10.0,
            "slippage_model": "none",
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
        diagnostics={},
    )

    returns_df = returns_df.set_index("ts")
    # Turnover on 2020-01-02 should apply to realized return on 2020-01-03.
    assert round(returns_df.loc[pd.Timestamp("2020-01-03"), "costs"], 6) == 0.001
