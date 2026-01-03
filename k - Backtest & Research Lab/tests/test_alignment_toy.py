from __future__ import annotations

import pandas as pd

from backtest_lab.execution.accounting import run_backtest


def test_alignment_toy_dataset_off_by_one() -> None:
    prices = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 110.0, 121.0],
            "volume": [100, 100, 100],
        }
    )
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "symbol": ["AAA", "AAA"],
            "weight": [1.0, 1.0],
        }
    )

    returns_df, _, _ = run_backtest(
        prices,
        weights,
        {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
        diagnostics={},
    )

    returns_df = returns_df.set_index("ts")
    # 100 -> 110 = +10% realized on 2021-01-02 from decision 2021-01-01
    assert round(returns_df.loc[pd.Timestamp("2021-01-02"), "gross"], 6) == 0.10
    # 110 -> 121 = +10% realized on 2021-01-03 from decision 2021-01-02
    assert round(returns_df.loc[pd.Timestamp("2021-01-03"), "gross"], 6) == 0.10
