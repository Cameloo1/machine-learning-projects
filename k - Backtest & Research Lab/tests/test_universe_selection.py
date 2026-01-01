from __future__ import annotations

import pandas as pd

from backtest_lab.data.universe import select_universe


def _make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"] * 2
            ),
            "symbol": ["AAA"] * 3 + ["BBB"] * 3,
            "open": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [1, 2, 3, 4, 5, 6],
            "close": [1, 2, 3, 4, 5, 6],
            "volume": [10, 20, 30, 40, 50, 60],
        }
    )


def test_universe_deterministic_hash() -> None:
    prices = _make_prices().sample(frac=1, random_state=7).reset_index(drop=True)

    start_ts = pd.Timestamp("2020-01-01")
    end_ts = pd.Timestamp("2020-01-03")

    _, diag_first = select_universe(prices, start_ts, end_ts, min_history=2, policy="keep_gaps")
    _, diag_second = select_universe(prices, start_ts, end_ts, min_history=2, policy="keep_gaps")

    assert diag_first["final_assets"] == diag_second["final_assets"]
    assert diag_first["asset_hash"] == diag_second["asset_hash"]
