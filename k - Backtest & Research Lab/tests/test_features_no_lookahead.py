from __future__ import annotations

import pandas as pd

from backtest_lab.signals.technical import compute_sma


def test_sma_uses_past_only_no_lookahead() -> None:
    prices = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "symbol": ["AAA", "AAA", "AAA"],
            "close": [1.0, 1.0, 1000.0],
        }
    )
    sma = compute_sma(prices, window=2, col="close")
    sma = sma.set_index("ts")
    # At 2020-01-02, SMA should be mean of [1.0, 1.0] = 1.0, not affected by 1000.0 at 2020-01-03.
    assert round(float(sma.loc[pd.Timestamp("2020-01-02"), "sma"]), 6) == 1.0
