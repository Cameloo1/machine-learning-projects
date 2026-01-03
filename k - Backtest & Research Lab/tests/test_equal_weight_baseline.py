from __future__ import annotations

import pandas as pd

from backtest_lab.strategies.equal_weight import EqualWeightStrategy


def _make_prices() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"]
    )
    rows = []
    for symbol in ["AAA", "BBB"]:
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


def test_equal_weight_monthly_rebalance() -> None:
    prices = _make_prices()
    cfg = {"strategy": {"params": {"rebalance_frequency": "monthly"}}}
    strategy = EqualWeightStrategy(cfg)

    weights = strategy.predict_weights(prices, prices, cfg)

    by_ts = weights.groupby("ts")["weight"].sum()
    assert all(abs(val - 1.0) < 1e-9 for val in by_ts.values)

    jan_weights = weights.loc[weights["ts"] == pd.Timestamp("2020-01-30"), "weight"].values
    jan_next = weights.loc[weights["ts"] == pd.Timestamp("2020-01-31"), "weight"].values
    assert (jan_weights == jan_next).all()
