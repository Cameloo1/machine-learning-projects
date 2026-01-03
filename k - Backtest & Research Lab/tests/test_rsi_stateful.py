from __future__ import annotations

import pandas as pd

from backtest_lab.strategies.rsi_mr import RsiMeanReversionStrategy


def test_rsi_stateful_enter_exit() -> None:
    ts = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    )
    prices = pd.DataFrame({"ts": ts, "symbol": ["AAA"] * len(ts)})
    features = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["AAA"] * len(ts),
            "rsi": [None, 25.0, 40.0, 80.0, 50.0],
        }
    )

    strategy = RsiMeanReversionStrategy({})
    diagnostics = {}
    weights = strategy.predict_weights(
        prices,
        features,
        {"features": {"rsi_low": 30, "rsi_high": 70}},
        diagnostics=diagnostics,
    )

    got = weights.sort_values("ts")["weight"].tolist()
    assert got == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert diagnostics["rsi_policy"] == "stateful_enter_exit"
