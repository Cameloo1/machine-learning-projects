from __future__ import annotations

import pandas as pd

from backtest_lab.signals.features import compute_features
from backtest_lab.strategies.sma_trend import SmaTrendStrategy


def test_sma_warmup_nan_policy() -> None:
    prices = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ).tolist()
            * 2,
            "symbol": ["AAA"] * 4 + ["BBB"] * 4,
            "open": [1, 2, 3, 4] * 2,
            "high": [1, 2, 3, 4] * 2,
            "low": [1, 2, 3, 4] * 2,
            "close": [1, 2, 3, 4] * 2,
            "volume": [100, 100, 100, 100] * 2,
        }
    )

    features = compute_features(
        prices, {"sma_fast": 2, "sma_slow": 3, "rsi_window": 2, "rsi_low": 30, "rsi_high": 70}
    )

    strategy = SmaTrendStrategy({})
    diagnostics = {}
    weights = strategy.predict_weights(
        prices,
        features,
        {"execution": {"max_leverage": 1.0, "max_weight_per_asset": 1.0}},
        diagnostics=diagnostics,
    )

    warmup_mask = features[["sma_fast", "sma_slow"]].isna().any(axis=1)
    merged = features.merge(weights, on=["ts", "symbol"], how="left")

    assert diagnostics["warmup_policy"] == "zero_weight"
    assert diagnostics["warmup_nan_rows"] == int(warmup_mask.sum())
    assert diagnostics["warmup_nan_rows_by_symbol"]["AAA"] == 2
    assert diagnostics["warmup_nan_rows_by_symbol"]["BBB"] == 2
    assert (merged.loc[warmup_mask, "weight"] == 0.0).all()
