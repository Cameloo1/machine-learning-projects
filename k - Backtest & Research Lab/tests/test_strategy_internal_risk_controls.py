from __future__ import annotations

import pandas as pd

from backtest_lab.strategies.sma_trend import SmaTrendStrategy


def _make_prices_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(["2020-01-01", "2020-01-02"])
    prices = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["AAA", "AAA"],
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1100],
        }
    )
    features = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["AAA", "AAA"],
            "sma_fast": [10.0, 10.0],
            "sma_slow": [5.0, 5.0],
        }
    )
    return prices, features


def test_strategy_internal_controls_off_by_default() -> None:
    prices, features = _make_prices_features()
    cfg = {
        "execution": {"max_leverage": 0.2, "max_weight_per_asset": 0.2},
        "strategy_internal_risk_controls": False,
    }
    strategy = SmaTrendStrategy(cfg)
    weights = strategy.predict_weights(prices, features, cfg)
    assert weights["weight"].max() > 0.2


def test_strategy_internal_controls_apply_caps() -> None:
    prices, features = _make_prices_features()
    cfg = {
        "execution": {"max_leverage": 0.2, "max_weight_per_asset": 0.2},
        "strategy_internal_risk_controls": True,
    }
    strategy = SmaTrendStrategy(cfg)
    weights = strategy.predict_weights(prices, features, cfg)
    assert round(weights["weight"].max(), 6) == 0.2
