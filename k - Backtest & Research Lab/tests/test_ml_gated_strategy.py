from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.signals.ml_ingest import load_predictions
from backtest_lab.strategies.ml_gated import MLGatedStrategy


def test_ml_gated_missing_preds_zeroed(tmp_path) -> None:
    ts = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame(
        {
            "ts": ts,
            "symbol": ["AAA"] * 3,
            "open": [100, 101, 102],
            "high": [100, 101, 102],
            "low": [100, 101, 102],
            "close": [100, 101, 102],
            "volume": [1000, 1100, 1200],
        }
    )

    preds_path = tmp_path / "preds.csv"
    pd.DataFrame(
        {
            "ts": [ts[0], ts[2]],
            "symbol": ["AAA", "AAA"],
            "pred": [0.6, 0.6],
        }
    ).to_csv(preds_path, index=False)

    cfg = {
        "strategy": {
            "name": "ml_gated",
            "params": {
                "preds_path": str(preds_path),
                "threshold": 0.55,
                "base_strategy": {"name": "equal_weight", "params": {"rebalance_frequency": "daily"}},
            },
        }
    }

    strategy = MLGatedStrategy(cfg)
    weights = strategy.predict_weights(prices, prices, cfg)
    by_ts = weights.sort_values("ts").set_index("ts")["weight"]
    assert by_ts.loc[ts[0]] == 1.0
    assert by_ts.loc[ts[1]] == 0.0
    assert by_ts.loc[ts[2]] == 1.0


def test_load_predictions_rejects_duplicates(tmp_path) -> None:
    preds_path = tmp_path / "preds_dup.csv"
    pd.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-01"],
            "symbol": ["AAA", "AAA"],
            "pred": [0.5, 0.6],
        }
    ).to_csv(preds_path, index=False)

    with pytest.raises(ValueError):
        load_predictions(preds_path)
