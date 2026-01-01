from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.execution.accounting import run_backtest


def _make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "open": [1.0, 1.1, 2.0, 2.1],
            "high": [1.0, 1.1, 2.0, 2.1],
            "low": [1.0, 1.1, 2.0, 2.1],
            "close": [1.0, 1.1, 2.0, 2.1],
            "volume": [100, 110, 200, 210],
        }
    )


def test_accounting_weight_fill_logging_and_strict_mode() -> None:
    prices = _make_prices()
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
        }
    )

    diagnostics = {}
    run_backtest(
        prices,
        weights,
        {
            "cost_bps": 0.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
            "strict_weight_alignment": False,
        },
        diagnostics=diagnostics,
    )

    assert diagnostics["n_missing_weight_keys"] == 2
    assert diagnostics["n_filled_weight_rows"] == 2

    weights_extra = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-03"]),
            "symbol": ["AAA"],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError):
        run_backtest(
            prices,
            weights_extra,
            {
                "cost_bps": 0.0,
                "slippage_model": "none",
                "slippage_params": {},
                "max_leverage": 1.0,
                "max_weight_per_asset": 1.0,
                "strict_weight_alignment": True,
            },
            diagnostics={},
        )
