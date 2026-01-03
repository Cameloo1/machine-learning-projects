from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.execution.validate_outputs import validate_returns_df, validate_trades_df


def test_validate_returns_missing_column_raises() -> None:
    returns_df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-02"]),
            "decision_ts": pd.to_datetime(["2020-01-01"]),
            "gross": [0.01],
            "net": [0.01],
            "exposure": [1.0],
            "turnover": [0.0],
            "txn_cost": [0.0],
            "slippage_cost": [0.0],
            # "costs" intentionally missing
        }
    )
    with pytest.raises(ValueError) as exc:
        validate_returns_df(returns_df)
    assert "missing columns" in str(exc.value)


def test_validate_trades_non_numeric_raises() -> None:
    trades_df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01"]),
            "symbol": ["AAA"],
            "weight": [0.1],
            "dw": [0.1],
            "abs_dw": [0.1],
            "txn_cost": ["bad"],
            "slippage_cost": [0.0],
            "cost": [0.0],
        }
    )
    with pytest.raises(ValueError) as exc:
        validate_trades_df(trades_df)
    assert "non-numeric" in str(exc.value)
