from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.data.validate import validate_prices


def _base_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "symbol": ["AAA", "AAA"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [100, 200],
        }
    )


def test_duplicate_keys_identical_values_are_dropped() -> None:
    df = _base_prices()
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True)

    out, diagnostics = validate_prices(df)

    assert len(out) == 2
    dup_diag = diagnostics["duplicate_diagnostics"]
    assert dup_diag["duplicate_count_before"] == 2
    assert dup_diag["duplicate_conflict_count"] == 0
    integrity = diagnostics["data_integrity_report"]
    assert integrity["row_count_before"] == 3
    assert integrity["row_count_after"] == 2


def test_duplicate_keys_conflicting_values_raise() -> None:
    df = _base_prices()
    dup = df.iloc[[0]].copy()
    dup.loc[dup.index[0], "close"] = 99.0
    df = pd.concat([df, dup], ignore_index=True)

    with pytest.raises(ValueError):
        validate_prices(df)
