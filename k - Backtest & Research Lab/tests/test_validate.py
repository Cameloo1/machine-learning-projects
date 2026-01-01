from __future__ import annotations

import pandas as pd
import pytest

from backtest_lab.data.validate import validate_prices


def _base_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "symbol": ["AAA", "AAA", "AAA"],
            "open": [100.0, 101.0, 102.0],
            "high": [110.0, 111.0, 112.0],
            "low": [90.0, 91.0, 92.0],
            "close": [105.0, 106.0, 107.0],
            "volume": [1000, 1100, 1200],
        }
    )


def test_missing_required_columns_raises() -> None:
    df = _base_prices().drop(columns=["close"])
    with pytest.raises(ValueError) as exc_info:
        validate_prices(df)
    message = str(exc_info.value)
    assert "close" in message
    assert "Missing required columns" in message


def test_bad_timestamps_raise() -> None:
    df = _base_prices()
    df.loc[1, "ts"] = "bad-date"
    with pytest.raises(ValueError) as exc_info:
        validate_prices(df)
    message = str(exc_info.value)
    assert "invalid ts" in message
    assert "1" in message


def test_duplicates_removed_deterministically_keep_first_after_sort() -> None:
    df = pd.DataFrame(
        {
            "ts": ["2020-01-02", "2020-01-02", "2020-01-01", "2020-01-01"],
            "symbol": ["AAA", "AAA", "AAA", "AAA"],
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [10.0, 20.0, 30.0, 40.0],
            "volume": [100, 200, 300, 400],
        }
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    out, diagnostics = validate_prices(df)

    expected = (
        df.assign(ts=pd.to_datetime(df["ts"]))
        .sort_values(["symbol", "ts"], kind="mergesort")
        .drop_duplicates(subset=["ts", "symbol"], keep="first")
        .reset_index(drop=True)
    )

    assert len(out) == len(expected)
    assert not out.duplicated(subset=["ts", "symbol"]).any()
    assert out.loc[out["ts"] == pd.Timestamp("2020-01-01"), "close"].iloc[0] == expected.loc[
        expected["ts"] == pd.Timestamp("2020-01-01"), "close"
    ].iloc[0]
    assert diagnostics["n_dup_rows_removed"] == 2
    assert diagnostics["n_dup_keys"] == 2
    assert any("2020-01-01" in sample for sample in diagnostics["dup_key_samples"])


def test_output_unique_and_sorted() -> None:
    df = pd.DataFrame(
        {
            "ts": ["2020-01-02", "2020-01-01", "2020-01-03", "2020-01-01"],
            "symbol": ["BBB", "AAA", "AAA", "BBB"],
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [10.0, 20.0, 30.0, 40.0],
            "volume": [100, 200, 300, 400],
        }
    )
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    out, _ = validate_prices(df)
    sorted_out = out.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

    assert out.reset_index(drop=True)[["symbol", "ts"]].equals(
        sorted_out[["symbol", "ts"]]
    )
    assert not out.duplicated(subset=["ts", "symbol"]).any()


def test_null_close_dropped_when_enabled() -> None:
    df = _base_prices()
    df.loc[1, "close"] = None

    out, diagnostics = validate_prices(df, drop_rows_with_null_close=True)

    assert len(out) == len(df) - 1
    assert diagnostics["n_null_close"] == 1
    assert diagnostics["n_rows_dropped_null_close"] == 1
    assert not out["close"].isna().any()
