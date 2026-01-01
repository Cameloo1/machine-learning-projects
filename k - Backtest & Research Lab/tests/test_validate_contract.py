from __future__ import annotations

import json
import pandas as pd
import pytest

from backtest_lab.data.validate import validate_prices


REQUIRED_SET = {"ts", "symbol", "open", "high", "low", "close", "volume"}


def make_prices_df(*, ts_as_strings: bool = False, shuffle: bool = True) -> pd.DataFrame:
    rows = []
    for symbol in ["AAA", "BBB"]:
        for day in range(1, 7):
            ts = pd.Timestamp(f"2020-01-{day:02d}")
            ts_value = ts.strftime("%Y-%m-%d") if ts_as_strings else ts
            rows.append(
                {
                    "ts": ts_value,
                    "symbol": symbol,
                    "open": float(day),
                    "high": float(day) + 1.0,
                    "low": float(day) - 1.0,
                    "close": float(day) + 0.5,
                    "volume": 1000 + day,
                }
            )
    df = pd.DataFrame(rows)
    df = df.astype(
        {
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        }
    )
    df["symbol"] = df["symbol"].astype("string")
    if ts_as_strings:
        df["ts"] = df["ts"].astype("string")
    if shuffle:
        df = df.sample(frac=1, random_state=11).reset_index(drop=True)
    return df


def assert_sorted_by_symbol_ts(df: pd.DataFrame) -> None:
    sorted_df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    assert df.reset_index(drop=True)[["symbol", "ts"]].equals(sorted_df[["symbol", "ts"]])


def assert_json_serializable(obj: object) -> None:
    json.dumps(obj)


def test_validate_prices_contract_invariants() -> None:
    df = make_prices_df()
    out, diagnostics = validate_prices(df)

    assert set(out.columns) == REQUIRED_SET
    assert out["ts"].notna().all()
    assert out["symbol"].notna().all()
    assert out["symbol"].astype(str).str.strip().ne("").all()
    assert_sorted_by_symbol_ts(out)
    assert not out.duplicated(subset=["ts", "symbol"]).any()
    assert out["close"].notna().all()
    assert_json_serializable(diagnostics)

    for key in [
        "n_rows_in",
        "n_rows_out",
        "n_symbols_in",
        "n_symbols_out",
        "duplicate_policy",
        "n_dup_keys",
        "n_dup_rows_removed",
        "n_bad_ts",
        "n_empty_symbol",
        "n_null_close",
        "n_rows_dropped_null_close",
        "missingness_global",
        "missingness_by_symbol",
    ]:
        assert key in diagnostics


def test_validate_prices_raises_on_invalid_ts() -> None:
    df = make_prices_df(ts_as_strings=True, shuffle=False)
    df.loc[df.index[-1], "ts"] = "not-a-date"
    with pytest.raises(ValueError):
        validate_prices(df)


def test_validate_prices_raises_on_empty_symbol() -> None:
    df = make_prices_df()
    df.loc[0, "symbol"] = "   "
    with pytest.raises(ValueError):
        validate_prices(df)


def test_validate_prices_dedupes_duplicates() -> None:
    df = make_prices_df()
    duplicate_row = df.iloc[0].copy()
    df = pd.concat([df, duplicate_row.to_frame().T], ignore_index=True)
    out, diagnostics = validate_prices(df)

    assert not out.duplicated(subset=["ts", "symbol"]).any()
    assert diagnostics["n_dup_keys"] >= 1
    assert diagnostics["n_dup_rows_removed"] >= 1


def test_validate_prices_drops_null_close() -> None:
    df = make_prices_df()
    df.loc[0, "close"] = None

    out, diagnostics = validate_prices(df, drop_rows_with_null_close=True)

    assert diagnostics["n_null_close"] == 1
    assert diagnostics["n_rows_dropped_null_close"] == 1
    assert diagnostics["n_rows_out"] == diagnostics["n_rows_in"] - 1
    assert out["close"].notna().all()
