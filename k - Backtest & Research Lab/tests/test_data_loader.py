from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

from backtest_lab.data.loader import load_prices_from_csv, load_prices_from_csv_dir


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_base_df(date_col: str = "Date") -> pd.DataFrame:
    return pd.DataFrame(
        {
            date_col: ["2020-01-01", "2020-01-02", "2020-01-03"],
            "Open": [100, 101, 102],
            "High": [110, 111, 112],
            "Low": [90, 91, 92],
            "Close": [105, 106, 107],
            "Volume": [1000, 1100, 1200],
        }
    )


def test_single_asset_csv_schema_and_types(tmp_path: Path) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    _write_csv(csv_path, df)

    out = load_prices_from_csv(csv_path, symbol="SPY")

    assert out.columns.tolist() == ["ts", "symbol", "open", "high", "low", "close", "volume"]
    assert is_datetime64_any_dtype(out["ts"])
    assert is_string_dtype(out["symbol"])
    assert out["symbol"].nunique() == 1
    assert out["symbol"].unique().tolist() == ["SPY"]
    for col in ["open", "high", "low", "close", "volume"]:
        assert is_numeric_dtype(out[col])
    assert out["ts"].is_monotonic_increasing
    assert not out.duplicated(subset=["ts", "symbol"]).any()


@pytest.mark.parametrize("date_col", ["Date", "date", "Datetime", "timestamp", "ts"])
def test_date_column_name_variations(tmp_path: Path, date_col: str) -> None:
    csv_path = tmp_path / f"{date_col}.csv"
    df = _make_base_df(date_col=date_col)
    _write_csv(csv_path, df)

    out = load_prices_from_csv(csv_path, symbol="SPY")
    assert "ts" in out.columns
    assert is_datetime64_any_dtype(out["ts"])


def test_missing_date_column_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    df = df.rename(columns={"Date": "TradeDate"})
    _write_csv(csv_path, df)

    with pytest.raises(ValueError) as exc_info:
        load_prices_from_csv(csv_path, symbol="SPY")
    message = str(exc_info.value)
    assert "Missing date column" in message
    assert "Expected one of" in message
    assert "date" in message


def test_missing_ohlcv_columns_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    df = df.drop(columns=["Volume"])
    _write_csv(csv_path, df)

    with pytest.raises(ValueError) as exc_info:
        load_prices_from_csv(csv_path, symbol="SPY")
    message = str(exc_info.value)
    assert "Missing required columns" in message
    assert "volume" in message
    assert "Columns found" in message


def test_unparseable_timestamp_raises_clear_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    df.loc[1, "Date"] = "not-a-date"
    _write_csv(csv_path, df)

    with pytest.raises(ValueError) as exc_info:
        load_prices_from_csv(csv_path, symbol="SPY")
    message = str(exc_info.value)
    assert "parse timestamps" in message


def test_duplicates_dropped_deterministically(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    _write_csv(csv_path, df)

    caplog.set_level(logging.INFO)
    out = load_prices_from_csv(csv_path, symbol="SPY")

    assert len(out) == len(df) - 1
    assert not out.duplicated(subset=["ts", "symbol"]).any()
    assert any("Dropped duplicate rows" in record.message for record in caplog.records)


def test_missingness_logged_not_imputed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    df.loc[1, "Close"] = None
    _write_csv(csv_path, df)

    caplog.set_level(logging.INFO)
    out = load_prices_from_csv(csv_path, symbol="SPY")

    assert out["close"].isna().any()
    assert any("Missing values for SPY" in record.message for record in caplog.records)


def test_multi_asset_dir_loads_and_combines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    symbols = [f"SYM{i:02d}" for i in range(10)]
    for symbol in symbols:
        df = pd.DataFrame(
            {
                "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "Open": [1.0, 1.1, 1.2],
                "High": [1.2, 1.3, 1.4],
                "Low": [0.8, 0.9, 1.0],
                "Close": [1.1, 1.2, 1.3],
                "Volume": [100, 110, 120],
            }
        )
        _write_csv(tmp_path / f"{symbol}.csv", df)

    caplog.set_level(logging.INFO)
    out = load_prices_from_csv_dir(tmp_path)

    assert out.columns.tolist() == ["ts", "symbol", "open", "high", "low", "close", "volume"]
    assert out["symbol"].nunique() >= 10
    assert out["ts"].is_monotonic_increasing
    assert not out.duplicated(subset=["ts", "symbol"]).any()

    for symbol in symbols:
        assert any(f"symbol={symbol}" in record.message for record in caplog.records)


def test_symbol_inferred_from_filename_when_not_provided(tmp_path: Path) -> None:
    csv_path = tmp_path / "SPY.csv"
    df = _make_base_df()
    _write_csv(csv_path, df)

    out = load_prices_from_csv(csv_path)
    assert out["symbol"].nunique() == 1
    assert out["symbol"].unique().tolist() == ["SPY"]
