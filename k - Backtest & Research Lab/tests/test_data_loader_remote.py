from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from backtest_lab.data import loader as loader_mod


def _make_yf_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [110, 111, 112],
            "Low": [90, 91, 92],
            "Close": [105, 106, 107],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )
    df.index.name = "Date"
    return df


def test_remote_loader_writes_cache_and_reads_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not hasattr(loader_mod, "load_prices_remote_yfinance"):
        pytest.skip("yfinance loader not implemented")

    df_map = {"SPY": _make_yf_df(), "QQQ": _make_yf_df()}
    call_counter = {"count": 0}

    def download(symbol: str, start: str, end: str, auto_adjust: bool, progress: bool) -> pd.DataFrame:
        call_counter["count"] += 1
        return df_map[symbol]

    yf_module = types.SimpleNamespace(download=download)
    monkeypatch.setitem(sys.modules, "yfinance", yf_module)

    out = loader_mod.load_prices_remote_yfinance(
        ["SPY", "QQQ"],
        start="2020-01-01",
        end="2020-01-10",
        cache_dir=tmp_path,
        refresh=True,
    )

    assert call_counter["count"] == 2
    assert (tmp_path / "SPY.csv").exists()
    assert (tmp_path / "QQQ.csv").exists()
    assert out.columns.tolist() == ["ts", "symbol", "open", "high", "low", "close", "volume"]
    assert out["symbol"].nunique() == 2

    def download_raise(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise AssertionError("yfinance.download called despite refresh=False cache hit")

    yf_module_cached = types.SimpleNamespace(download=download_raise)
    monkeypatch.setitem(sys.modules, "yfinance", yf_module_cached)

    out_cached = loader_mod.load_prices_remote_yfinance(
        ["SPY", "QQQ"],
        start="2020-01-01",
        end="2020-01-10",
        cache_dir=tmp_path,
        refresh=False,
    )

    assert out_cached["symbol"].nunique() == 2
