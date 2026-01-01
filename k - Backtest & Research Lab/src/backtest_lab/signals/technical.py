from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sma(prices: pd.DataFrame, window: int, *, col: str = "close") -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    df = prices[["ts", "symbol", col]].copy()
    df = df.sort_values(["symbol", "ts"], kind="mergesort")
    df["sma"] = df.groupby("symbol", sort=False)[col].transform(
        lambda s: s.rolling(window=window, min_periods=window).mean()
    )
    return df[["ts", "symbol", "sma"]]


def compute_rsi(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    df = prices[["ts", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "ts"], kind="mergesort")
    delta = df.groupby("symbol", sort=False)["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(window=window, min_periods=window).mean()
    )
    avg_loss = loss.groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(window=window, min_periods=window).mean()
    )

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return pd.DataFrame({"ts": df["ts"], "symbol": df["symbol"], "rsi": rsi})


def compute_rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    df = returns[["ts", "symbol", "ret"]].copy()
    df = df.sort_values(["symbol", "ts"], kind="mergesort")
    df["rolling_vol"] = df.groupby("symbol", sort=False)["ret"].transform(
        lambda s: s.rolling(window=window, min_periods=window).std(ddof=0)
    )
    return df[["ts", "symbol", "rolling_vol"]]
