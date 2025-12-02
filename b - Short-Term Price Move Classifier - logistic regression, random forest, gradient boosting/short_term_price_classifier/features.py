"""Feature engineering utilities."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from ta.momentum import RSIIndicator

FEATURE_COLUMNS: List[str] = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "ma_5_ratio",
    "ma_10_ratio",
    "vol_10d",
    "vol_20d",
    "vol_rel_20d",
    "rsi_14",
]


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily returns and lagged returns."""

    df = df.copy()
    df["ret_1d"] = df["Close"].pct_change()
    for lag in range(1, 6):
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple moving averages and ratios."""

    df = df.copy()
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_5_ratio"] = df["ma_5"] / df["Close"]
    df["ma_10_ratio"] = df["ma_10"] / df["Close"]
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling volatility measures."""

    df = df.copy()
    df["vol_10d"] = df["ret_1d"].rolling(window=10).std()
    df["vol_20d"] = df["ret_1d"].rolling(window=20).std()
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative volume features."""

    df = df.copy()
    df["vol_rel_20d"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) feature."""

    df = df.copy()
    rsi_indicator = RSIIndicator(close=df["Close"], window=window)
    df[f"rsi_{window}"] = rsi_indicator.rsi()
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target based on next-day return."""

    df = df.copy()
    df["ret_next_1d"] = df["Close"].pct_change().shift(-1)
    df["target"] = (df["ret_next_1d"] > 0).astype(int)
    return df


def build_feature_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Construct the full feature matrix and target column."""

    feature_df = df.copy()
    feature_df = add_returns(feature_df)
    feature_df = add_moving_averages(feature_df)
    feature_df = add_volatility_features(feature_df)
    feature_df = add_volume_features(feature_df)
    feature_df = add_rsi(feature_df)
    feature_df = add_target(feature_df)

    final_df = feature_df[FEATURE_COLUMNS + ["target"]].dropna().copy()
    return final_df, FEATURE_COLUMNS.copy()

