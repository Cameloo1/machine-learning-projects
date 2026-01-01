from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from backtest_lab.signals import technical


def compute_features(prices: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    sma_fast = int(cfg["sma_fast"])
    sma_slow = int(cfg["sma_slow"])
    rsi_window = int(cfg["rsi_window"])
    if sma_fast <= 0 or sma_slow <= 0 or rsi_window <= 0:
        raise ValueError("Feature windows must be positive integers")

    base = prices[["ts", "symbol", "close"]].copy()
    base = base.sort_values(["symbol", "ts"], kind="mergesort")

    sma_fast_df = technical.compute_sma(base, window=sma_fast, col="close").rename(
        columns={"sma": "sma_fast"}
    )
    sma_slow_df = technical.compute_sma(base, window=sma_slow, col="close").rename(
        columns={"sma": "sma_slow"}
    )
    rsi_df = technical.compute_rsi(base, window=rsi_window)

    features = base.merge(sma_fast_df, on=["ts", "symbol"], how="left")
    features = features.merge(sma_slow_df, on=["ts", "symbol"], how="left")
    features = features.merge(rsi_df, on=["ts", "symbol"], how="left")

    return features
