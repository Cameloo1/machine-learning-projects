from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


def validate_weights_df(
    weights: pd.DataFrame,
    *,
    prices: pd.DataFrame | None = None,
) -> None:
    """
    Validate strategy output weights contract (ts, symbol, weight) with basic invariants.
    """
    required = {"ts", "symbol", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"Missing columns in weights: {sorted(missing)}")
    if weights["ts"].isna().any():
        raise ValueError("Weights contain null ts values")
    if weights["symbol"].isna().any() or weights["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("Weights contain empty symbols")
    if weights["weight"].isna().any():
        raise ValueError("Weights contain null weight values")
    if weights.duplicated(subset=["ts", "symbol"]).any():
        dup_count = int(weights.duplicated(subset=["ts", "symbol"]).sum())
        raise ValueError(f"Duplicate weights rows detected: {dup_count}")
    if prices is not None:
        weight_symbols = set(weights["symbol"].astype(str).unique())
        price_symbols = set(prices["symbol"].astype(str).unique())
        if not weight_symbols.issubset(price_symbols):
            raise ValueError("Weights contain symbols not present in prices")


class StrategyBase(ABC):
    """
    Base interface for strategy implementations.
    """

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        returns: pd.DataFrame | None,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> None:
        return None

    @abstractmethod
    def predict_weights(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
