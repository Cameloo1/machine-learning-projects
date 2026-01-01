from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from backtest_lab.strategies.base import StrategyBase

logger = logging.getLogger(__name__)


@dataclass
class RsiMeanReversionStrategy(StrategyBase):
    cfg: Dict[str, Any]

    def predict_weights(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        rsi_low = float(cfg["features"]["rsi_low"])
        rsi_high = float(cfg["features"]["rsi_high"])

        required_cols = {"ts", "symbol", "rsi"}
        if not required_cols.issubset(features.columns):
            missing = sorted(required_cols - set(features.columns))
            raise ValueError(f"Missing features columns: {missing}")

        df = features[["ts", "symbol", "rsi"]].copy()
        df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

        warmup_mask = df["rsi"].isna()
        n_warmup = int(warmup_mask.sum())
        warmup_by_symbol = (
            df.loc[warmup_mask].groupby("symbol", sort=True).size().to_dict()
        )
        warmup_by_symbol = {str(k): int(v) for k, v in warmup_by_symbol.items()}

        if n_warmup:
            logger.info(
                "RSI warmup rows treated as zero-weight n=%s by_symbol=%s",
                n_warmup,
                warmup_by_symbol,
            )

        if diagnostics is not None:
            diagnostics["warmup_policy"] = "zero_weight"
            diagnostics["warmup_nan_rows"] = n_warmup
            diagnostics["warmup_nan_rows_by_symbol"] = warmup_by_symbol

        df["signal"] = 0.0
        ready_mask = ~warmup_mask
        if ready_mask.any():
            # Decision: long below rsi_low, short above rsi_high, flat otherwise.
            df.loc[ready_mask, "signal"] = np.where(
                df.loc[ready_mask, "rsi"] < rsi_low,
                1.0,
                np.where(df.loc[ready_mask, "rsi"] > rsi_high, -1.0, 0.0),
            )

        active_counts = df.groupby("ts")["signal"].transform(lambda s: (s != 0).sum())
        weight = df["signal"] / active_counts.replace(0, np.nan)
        weight = weight.fillna(0.0)

        weights = df[["ts", "symbol"]].copy()
        weights["weight"] = weight

        if not set(weights["symbol"].unique()).issubset(set(prices["symbol"].unique())):
            raise ValueError("Weights contain symbols not present in prices")

        return weights
