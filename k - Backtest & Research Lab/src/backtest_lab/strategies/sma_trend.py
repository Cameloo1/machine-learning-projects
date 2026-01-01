from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from backtest_lab.strategies.base import StrategyBase

logger = logging.getLogger(__name__)


@dataclass
class SmaTrendStrategy(StrategyBase):
    cfg: Dict[str, Any]

    def predict_weights(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        max_leverage = cfg["execution"]["max_leverage"]
        max_weight = cfg["execution"]["max_weight_per_asset"]

        required_cols = {"ts", "symbol", "sma_fast", "sma_slow"}
        if not required_cols.issubset(features.columns):
            missing = sorted(required_cols - set(features.columns))
            raise ValueError(f"Missing features columns: {missing}")

        df = features[["ts", "symbol", "sma_fast", "sma_slow"]].copy()
        df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

        # Explicit warmup/NaN policy: treat missing SMA inputs as no-signal (weight=0).
        warmup_mask = df[["sma_fast", "sma_slow"]].isna().any(axis=1)
        n_warmup = int(warmup_mask.sum())
        warmup_by_symbol = (
            df.loc[warmup_mask].groupby("symbol", sort=True).size().to_dict()
        )
        warmup_by_symbol = {str(k): int(v) for k, v in warmup_by_symbol.items()}

        if n_warmup:
            logger.info(
                "SMA warmup rows treated as zero-weight n=%s by_symbol=%s",
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
            df.loc[ready_mask, "signal"] = np.where(
                df.loc[ready_mask, "sma_fast"] > df.loc[ready_mask, "sma_slow"],
                1.0,
                0.0,
            )

        active_counts = df.groupby("ts")["signal"].transform("sum")
        base_weight = df["signal"] / active_counts
        base_weight = base_weight.fillna(0.0)
        weight = base_weight.clip(upper=max_weight)

        total = weight.groupby(df["ts"]).transform("sum")
        scale = np.where(total > max_leverage, max_leverage / total, 1.0)
        weight = weight * scale

        weights = df[["ts", "symbol"]].copy()
        weights["weight"] = weight

        if not set(weights["symbol"].unique()).issubset(set(prices["symbol"].unique())):
            raise ValueError("Weights contain symbols not present in prices")

        return weights
