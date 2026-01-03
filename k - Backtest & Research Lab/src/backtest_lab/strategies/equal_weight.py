from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from backtest_lab.strategies.base import StrategyBase, validate_weights_df

logger = logging.getLogger(__name__)


@dataclass
class EqualWeightStrategy(StrategyBase):
    cfg: Dict[str, Any]

    def predict_weights(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = cfg.get("strategy", {}).get("params", {}) if isinstance(cfg, dict) else {}
        rebalance = str(params.get("rebalance_frequency", "monthly")).lower()
        if rebalance not in {"daily", "monthly"}:
            raise ValueError("equal_weight rebalance_frequency must be daily or monthly")

        base = prices[["ts", "symbol"]].copy()
        base["ts"] = pd.to_datetime(base["ts"], errors="coerce")
        base = base.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

        counts = base.groupby("ts", sort=False)["symbol"].transform("count")
        daily_weight = 1.0 / counts.replace(0, np.nan)

        if rebalance == "daily":
            weights = base.copy()
            weights["weight"] = daily_weight.fillna(0.0)
            n_rebalance = int(weights["ts"].nunique())
        else:
            base["month"] = base["ts"].dt.to_period("M")
            first_ts = base.groupby("month", sort=False)["ts"].transform("min")
            rebalance_mask = base["ts"] == first_ts

            weights = base.copy()
            weights["weight"] = np.nan
            weights.loc[rebalance_mask, "weight"] = daily_weight.loc[rebalance_mask]
            weights = weights.sort_values(["symbol", "ts"], kind="mergesort")
            weights["weight"] = weights.groupby("symbol", sort=False)["weight"].ffill().fillna(0.0)
            n_rebalance = int(rebalance_mask.sum())
            weights = weights.drop(columns=["month"])

        weights = weights[["ts", "symbol", "weight"]].copy()

        if diagnostics is not None:
            diagnostics["equal_weight_rebalance_frequency"] = rebalance
            diagnostics["equal_weight_rebalance_events"] = int(n_rebalance)

        validate_weights_df(weights, prices=prices)

        return weights
