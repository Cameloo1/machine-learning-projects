from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from backtest_lab.strategies.base import StrategyBase
from backtest_lab.strategies.sma_trend import SmaTrendStrategy

logger = logging.getLogger(__name__)


@dataclass
class VolTargetOverlay:
    target_vol: float
    vol_window: int
    min_vol: float = 1e-6
    max_scale: float = 5.0

    def apply(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if self.vol_window < 2:
            raise ValueError("vol_window must be >= 2 for vol targeting")

        vol = prices[["ts", "symbol", "close"]].copy()
        vol = vol.sort_values(["symbol", "ts"], kind="mergesort")
        vol["ret"] = vol.groupby("symbol", sort=False)["close"].pct_change()
        vol["vol"] = vol.groupby("symbol", sort=False)["ret"].transform(
            lambda s: s.rolling(window=self.vol_window).std(ddof=0)
        )
        vol["vol"] = vol["vol"] * np.sqrt(252)

        merged = weights.merge(vol[["ts", "symbol", "vol"]], on=["ts", "symbol"], how="left")
        missing_vol = int(merged["vol"].isna().sum())
        if missing_vol:
            logger.info("Vol targeting missing vol rows (zeroed): %s", missing_vol)

        vol_safe = merged["vol"].clip(lower=self.min_vol)
        scale = (self.target_vol / vol_safe).clip(upper=self.max_scale)
        scale = scale.fillna(0.0)

        merged["weight"] = merged["weight"] * scale
        merged = merged.drop(columns=["vol"])

        if diagnostics is not None:
            diagnostics["vol_target_missing_vol"] = missing_vol
            diagnostics["vol_target_target_vol"] = self.target_vol
            diagnostics["vol_target_vol_window"] = self.vol_window

        return merged


@dataclass
class VolTargetTrendStrategy(StrategyBase):
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
        target_vol = float(params.get("target_vol", 0.15))
        vol_window = int(params.get("vol_window", 20))
        min_vol = float(params.get("min_vol", 1e-6))
        max_scale = float(params.get("max_scale", 5.0))

        base = SmaTrendStrategy(cfg)
        base_weights = base.predict_weights(prices, features, cfg, diagnostics=diagnostics)

        overlay = VolTargetOverlay(
            target_vol=target_vol, vol_window=vol_window, min_vol=min_vol, max_scale=max_scale
        )
        return overlay.apply(base_weights, prices, diagnostics=diagnostics)
