from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from backtest_lab.strategies.base import StrategyBase, validate_weights_df

logger = logging.getLogger(__name__)


@dataclass
class _VolTargetScaler:
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
            lambda s: s.rolling(window=self.vol_window, min_periods=self.vol_window).std(ddof=0)
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
            diagnostics["vol_target_target_vol"] = float(self.target_vol)
            diagnostics["vol_target_vol_window"] = int(self.vol_window)
            diagnostics["vol_target_min_vol"] = float(self.min_vol)
            diagnostics["vol_target_max_scale"] = float(self.max_scale)
            if len(scale):
                diagnostics["vol_target_scale_mean"] = float(scale.mean())
                diagnostics["vol_target_scale_min"] = float(scale.min())
                diagnostics["vol_target_scale_max"] = float(scale.max())

        return merged


@dataclass
class VolTargetOverlay(StrategyBase):
    base_strategy: StrategyBase
    params: Dict[str, Any]
    base_cfg: Dict[str, Any]

    def predict_weights(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        params = dict(self.params or {})
        cfg_params = cfg.get("strategy", {}).get("params", {}) if isinstance(cfg, dict) else {}
        params.update(cfg_params)

        target_vol = float(params.get("target_vol", 0.15))
        vol_window = int(params.get("vol_window", 20))
        min_vol = float(params.get("min_vol", 1e-6))
        max_scale = float(params.get("max_scale", 5.0))

        base_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        base_cfg["strategy"] = self.base_cfg

        base_weights = self.base_strategy.predict_weights(
            prices, features, base_cfg, diagnostics=diagnostics
        )

        overlay = _VolTargetScaler(
            target_vol=target_vol,
            vol_window=vol_window,
            min_vol=min_vol,
            max_scale=max_scale,
        )
        weights = overlay.apply(base_weights, prices, diagnostics=diagnostics)
        validate_weights_df(weights, prices=prices)
        return weights
