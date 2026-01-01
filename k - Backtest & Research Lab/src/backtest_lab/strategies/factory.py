from __future__ import annotations

from typing import Any, Dict

from backtest_lab.strategies.ml_gated import MLGatedStrategy
from backtest_lab.strategies.overlays import VolTargetOverlay
from backtest_lab.strategies.rsi_mr import RsiMeanReversionStrategy
from backtest_lab.strategies.sma_trend import SmaTrendStrategy


def create(cfg: Dict[str, Any]) -> Any:
    name = cfg.get("name")
    if name == "sma_trend":
        return SmaTrendStrategy(cfg)
    if name == "rsi_mr":
        return RsiMeanReversionStrategy(cfg)
    if name == "vol_target_trend":
        params = dict(cfg.get("params", {}) or {})
        base_name = params.get("base", "sma_trend")
        base_params = params.get("base_params", {}) or {}
        if base_name == "vol_target_trend":
            raise ValueError("vol_target_trend cannot wrap itself")
        base_strategy = create({"name": base_name, "params": base_params})
        return VolTargetOverlay(base_strategy=base_strategy, params=params)
    if name == "ml_gated":
        return MLGatedStrategy(cfg)
    raise ValueError(f"Unknown strategy name: {name}")
