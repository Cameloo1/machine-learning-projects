from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

import pandas as pd

from backtest_lab.signals.ml_ingest import load_predictions
from backtest_lab.strategies.base import StrategyBase, validate_weights_df

logger = logging.getLogger(__name__)


@dataclass
class MLGatedStrategy(StrategyBase):
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
        preds_path = params.get("preds_path") or params.get("predictions_path")
        if preds_path is None:
            raise ValueError("ML gated strategy requires strategy.params.preds_path")
        threshold = float(params.get("threshold", 0.5))
        pred_col = params.get("pred_col") or params.get("prob_col") or "pred"

        base_cfg = params.get("base_strategy") or {"name": "sma_trend", "params": {}}
        if not isinstance(base_cfg, dict) or "name" not in base_cfg:
            raise ValueError("strategy.params.base_strategy must be a mapping with a name")

        cfg_base = dict(cfg)
        cfg_base["strategy"] = base_cfg
        from backtest_lab.strategies import factory as strategy_factory

        base_strategy = strategy_factory.create(base_cfg)

        base_weights = base_strategy.predict_weights(
            prices, features, cfg_base, diagnostics=diagnostics
        )

        preds = load_predictions(preds_path, pred_col=pred_col)
        merged = base_weights.merge(preds, on=["ts", "symbol"], how="left")
        n_missing = int(merged["pred"].isna().sum())
        if n_missing:
            logger.info("ML gated strategy missing preds rows (zeroed): %s", n_missing)

        merged["gate"] = merged["pred"] >= threshold
        merged["weight"] = merged["weight"].where(merged["gate"], 0.0)
        merged = merged.drop(columns=["pred", "gate"])

        if diagnostics is not None:
            diagnostics["ml_gate_threshold"] = threshold
            diagnostics["ml_gate_missing_preds"] = n_missing
            diagnostics["ml_gate_base_strategy"] = base_cfg.get("name")

        validate_weights_df(merged, prices=prices)
        return merged
