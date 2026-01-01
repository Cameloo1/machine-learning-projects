from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_constraints(
    weights: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    diagnostics: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Apply per-asset caps and leverage caps to weights.
    """
    required = {"ts", "symbol", "weight"}
    if not required.issubset(weights.columns):
        missing = sorted(required - set(weights.columns))
        raise ValueError(f"Missing columns in weights: {missing}")

    max_leverage = float(cfg.get("max_leverage", 1.0))
    max_weight = float(cfg.get("max_weight_per_asset", max_leverage))

    df = weights[["ts", "symbol", "weight"]].copy()
    df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

    if df.duplicated(subset=["ts", "symbol"]).any():
        dup_count = int(df.duplicated(subset=["ts", "symbol"]).sum())
        raise ValueError(f"Duplicate weights rows detected: {dup_count}")

    before_clip = df["weight"].copy()
    df["weight"] = df["weight"].clip(lower=-max_weight, upper=max_weight)
    n_clipped = int((before_clip != df["weight"]).sum())

    grouped = df.groupby("ts", sort=False)["weight"]
    leverage = grouped.apply(lambda s: s.abs().sum())
    scale = np.where(leverage > max_leverage, max_leverage / leverage, 1.0)
    scale_series = pd.Series(scale, index=leverage.index)

    df = df.merge(scale_series.rename("scale"), left_on="ts", right_index=True, how="left")
    df["weight"] = df["weight"] * df["scale"]
    df = df.drop(columns=["scale"])

    n_scaled = int((leverage > max_leverage).sum())
    if n_clipped:
        logger.info("Clipped weights to per-asset cap: %s", n_clipped)
    if n_scaled:
        logger.info("Scaled weights to leverage cap: %s", n_scaled)

    if diagnostics is not None:
        diagnostics.update(
            {
                "constraints_max_leverage": max_leverage,
                "constraints_max_weight_per_asset": max_weight,
                "constraints_n_clipped": n_clipped,
                "constraints_n_scaled_dates": n_scaled,
                "constraints_leverage_max": float(leverage.max()) if len(leverage) else 0.0,
            }
        )

    return df
