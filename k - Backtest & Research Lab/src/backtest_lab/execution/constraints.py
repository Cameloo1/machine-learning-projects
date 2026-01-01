from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "constraints_v1"


def apply_constraints(
    weights: pd.DataFrame,
    *,
    max_leverage: float,
    max_weight_per_asset: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enforce per-asset and leverage caps with deterministic renormalization.

    - Per-asset cap is applied to absolute weight.
    - Leverage cap uses sum(abs(weight)) per timestamp.
    """
    required = {"ts", "symbol", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"apply_constraints missing columns: {sorted(missing)}")

    df = weights[["ts", "symbol", "weight"]].copy()
    df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

    n_nan_weights = int(df["weight"].isna().sum())
    if n_nan_weights:
        logger.info("Filling NaN weights with 0.0: %s", n_nan_weights)
        df["weight"] = df["weight"].fillna(0.0)

    before = df["weight"].copy()
    df["weight"] = df["weight"].clip(lower=-max_weight_per_asset, upper=max_weight_per_asset)
    n_clipped = int((before != df["weight"]).sum())

    leverage = df.groupby("ts", sort=False)["weight"].apply(lambda s: float(np.abs(s).sum()))
    leverage = leverage.reindex(df["ts"].unique())

    scale_map = {}
    n_scaled_ts = 0
    for ts_val, lev in leverage.items():
        if lev > max_leverage and lev > 0:
            scale = max_leverage / lev
            scale_map[ts_val] = scale
            n_scaled_ts += 1
        else:
            scale_map[ts_val] = 1.0

    df["weight"] = df.apply(
        lambda row: row["weight"] * scale_map.get(row["ts"], 1.0), axis=1
    )

    diagnostics: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_rows": int(len(df)),
        "n_nan_weights": n_nan_weights,
        "n_clipped": n_clipped,
        "n_scaled_timestamps": n_scaled_ts,
        "max_leverage": float(max_leverage),
        "max_weight_per_asset": float(max_weight_per_asset),
    }

    return df, diagnostics
