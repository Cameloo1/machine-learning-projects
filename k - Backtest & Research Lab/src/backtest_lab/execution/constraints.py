from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RENORM_POLICIES = {"scale_down_if_exceeded", "error_if_exceeded"}


def apply_constraints(
    weights: pd.DataFrame,
    max_leverage: float,
    max_weight_per_asset: float,
    *,
    renorm_policy: str = "scale_down_if_exceeded",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    required = {"ts", "symbol", "weight"}
    if not required.issubset(weights.columns):
        missing = sorted(required - set(weights.columns))
        raise ValueError(f"Missing columns in weights: {missing}")

    renorm_policy = str(renorm_policy)
    if renorm_policy not in _RENORM_POLICIES:
        raise ValueError(f"Unsupported renorm_policy: {renorm_policy}")

    df = weights[["ts", "symbol", "weight"]].copy()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    if df["weight"].isna().any():
        bad_count = int(df["weight"].isna().sum())
        raise ValueError(f"Non-numeric weights detected: {bad_count}")
    if not np.isfinite(df["weight"].to_numpy()).all():
        raise ValueError("Non-finite weights detected")

    df = df.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)

    if df.duplicated(subset=["ts", "symbol"]).any():
        dup_count = int(df.duplicated(subset=["ts", "symbol"]).sum())
        raise ValueError(f"Duplicate weights rows detected: {dup_count}")

    gross_pre = df.groupby("ts", sort=False)["weight"].apply(lambda s: s.abs().sum())
    exceeds_weight = df["weight"].abs() > float(max_weight_per_asset)
    exceeds_leverage = gross_pre > float(max_leverage)

    if renorm_policy == "error_if_exceeded":
        if exceeds_weight.any():
            count = int(exceeds_weight.sum())
            raise ValueError(f"Per-asset cap exceeded in {count} rows")
        if exceeds_leverage.any():
            count = int(exceeds_leverage.sum())
            raise ValueError(f"Leverage cap exceeded on {count} timestamps")

    before_clip = df["weight"].copy()
    clipped_weight = df["weight"].clip(lower=-max_weight_per_asset, upper=max_weight_per_asset)
    df["weight"] = clipped_weight
    n_clipped = int((before_clip != clipped_weight).sum())

    leverage = df.groupby("ts", sort=False)["weight"].apply(lambda s: s.abs().sum())
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

    clipped_by_ts = (
        df.assign(_clipped=(before_clip != clipped_weight))
        .groupby("ts", sort=False)["_clipped"]
        .sum()
        .astype(int)
    )
    gross_post = df.groupby("ts", sort=False)["weight"].apply(lambda s: s.abs().sum())
    per_ts = pd.DataFrame(
        {
            "ts": gross_pre.index,
            "gross_pre": gross_pre.values,
            "gross_post": gross_post.reindex(gross_pre.index).fillna(0.0).values,
            "scale_factor": scale_series.reindex(gross_pre.index).fillna(1.0).values,
            "n_clipped_symbols": clipped_by_ts.reindex(gross_pre.index).fillna(0).astype(int).values,
        }
    )
    per_ts = per_ts.sort_values("gross_pre", ascending=False)
    worst_limit = 10
    per_ts_top = []
    for row in per_ts.head(worst_limit).itertuples(index=False):
        per_ts_top.append(
            {
                "ts": pd.Timestamp(row.ts).isoformat(),
                "gross_pre": float(row.gross_pre),
                "gross_post": float(row.gross_post),
                "scale_factor": float(row.scale_factor),
                "n_clipped_symbols": int(row.n_clipped_symbols),
            }
        )

    diagnostics = {
        "constraints_max_leverage": float(max_leverage),
        "constraints_max_weight_per_asset": float(max_weight_per_asset),
        "constraints_renorm_policy": renorm_policy,
        "constraints_n_clipped": float(n_clipped),
        "constraints_n_scaled_dates": float(n_scaled),
        "constraints_leverage_max": float(gross_pre.max()) if len(gross_pre) else 0.0,
        "constraints_leverage_post_max": float(gross_post.max()) if len(gross_post) else 0.0,
        "constraints_gross_pre_mean": float(gross_pre.mean()) if len(gross_pre) else 0.0,
        "constraints_gross_post_mean": float(gross_post.mean()) if len(gross_post) else 0.0,
        "constraints_per_ts_top": per_ts_top,
        "n_clipped": float(n_clipped),
        "n_scaled_timestamps": float(n_scaled),
    }

    df = df.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
    return df, diagnostics
