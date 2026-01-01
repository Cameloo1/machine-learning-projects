from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from backtest_lab.metrics.returns import compute_returns

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "alignment_v1"


def align_weights_to_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    strict_weight_alignment: bool = False,
    missing_return_policy: str = "zero_weight",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Align weights at time t to returns at time t+1.

    Contract:
    - weights are decision-time (end of day t)
    - returns applied are next-day returns (t+1)
    - no other shifting is allowed outside this module
    """
    if missing_return_policy not in {"zero_weight", "drop_rows"}:
        raise ValueError("missing_return_policy must be 'zero_weight' or 'drop_rows'")

    if weights.duplicated(subset=["ts", "symbol"]).any():
        dup_count = int(weights.duplicated(subset=["ts", "symbol"]).sum())
        raise ValueError(f"Duplicate weights rows detected: {dup_count}")

    prices_keys = prices[["ts", "symbol"]].drop_duplicates()
    weight_keys = weights[["ts", "symbol"]].drop_duplicates()

    missing_keys = prices_keys.merge(weight_keys, on=["ts", "symbol"], how="left", indicator=True)
    n_missing = int((missing_keys["_merge"] == "left_only").sum())

    extra_keys = weight_keys.merge(prices_keys, on=["ts", "symbol"], how="left", indicator=True)
    n_extra = int((extra_keys["_merge"] == "left_only").sum())

    if strict_weight_alignment and (n_missing > 0 or n_extra > 0):
        raise ValueError(
            f"Weight alignment error: missing={n_missing} extra={n_extra} strict_weight_alignment=True"
        )

    returns = compute_returns(prices)
    returns = returns.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    returns["ret_next"] = returns.groupby("symbol", sort=False)["ret"].shift(-1)

    aligned = prices_keys.merge(returns[["ts", "symbol", "ret_next"]], on=["ts", "symbol"], how="left")
    aligned = aligned.merge(weights[["ts", "symbol", "weight"]], on=["ts", "symbol"], how="left")

    n_filled = int(aligned["weight"].isna().sum())
    if n_filled:
        logger.info("Filled missing weights with zeros: %s", n_filled)
    aligned["weight"] = aligned["weight"].fillna(0.0)

    n_missing_ret_next = int(aligned["ret_next"].isna().sum())
    if n_missing_ret_next:
        logger.info("Missing next-day returns: %s", n_missing_ret_next)
        if missing_return_policy == "zero_weight":
            # Decision: missing t+1 returns zero the weight at t to avoid leakage.
            aligned.loc[aligned["ret_next"].isna(), "weight"] = 0.0
            aligned["ret_next"] = aligned["ret_next"].fillna(0.0)
        else:
            aligned = aligned.loc[aligned["ret_next"].notna()].copy()

    aligned = aligned.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

    diagnostics: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_price_rows": int(len(prices)),
        "n_weight_rows": int(len(weights)),
        "n_unique_price_keys": int(len(prices_keys)),
        "n_unique_weight_keys": int(len(weight_keys)),
        "n_missing_weight_keys": n_missing,
        "n_extra_weight_keys": n_extra,
        "n_filled_weight_rows": n_filled,
        "n_missing_ret_next": n_missing_ret_next,
        "missing_return_policy": missing_return_policy,
        "strict_weight_alignment": bool(strict_weight_alignment),
    }

    return aligned, diagnostics
