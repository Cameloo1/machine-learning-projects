from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

ALIGNMENT_VERSION = "weights_t_returns_tplus1"


def build_alignment_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build a global trading calendar and its next-day mapping.
    """
    calendar = pd.Series(pd.unique(prices["ts"].sort_values()))
    cal_df = pd.DataFrame({"ts": calendar})
    cal_df["ts_next"] = cal_df["ts"].shift(-1)
    return cal_df


def align_weights_to_returns(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    strict: bool = False,
    diagnostics: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align decision-time weights at t with realized returns at t+1.

    Returns:
        aligned_df: ts (decision), ts_next (realized), symbol, weight, ret_fwd
        calendar_df: global calendar mapping ts -> ts_next
    """
    if diagnostics is None:
        diagnostics = {}

    prices_sorted = prices.sort_values(["symbol", "ts"], kind="mergesort").copy()
    weights_sorted = weights.sort_values(["symbol", "ts"], kind="mergesort").copy()

    if weights_sorted.duplicated(subset=["ts", "symbol"]).any():
        dup_count = int(weights_sorted.duplicated(subset=["ts", "symbol"]).sum())
        raise ValueError(f"Duplicate weights rows detected: {dup_count}")

    price_keys = prices_sorted[["ts", "symbol"]].drop_duplicates()
    weight_keys = weights_sorted[["ts", "symbol"]].drop_duplicates()

    missing_keys = price_keys.merge(weight_keys, on=["ts", "symbol"], how="left", indicator=True)
    n_missing = int((missing_keys["_merge"] == "left_only").sum())
    extra_keys = weight_keys.merge(price_keys, on=["ts", "symbol"], how="left", indicator=True)
    n_extra = int((extra_keys["_merge"] == "left_only").sum())

    if strict and (n_missing > 0 or n_extra > 0):
        raise ValueError(
            f"Weight alignment error: missing={n_missing} extra={n_extra} strict_weight_alignment=True"
        )

    calendar_df = build_alignment_calendar(prices_sorted)
    calendar_map = dict(zip(calendar_df["ts"], calendar_df["ts_next"]))

    prices_sorted["ts_next_global"] = prices_sorted["ts"].map(calendar_map)
    prices_sorted["ts_next_symbol"] = prices_sorted.groupby("symbol", sort=False)["ts"].shift(-1)
    prices_sorted["close_next"] = prices_sorted.groupby("symbol", sort=False)["close"].shift(-1)

    gap_mask = prices_sorted["ts_next_symbol"] != prices_sorted["ts_next_global"]
    missing_next = prices_sorted["ts_next_global"].isna() | prices_sorted["close_next"].isna()
    missing_ret_mask = gap_mask | missing_next

    prices_sorted["ret_fwd"] = prices_sorted["close_next"] / prices_sorted["close"] - 1.0
    prices_sorted.loc[missing_ret_mask, "ret_fwd"] = pd.NA

    merged = prices_sorted.merge(weights_sorted, on=["ts", "symbol"], how="left")
    n_filled = int(merged["weight"].isna().sum())
    if n_filled:
        logger.info("Filled missing weights with zeros: %s", n_filled)
    merged["weight"] = merged["weight"].fillna(0.0)

    weight_pre_zero = merged["weight"].copy()
    n_missing_returns = int(merged["ret_fwd"].isna().sum())
    forced_zero_mask = merged["ret_fwd"].isna() & (weight_pre_zero != 0)
    forced_zero_count = int(forced_zero_mask.sum())
    if n_missing_returns:
        logger.warning("Missing next-day returns detected, zeroing weights: %s", n_missing_returns)
        merged.loc[merged["ret_fwd"].isna(), "weight"] = 0.0

    forced_zero_top = []
    if forced_zero_count:
        counts = (
            merged.loc[forced_zero_mask]
            .groupby("symbol", sort=False)
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        forced_zero_top = [
            {"symbol": str(symbol), "count": int(count)} for symbol, count in counts.items()
        ]

    diagnostics.update(
        {
            "alignment_version": ALIGNMENT_VERSION,
            "n_price_rows": int(len(prices_sorted)),
            "n_weight_rows": int(len(weights_sorted)),
            "n_unique_price_keys": int(len(price_keys)),
            "n_unique_weight_keys": int(len(weight_keys)),
            "n_missing_weight_keys": n_missing,
            "n_extra_weight_keys": n_extra,
            "n_filled_weight_rows": n_filled,
            "n_missing_returns": n_missing_returns,
            "alignment_forced_zero_count": forced_zero_count,
            "alignment_forced_zero_by_symbol_top": forced_zero_top,
            "n_gap_returns": int(gap_mask.sum()),
            "strict_weight_alignment": bool(strict),
        }
    )

    aligned = merged[["ts", "ts_next_global", "symbol", "weight", "ret_fwd"]].rename(
        columns={"ts_next_global": "ts_next"}
    )
    return aligned, calendar_df
