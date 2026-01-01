from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "costs_v1"


def compute_trade_deltas(weights_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset weight changes (dw) in long format.
    """
    dw = weights_wide.diff().fillna(weights_wide)
    trades = (
        dw.reset_index()
        .melt(id_vars="ts", var_name="symbol", value_name="dw")
        .sort_values(["ts", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    trades["abs_dw"] = trades["dw"].abs()
    return trades


def compute_turnover(weights_wide: pd.DataFrame) -> pd.Series:
    dw = weights_wide.diff().fillna(weights_wide)
    return dw.abs().sum(axis=1)


def compute_costs(
    trades: pd.DataFrame,
    *,
    cost_bps: float,
    slippage_model: str,
    slippage_params: Dict[str, Any] | None = None,
    rolling_vol: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Compute transaction costs and slippage from trades.
    """
    if slippage_params is None:
        slippage_params = {}

    trades = trades.copy()
    trades["cost"] = trades["abs_dw"] * (float(cost_bps) / 10000.0)

    slippage = pd.Series(0.0, index=trades.index)
    n_missing_vol = 0

    if slippage_model == "none":
        slippage = pd.Series(0.0, index=trades.index)
    elif slippage_model == "bps":
        slip_bps = slippage_params.get("slip_bps")
        if slip_bps is None:
            raise ValueError("slippage_params.slip_bps is required when slippage_model == 'bps'")
        slippage = trades["abs_dw"] * (float(slip_bps) / 10000.0)
    elif slippage_model == "vol_prop":
        slip_mult = slippage_params.get("slip_mult")
        if slip_mult is None:
            raise ValueError("slippage_params.slip_mult is required when slippage_model == 'vol_prop'")
        if rolling_vol is None:
            raise ValueError("rolling_vol is required when slippage_model == 'vol_prop'")
        merged = trades.merge(
            rolling_vol[["ts", "symbol", "rolling_vol"]],
            on=["ts", "symbol"],
            how="left",
        )
        n_missing_vol = int(merged["rolling_vol"].isna().sum())
        if n_missing_vol:
            logger.info("Missing rolling vol for slippage rows=%s", n_missing_vol)
        vol = merged["rolling_vol"].fillna(0.0)
        slippage = merged["abs_dw"] * float(slip_mult) * vol
        trades = merged.drop(columns=["rolling_vol"])
    else:
        raise ValueError(f"Unsupported slippage_model: {slippage_model}")

    trades["slippage"] = slippage
    trades["total_cost"] = trades["cost"] + trades["slippage"]

    costs_by_ts = (
        trades.groupby("ts", sort=False)[["cost", "slippage", "total_cost"]]
        .sum()
        .reset_index()
    )

    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "n_trades": int(len(trades)),
        "cost_bps": float(cost_bps),
        "slippage_model": str(slippage_model),
        "n_missing_vol": int(n_missing_vol),
    }

    return trades, costs_by_ts, diagnostics
