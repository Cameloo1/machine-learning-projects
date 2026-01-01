from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_vol(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    df = prices[["ts", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "ts"], kind="mergesort")
    df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["rolling_vol"] = df.groupby("symbol", sort=False)["ret"].transform(
        lambda s: s.rolling(window=window).std(ddof=0)
    )
    return df[["ts", "symbol", "rolling_vol"]]


def compute_trade_costs(
    trades_df: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    diagnostics: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute transaction and slippage costs for each trade row.
    """
    df = trades_df.copy()
    df["abs_dw"] = df["dw"].abs()

    cost_bps = float(cfg.get("cost_bps", 0.0))
    df["txn_cost"] = df["abs_dw"] * (cost_bps / 10000.0)

    slippage_model = cfg.get("slippage_model", "none")
    slippage_params = cfg.get("slippage_params", {}) or {}
    df["slippage_cost"] = 0.0

    if slippage_model == "bps":
        slip_bps = float(slippage_params.get("slippage_bps", 0.0))
        df["slippage_cost"] = df["abs_dw"] * (slip_bps / 10000.0)
    elif slippage_model == "vol_prop":
        vol_window = int(slippage_params.get("vol_window", 0))
        slip_mult = float(slippage_params.get("slip_mult", 0.0))
        vol_df = _rolling_vol(prices, window=vol_window)
        df = df.merge(vol_df, on=["ts", "symbol"], how="left")
        n_missing_vol = int(df["rolling_vol"].isna().sum())
        if n_missing_vol:
            logger.info("Missing rolling vol for slippage rows: %s", n_missing_vol)
        df["slippage_cost"] = df["abs_dw"] * df["rolling_vol"].fillna(0.0) * slip_mult
        df = df.drop(columns=["rolling_vol"])
    elif slippage_model == "none":
        pass
    else:
        raise ValueError(f"Unsupported slippage_model: {slippage_model}")

    df["cost"] = df["txn_cost"] + df["slippage_cost"]

    costs_by_ts = (
        df.groupby("ts", sort=False)[["txn_cost", "slippage_cost", "cost"]].sum().reset_index()
    )

    if diagnostics is not None:
        diagnostics.update(
            {
                "cost_bps": cost_bps,
                "slippage_model": slippage_model,
                "slippage_params": slippage_params,
                "costs_total": float(df["cost"].sum()) if len(df) else 0.0,
            }
        )

    return df, costs_by_ts
