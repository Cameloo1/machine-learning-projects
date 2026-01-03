from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd


def compute_trade_deltas(weights: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "symbol", "weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"compute_trade_deltas missing columns: {sorted(missing)}")
    df = weights.sort_values(["symbol", "ts"], kind="mergesort").copy()
    df["dw"] = df.groupby("symbol", sort=False)["weight"].diff().fillna(df["weight"])
    df["abs_dw"] = df["dw"].abs()
    return df


def compute_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    df = compute_trade_deltas(weights)
    turnover = df.groupby("ts", sort=False)["abs_dw"].sum().reset_index()
    return turnover.rename(columns={"abs_dw": "turnover"})


def compute_costs(
    trades: pd.DataFrame,
    *,
    cost_bps: float,
    slippage_model: str,
    slippage_params: Dict[str, Any] | None = None,
    rolling_vol: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    trades = trades.copy()
    trades["abs_dw"] = trades["dw"].abs()
    trades["txn_cost"] = trades["abs_dw"] * (float(cost_bps) / 10000.0)

    slippage_model = str(slippage_model)
    params = dict(slippage_params or {})
    missing_vol = 0

    if slippage_model == "none":
        trades["slippage_cost"] = 0.0
    elif slippage_model == "bps":
        slip_bps = float(params.get("slip_bps", 0.0))
        trades["slippage_cost"] = trades["abs_dw"] * (slip_bps / 10000.0)
    elif slippage_model == "vol_prop":
        if rolling_vol is None:
            raise ValueError("rolling_vol required when slippage_model == 'vol_prop'")
        slip_mult = float(params.get("slip_mult", 0.0))
        trades = trades.merge(rolling_vol, on=["ts", "symbol"], how="left")
        missing_vol = int(trades["rolling_vol"].isna().sum())
        trades["slippage_cost"] = trades["abs_dw"] * trades["rolling_vol"].fillna(0.0) * slip_mult
        trades = trades.drop(columns=["rolling_vol"])
    else:
        raise ValueError(f"Unsupported slippage_model: {slippage_model}")

    trades["total_cost"] = trades["txn_cost"] + trades["slippage_cost"]
    costs_by_ts = trades.groupby("ts", sort=False)[["txn_cost", "slippage_cost", "total_cost"]].sum()
    costs_by_ts = costs_by_ts.reset_index()

    diagnostics = {
        "cost_bps": float(cost_bps),
        "slippage_model": slippage_model,
        "slippage_params": params,
        "slippage_missing_vol_rows": int(missing_vol),
    }

    return trades, costs_by_ts, diagnostics
