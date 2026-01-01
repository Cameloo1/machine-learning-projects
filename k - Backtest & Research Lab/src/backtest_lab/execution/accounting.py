from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from backtest_lab.execution.constraints import apply_constraints
from backtest_lab.execution.costs import compute_trade_costs
from backtest_lab.signals.align import align_weights_to_returns

logger = logging.getLogger(__name__)


def _pivot_weights(aligned: pd.DataFrame) -> pd.DataFrame:
    weights_pivot = aligned.pivot_table(
        index="ts", columns="symbol", values="weight", fill_value=0.0
    )
    weights_pivot = weights_pivot.sort_index()
    return weights_pivot


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    diagnostics: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply alignment, constraints, and cost model to compute backtest returns.
    """
    if diagnostics is None:
        diagnostics = {}

    constrained = apply_constraints(weights, cfg, diagnostics=diagnostics)
    aligned, calendar = align_weights_to_returns(
        prices,
        constrained,
        strict=bool(cfg.get("strict_weight_alignment", False)),
        diagnostics=diagnostics,
    )

    aligned = aligned.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
    aligned["ret_fwd"] = aligned["ret_fwd"].fillna(0.0)

    aligned = aligned.merge(calendar, on="ts", how="left", suffixes=("", "_cal"))
    aligned["ts_next"] = aligned["ts_next"].fillna(aligned["ts_next_cal"])
    aligned = aligned.drop(columns=["ts_next_cal"])

    aligned["gross_component"] = aligned["weight"] * aligned["ret_fwd"]
    gross = (
        aligned.loc[aligned["ts_next"].notna()]
        .groupby("ts_next", sort=False)["gross_component"]
        .sum()
    )

    weights_pivot = _pivot_weights(aligned)
    dw = weights_pivot.diff().fillna(weights_pivot)
    turnover = dw.abs().sum(axis=1)
    exposure = weights_pivot.abs().sum(axis=1)

    trades_df = (
        dw.reset_index()
        .melt(id_vars="ts", var_name="symbol", value_name="dw")
        .sort_values(["ts", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )

    trades_df, costs_by_ts = compute_trade_costs(
        trades_df, prices, cfg, diagnostics=diagnostics
    )

    turnover_df = (
        turnover.rename("turnover")
        .to_frame()
        .merge(exposure.rename("gross_exposure"), left_index=True, right_index=True)
        .reset_index()
    )

    calendar_map = calendar.rename(columns={"ts": "ts_decision", "ts_next": "ts"})

    turnover_df = turnover_df.merge(
        calendar_map, left_on="ts", right_on="ts_decision", how="left"
    )
    turnover_df = turnover_df.drop(columns=["ts_decision"])

    costs_by_ts = costs_by_ts.merge(
        calendar_map, left_on="ts", right_on="ts_decision", how="left"
    ).drop(columns=["ts_decision"])

    turnover_df = turnover_df.rename(columns={"ts": "ts_realized"})
    costs_by_ts = costs_by_ts.rename(columns={"ts": "ts_realized"})

    gross_df = gross.rename("gross").reset_index().rename(columns={"ts_next": "ts_realized"})

    returns_df = gross_df.merge(turnover_df, on="ts_realized", how="left").merge(
        costs_by_ts, on="ts_realized", how="left"
    )
    returns_df = returns_df.rename(columns={"ts_realized": "ts"})

    returns_df["turnover"] = returns_df["turnover"].fillna(0.0)
    returns_df["gross_exposure"] = returns_df["gross_exposure"].fillna(0.0)
    returns_df["txn_cost"] = returns_df["txn_cost"].fillna(0.0)
    returns_df["slippage_cost"] = returns_df["slippage_cost"].fillna(0.0)
    returns_df["costs"] = returns_df["cost"].fillna(0.0)
    returns_df = returns_df.drop(columns=["cost"])

    returns_df["net"] = returns_df["gross"] - returns_df["costs"]
    returns_df = returns_df.sort_values("ts", kind="mergesort").reset_index(drop=True)

    weights_df = (
        weights_pivot.reset_index()
        .melt(id_vars="ts", var_name="symbol", value_name="weight")
        .sort_values(["ts", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )

    trades_df = trades_df.sort_values(["ts", "symbol"], kind="mergesort").reset_index(
        drop=True
    )

    return returns_df, weights_df, trades_df
