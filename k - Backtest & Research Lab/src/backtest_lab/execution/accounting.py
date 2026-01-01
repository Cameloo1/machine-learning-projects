from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from backtest_lab.execution.constraints import apply_constraints
from backtest_lab.execution.costs import compute_costs, compute_trade_deltas, compute_turnover
from backtest_lab.metrics.returns import compute_returns
from backtest_lab.signals.align import align_weights_to_returns

logger = logging.getLogger(__name__)


def _compute_rolling_vol(returns: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    df = returns.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    df["rolling_vol"] = df.groupby("symbol", sort=False)["ret"].transform(
        lambda s: s.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
    )
    return df[["ts", "symbol", "rolling_vol"]]


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    diagnostics: Dict[str, Any] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = prices.sort_values(["symbol", "ts"], kind="mergesort").copy()
    weights = weights.sort_values(["symbol", "ts"], kind="mergesort").copy()

    constrained, constraints_diag = apply_constraints(
        weights,
        max_leverage=float(cfg["max_leverage"]),
        max_weight_per_asset=float(cfg["max_weight_per_asset"]),
    )

    aligned, align_diag = align_weights_to_returns(
        prices,
        constrained,
        strict_weight_alignment=bool(cfg.get("strict_weight_alignment", False)),
        missing_return_policy="zero_weight",
    )

    aligned["gross_component"] = aligned["weight"] * aligned["ret_next"]
    gross = aligned.groupby("ts", sort=False)["gross_component"].sum().sort_index()

    weights_pivot = aligned.pivot_table(
        index="ts", columns="symbol", values="weight", fill_value=0.0
    ).sort_index()

    exposure = weights_pivot.abs().sum(axis=1)
    turnover = compute_turnover(weights_pivot)
    trades_df = compute_trade_deltas(weights_pivot)

    rolling_vol = None
    if cfg.get("slippage_model") == "vol_prop":
        returns = compute_returns(prices)
        vol_window = int(cfg.get("slippage_params", {}).get("vol_window"))
        rolling_vol = _compute_rolling_vol(returns, vol_window)

    trades_df, costs_by_ts, costs_diag = compute_costs(
        trades_df,
        cost_bps=float(cfg.get("cost_bps", 0.0)),
        slippage_model=str(cfg.get("slippage_model", "none")),
        slippage_params=dict(cfg.get("slippage_params", {}) or {}),
        rolling_vol=rolling_vol,
    )

    costs_by_ts = costs_by_ts.set_index("ts")
    turnover = turnover.reindex(gross.index).fillna(0.0)
    exposure = exposure.reindex(gross.index).fillna(0.0)
    costs_by_ts = costs_by_ts.reindex(gross.index).fillna(0.0)

    total_costs = costs_by_ts["total_cost"]
    net = gross - total_costs

    returns_df = pd.DataFrame(
        {
            "ts": gross.index,
            "gross": gross.values,
            "net": net.values,
            "exposure": exposure.values,
            "turnover": turnover.values,
            "cost": costs_by_ts["cost"].values,
            "slippage": costs_by_ts["slippage"].values,
            "costs": total_costs.values,
        }
    )

    weights_df = (
        weights_pivot.reset_index()
        .melt(id_vars="ts", var_name="symbol", value_name="weight")
        .sort_values(["ts", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )

    trades_df = trades_df.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)

    if diagnostics is not None:
        diagnostics.update(align_diag)
        diagnostics["constraints"] = constraints_diag
        diagnostics["costs"] = costs_diag

    return returns_df, weights_df, trades_df
