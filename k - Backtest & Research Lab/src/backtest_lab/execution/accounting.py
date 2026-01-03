from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import pandas as pd

from backtest_lab.execution.constraints import apply_constraints
from backtest_lab.execution.costs import compute_costs, compute_trade_deltas
from backtest_lab.metrics.returns import compute_returns
from backtest_lab.signals.align import align_weights_to_returns

logger = logging.getLogger(__name__)


def _compute_rolling_vol(returns: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    df = returns.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    df["rolling_vol"] = df.groupby("symbol", sort=False)["ret"].transform(
        lambda s: s.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
    )
    return df[["ts", "symbol", "rolling_vol"]]


def _build_returns_frame(
    aligned: pd.DataFrame,
    trades_df: pd.DataFrame,
    costs_by_ts: pd.DataFrame,
) -> pd.DataFrame:
    aligned = aligned.copy()
    aligned["ret_fwd"] = aligned["ret_fwd"].fillna(0.0)
    aligned["gross_component"] = aligned["weight"] * aligned["ret_fwd"]

    gross = aligned.groupby("ts", sort=False)["gross_component"].sum()
    exposure = aligned.groupby("ts", sort=False)["weight"].apply(lambda s: s.abs().sum())
    turnover = trades_df.groupby("ts", sort=False)["abs_dw"].sum()

    decision_df = pd.DataFrame(
        {
            "decision_ts": gross.index,
            "gross": gross.values,
            "exposure": exposure.reindex(gross.index).fillna(0.0).values,
            "turnover": turnover.reindex(gross.index).fillna(0.0).values,
        }
    )

    costs_by_ts = costs_by_ts.set_index("ts").reindex(gross.index).fillna(0.0)
    decision_df["txn_cost"] = costs_by_ts["txn_cost"].values
    decision_df["slippage_cost"] = costs_by_ts["slippage_cost"].values
    decision_df["costs"] = costs_by_ts["total_cost"].values
    decision_df["net"] = decision_df["gross"] - decision_df["costs"]

    # Returns are indexed by realized date (t+1); decision_ts preserves the decision date.
    ts_map = aligned.groupby("ts", sort=False)["ts_next"].first()
    decision_df["ts"] = decision_df["decision_ts"].map(ts_map)

    returns_df = decision_df.loc[decision_df["ts"].notna()].copy()
    returns_df = returns_df[
        [
            "ts",
            "decision_ts",
            "gross",
            "net",
            "exposure",
            "turnover",
            "txn_cost",
            "slippage_cost",
            "costs",
        ]
    ]
    returns_df = returns_df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    return returns_df


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
        renorm_policy=str(cfg.get("renorm_policy", "scale_down_if_exceeded")),
    )

    align_diag: Dict[str, Any] = {}
    aligned, _calendar = align_weights_to_returns(
        prices,
        constrained,
        strict=bool(cfg.get("strict_weight_alignment", False)),
        diagnostics=align_diag,
    )

    trades_df = compute_trade_deltas(aligned[["ts", "symbol", "weight"]])

    rolling_vol = None
    if str(cfg.get("slippage_model", "none")) == "vol_prop":
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

    returns_df = _build_returns_frame(aligned, trades_df, costs_by_ts)

    weights_df = aligned[["ts", "symbol", "weight"]].copy()
    weights_df = weights_df.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)

    trades_df = trades_df.sort_values(["ts", "symbol"], kind="mergesort").reset_index(drop=True)
    trades_df = trades_df.rename(columns={"total_cost": "cost"})

    if diagnostics is not None:
        diagnostics.update(align_diag)
        diagnostics["constraints"] = constraints_diag
        diagnostics["costs"] = costs_diag

    return returns_df, weights_df, trades_df
