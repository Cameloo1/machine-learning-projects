from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _rolling_sharpe(returns: pd.Series, window: int, ann_factor: int) -> pd.Series:
    if returns.empty or window < 1 or len(returns) < window:
        return pd.Series([np.nan] * len(returns), index=returns.index)
    mean = returns.rolling(window=window, min_periods=window).mean()
    vol = returns.rolling(window=window, min_periods=window).std(ddof=0)
    sharpe = mean / vol.replace(0.0, np.nan)
    return sharpe * np.sqrt(ann_factor)


def compute_metrics(returns_df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    net = returns_df["net"].astype(float).fillna(0.0)
    gross = returns_df.get("gross", pd.Series([], dtype=float)).astype(float).fillna(0.0)
    turnover = returns_df.get("turnover", pd.Series([], dtype=float)).astype(float).fillna(0.0)
    costs = returns_df.get("costs", pd.Series([], dtype=float)).astype(float).fillna(0.0)
    exposure = returns_df.get("exposure", pd.Series([], dtype=float)).astype(float).fillna(0.0)

    n = int(len(net))
    ann_factor = 252
    equity = (1.0 + net).cumprod()

    if n > 0:
        total_return = float(equity.iloc[-1] - 1.0)
        cagr = float(equity.iloc[-1] ** (ann_factor / max(n, 1)) - 1.0)
        mean_ret = float(net.mean()) * ann_factor
    else:
        total_return = 0.0
        cagr = 0.0
        mean_ret = 0.0

    vol = net.std(ddof=0) * np.sqrt(ann_factor) if n > 1 else 0.0
    sharpe = (net.mean() / net.std(ddof=0)) * np.sqrt(ann_factor) if n > 1 and net.std(ddof=0) != 0 else 0.0

    downside = net.where(net < 0, 0.0)
    downside_vol = downside.std(ddof=0) * np.sqrt(ann_factor) if n > 1 else 0.0
    sortino = (net.mean() / downside.std(ddof=0)) * np.sqrt(ann_factor) if n > 1 and downside.std(ddof=0) != 0 else 0.0

    win_rate = float((net > 0).mean()) if n > 0 else 0.0
    turnover_avg = float(turnover.mean()) if n > 0 else 0.0
    costs_avg = float(costs.mean()) if n > 0 else 0.0
    exposure_avg = float(exposure.mean()) if n > 0 else 0.0

    rolling_window = int(cfg.get("metrics", {}).get("rolling_window", 63)) if isinstance(cfg, dict) else 63
    rolling_sharpe = _rolling_sharpe(net, window=rolling_window, ann_factor=ann_factor)
    rolling_insufficient = len(net) < int(rolling_window)

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annual_return": float(mean_ret),
        "annual_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(win_rate),
        "turnover_avg": float(turnover_avg),
        "costs_avg": float(costs_avg),
        "gross_exposure_avg": float(exposure_avg),
        "exposure_avg": float(exposure_avg),
        "gross_return_avg": float(gross.mean()) if n > 0 else 0.0,
        "net_return_avg": float(net.mean()) if n > 0 else 0.0,
        "equity_curve": equity,
        "rolling_sharpe": rolling_sharpe,
        "rolling_sharpe_insufficient_data": bool(rolling_insufficient),
    }


def write_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    row = {}
    for key, val in metrics.items():
        if key in {"equity_curve", "drawdown_series"}:
            continue
        if isinstance(val, (pd.Series, dict, list, tuple)):
            continue
        row[key] = val
    df = pd.DataFrame([row])
    df.to_csv(path, index=False)
