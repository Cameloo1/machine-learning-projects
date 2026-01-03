from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


_PERCENT_KEYS = {
    "total_return",
    "cagr",
    "annual_return",
    "annual_vol",
    "benchmark_total_return",
    "benchmark_cagr",
    "benchmark_annual_return",
    "benchmark_annual_vol",
    "win_rate",
    "turnover_avg",
    "costs_avg",
    "gross_exposure_avg",
    "exposure_avg",
    "gross_return_avg",
    "net_return_avg",
    "max_drawdown",
}

_LABELS = {
    "total_return": "Total Return",
    "cagr": "CAGR",
    "annual_return": "Annual Return",
    "annual_vol": "Annual Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "benchmark_total_return": "Benchmark Total Return",
    "benchmark_cagr": "Benchmark CAGR",
    "benchmark_annual_return": "Benchmark Annual Return",
    "benchmark_annual_vol": "Benchmark Annual Volatility",
    "benchmark_sharpe": "Benchmark Sharpe",
    "win_rate": "Win Rate",
    "turnover_avg": "Avg Turnover",
    "costs_avg": "Avg Costs",
    "gross_exposure_avg": "Avg Gross Exposure",
    "exposure_avg": "Avg Exposure",
    "gross_return_avg": "Avg Gross Return",
    "net_return_avg": "Avg Net Return",
    "max_drawdown": "Max Drawdown",
}

_ORDER = [
    "total_return",
    "cagr",
    "annual_return",
    "annual_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "benchmark_total_return",
    "benchmark_cagr",
    "benchmark_annual_return",
    "benchmark_annual_vol",
    "benchmark_sharpe",
    "win_rate",
    "gross_return_avg",
    "net_return_avg",
    "turnover_avg",
    "costs_avg",
    "gross_exposure_avg",
    "exposure_avg",
]


def _format_metric(key: str, value: Any) -> Tuple[str, str]:
    label = _LABELS.get(key, key)
    if value is None:
        return label, "n/a"
    if isinstance(value, (int, float)):
        if key in _PERCENT_KEYS:
            return label, f"{value * 100:.2f}%"
        return label, f"{value:.3f}"
    return label, str(value)


def build_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    ordered = [key for key in _ORDER if key in metrics]
    remainder = [key for key in metrics.keys() if key not in ordered]
    for key in ordered + remainder:
        val = metrics.get(key)
        if key in {"equity_curve", "drawdown_series", "rolling_sharpe", "benchmark_equity_curve"}:
            continue
        if isinstance(val, (pd.Series, dict, list, tuple)):
            continue
        label, formatted = _format_metric(key, val)
        rows.append({"metric": label, "value": formatted})
    return pd.DataFrame(rows)


def metrics_table(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return build_metrics_table(metrics).to_dict(orient="records")
