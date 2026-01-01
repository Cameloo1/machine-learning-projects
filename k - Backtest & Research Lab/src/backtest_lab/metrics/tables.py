from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def build_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, val in metrics.items():
        if key in {"equity_curve", "drawdown_series", "rolling_sharpe"}:
            continue
        if isinstance(val, (pd.Series, dict, list, tuple)):
            continue
        rows.append({"metric": key, "value": val})
    return pd.DataFrame(rows)


def metrics_table(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    return build_metrics_table(metrics).to_dict(orient="records")
