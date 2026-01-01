from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def compute_drawdown(equity_curve: pd.Series) -> Dict[str, Any]:
    if equity_curve is None or len(equity_curve) == 0:
        return {"max_drawdown": 0.0, "drawdown": pd.Series(dtype=float)}

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {"max_drawdown": max_dd, "drawdown": drawdown}
