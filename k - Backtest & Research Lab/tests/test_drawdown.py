from __future__ import annotations

import pandas as pd

from backtest_lab.metrics.drawdown import compute_drawdown


def test_drawdown_max() -> None:
    equity = pd.Series([1.0, 1.1, 1.05, 1.2])
    out = compute_drawdown(equity)
    assert round(out["max_drawdown"], 6) == round(-0.0454545, 6)
