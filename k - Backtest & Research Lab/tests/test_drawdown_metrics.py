from __future__ import annotations

import pandas as pd

from backtest_lab.metrics.drawdown import compute_drawdown


def test_drawdown_max() -> None:
    equity = pd.Series([1.0, 1.1, 1.0, 1.2], index=pd.RangeIndex(4))
    out = compute_drawdown(equity)
    assert abs(out["max_drawdown"] + 0.0909090909) < 1e-6
    assert len(out["drawdown"]) == 4
