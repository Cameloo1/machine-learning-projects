from __future__ import annotations

import pandas as pd

from backtest_lab.execution.constraints import apply_constraints


def test_constraints_clip_and_scale() -> None:
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB"],
            "weight": [0.8, 0.8],
        }
    )

    constrained, diagnostics = apply_constraints(
        weights, max_leverage=1.0, max_weight_per_asset=0.6
    )

    out = constrained.set_index("symbol")["weight"]
    assert round(out.loc["AAA"], 6) == 0.5
    assert round(out.loc["BBB"], 6) == 0.5
    assert diagnostics["n_clipped"] == 2
    assert diagnostics["n_scaled_timestamps"] == 1
