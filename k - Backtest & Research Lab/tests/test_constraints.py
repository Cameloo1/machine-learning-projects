from __future__ import annotations

import pandas as pd
import pytest

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


def test_constraints_scale_factor_known() -> None:
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB"],
            "weight": [0.8, 0.8],
        }
    )
    constrained, diagnostics = apply_constraints(
        weights, max_leverage=1.0, max_weight_per_asset=5.0
    )
    out = constrained.set_index("symbol")["weight"]
    assert round(out.loc["AAA"], 6) == 0.5
    assert round(out.loc["BBB"], 6) == 0.5
    assert diagnostics["constraints_n_scaled_dates"] == 1


def test_constraints_sorted_output() -> None:
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "symbol": ["BBB", "AAA"],
            "weight": [0.1, 0.2],
        }
    )
    constrained, _ = apply_constraints(
        weights, max_leverage=1.0, max_weight_per_asset=1.0
    )
    assert constrained[["ts", "symbol"]].values.tolist() == [
        [pd.Timestamp("2020-01-01"), "AAA"],
        [pd.Timestamp("2020-01-02"), "BBB"],
    ]


def test_constraints_duplicate_keys_raise() -> None:
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "AAA"],
            "weight": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError) as exc:
        apply_constraints(weights, max_leverage=1.0, max_weight_per_asset=1.0)
    assert "Duplicate weights rows detected" in str(exc.value)


def test_constraints_error_policy_raises() -> None:
    weights = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "symbol": ["AAA", "BBB"],
            "weight": [0.9, 0.9],
        }
    )
    with pytest.raises(ValueError) as exc:
        apply_constraints(
            weights,
            max_leverage=1.0,
            max_weight_per_asset=0.6,
            renorm_policy="error_if_exceeded",
        )
    assert "cap exceeded" in str(exc.value)
