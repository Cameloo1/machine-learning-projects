from __future__ import annotations

import pandas as pd

from backtest_lab.strategies.overlays import _VolTargetScaler


def test_vol_target_scales_down_on_higher_vol() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "ts": dates,
            "symbol": ["AAA"] * 6,
            "close": [100, 101, 102, 130, 90, 140],
        }
    )
    weights = pd.DataFrame({"ts": dates, "symbol": ["AAA"] * 6, "weight": [1.0] * 6})

    scaler = _VolTargetScaler(target_vol=0.1, vol_window=2, min_vol=1e-6, max_scale=5.0)
    scaled = scaler.apply(weights, prices)

    early = scaled.loc[scaled["ts"] <= dates[2], "weight"].abs().mean()
    late = scaled.loc[scaled["ts"] >= dates[3], "weight"].abs().mean()

    assert late < early
