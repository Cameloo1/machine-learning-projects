from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def generate_walkforward_windows(
    prices: pd.DataFrame,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    val_days: int = 0,
) -> List[Dict[str, Any]]:
    if train_days < 1 or test_days < 1 or step_days < 1:
        raise ValueError("train_days, test_days, and step_days must be >= 1")
    if val_days < 0:
        raise ValueError("val_days must be >= 0")

    dates = pd.Series(prices["ts"].unique()).sort_values()
    dates = dates.reset_index(drop=True)
    windows: List[Dict[str, Any]] = []

    start_idx = 0
    while True:
        train_start_idx = start_idx
        train_end_idx = train_start_idx + train_days - 1
        val_end_idx = train_end_idx + val_days
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= len(dates):
            break

        train_start = dates.iloc[train_start_idx]
        train_end = dates.iloc[train_end_idx]
        val_start = dates.iloc[train_end_idx + 1] if val_days > 0 else None
        val_end = dates.iloc[val_end_idx] if val_days > 0 else None
        test_start = dates.iloc[test_start_idx]
        test_end = dates.iloc[test_end_idx]

        windows.append(
            {
                "window_id": len(windows),
                "train_start": pd.Timestamp(train_start),
                "train_end": pd.Timestamp(train_end),
                "val_start": pd.Timestamp(val_start) if val_start is not None else None,
                "val_end": pd.Timestamp(val_end) if val_end is not None else None,
                "test_start": pd.Timestamp(test_start),
                "test_end": pd.Timestamp(test_end),
            }
        )

        start_idx += step_days

    return windows
