"""Utility helpers shared across modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def make_rng(random_state: int | None = None) -> np.random.Generator:
    """Return a numpy Generator seeded for reproducibility."""
    return np.random.default_rng(random_state)


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a ValueError if any of the required columns are missing."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

