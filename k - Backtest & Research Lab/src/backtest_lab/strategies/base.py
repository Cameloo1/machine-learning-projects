from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class StrategyBase(ABC):
    """
    Base interface for strategy implementations.
    """

    def fit(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> None:
        return None

    @abstractmethod
    def predict_weights(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        cfg: Dict[str, Any],
        *,
        diagnostics: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError
