"""Empirical CDF training-data primitives."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class EmpiricalCDFTrainingData:
    """Sorted scalar inputs and normalized empirical CDF rank targets."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]


@runtime_checkable
class CDFModel(Protocol):
    """Shared interface for learned CDF estimators."""

    method_name: str
    fit_ms: float | None

    def fit(self, values: Iterable[float]) -> "CDFModel":
        """Fit the model to raw scalar values."""

    def predict(self, values: Iterable[float]) -> NDArray[np.float64]:
        """Predict CDF-like ranks in [0, 1]."""


def build_empirical_cdf_training_data(values: Iterable[float]) -> EmpiricalCDFTrainingData:
    """Build `value -> normalized rank` training pairs for a CDF estimator.

    Duplicate values receive the average normalized rank for their tie group, so
    identical scalar inputs have identical targets.
    """
    array = values_to_1d_float_array(values)
    sorted_values = np.sort(array)
    n = len(sorted_values)

    if n == 0:
        return EmpiricalCDFTrainingData(
            x=np.empty((0, 1), dtype=np.float64),
            y=np.empty((0,), dtype=np.float64),
        )

    targets = _average_tie_group_ranks(sorted_values)
    return EmpiricalCDFTrainingData(
        x=sorted_values.reshape(-1, 1).astype(np.float64, copy=False),
        y=targets,
    )


def values_to_1d_float_array(values: Iterable[float]) -> NDArray[np.float64]:
    """Convert finite scalar values to a 1D float64 array."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be a 1D sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    return array


def values_to_model_input(values: Iterable[float]) -> NDArray[np.float64]:
    """Convert scalar values to the `(n, 1)` shape expected by CDF models."""
    return values_to_1d_float_array(values).reshape(-1, 1)


def clamp_cdf_predictions(predictions: Iterable[float]) -> NDArray[np.float64]:
    """Validate and clamp model predictions to CDF rank bounds."""
    array = values_to_1d_float_array(predictions)
    return np.clip(array, 0.0, 1.0).astype(np.float64, copy=False)


class LinearCDFModel:
    """CPU linear regression estimator for empirical CDF ranks."""

    method_name = "linear_cdf"

    def __init__(self) -> None:
        self.fit_ms: float | None = None
        self._model = LinearRegression()
        self._is_fit = False

    def fit(self, values: Iterable[float]) -> "LinearCDFModel":
        training = build_empirical_cdf_training_data(values)
        if len(training.y) == 0:
            raise ValueError("cannot fit LinearCDFModel with empty values")

        start = perf_counter()
        self._model.fit(training.x, training.y)
        self.fit_ms = (perf_counter() - start) * 1000.0
        self._is_fit = True
        return self

    def predict(self, values: Iterable[float]) -> NDArray[np.float64]:
        if not self._is_fit:
            raise RuntimeError("LinearCDFModel must be fit before predict")

        model_input = values_to_model_input(values)
        if len(model_input) == 0:
            return np.empty((0,), dtype=np.float64)

        predictions = self._model.predict(model_input)
        return clamp_cdf_predictions(predictions)


def _average_tie_group_ranks(sorted_values: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(sorted_values)
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    targets = np.empty(n, dtype=np.float64)
    start = 0

    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1

        midpoint_position = (start + end - 1) / 2.0
        targets[start:end] = midpoint_position / (n - 1)
        start = end

    return targets
