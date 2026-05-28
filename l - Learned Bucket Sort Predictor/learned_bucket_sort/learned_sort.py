"""Learned CDF bucket sort implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from learned_bucket_sort.cdf_model import CDFModel, LinearCDFModel, values_to_1d_float_array
from learned_bucket_sort.metrics import BucketOccupancy, bucket_occupancy


@dataclass(frozen=True)
class LearnedSortResult:
    """Sorted output plus bucket metrics and learned-model timings."""

    sorted_values: list[float]
    bucket_sizes: list[int]
    metrics: BucketOccupancy
    method: str
    fit_ms: float
    bucket_ms: float
    sort_ms: float


def assign_learned_bucket_index(predicted_rank: float, bucket_count: int) -> int:
    """Assign a CDF rank prediction to a bucket index."""
    _validate_bucket_count(bucket_count)
    if not math.isfinite(predicted_rank):
        raise ValueError("predicted_rank must be finite")

    clamped_rank = min(max(predicted_rank, 0.0), 1.0)
    raw_index = math.floor(bucket_count * clamped_rank)
    return min(max(raw_index, 0), bucket_count - 1)


def learned_bucket_sort(
    values: Iterable[float],
    bucket_count: int,
    model: CDFModel | None = None,
) -> LearnedSortResult:
    """Sort values using a learned CDF model for bucket assignment."""
    _validate_bucket_count(bucket_count)
    items = values_to_1d_float_array(values)
    model = model or LinearCDFModel()

    if len(items) == 0:
        bucket_sizes = [0] * bucket_count
        return LearnedSortResult(
            sorted_values=[],
            bucket_sizes=bucket_sizes,
            metrics=bucket_occupancy(bucket_sizes),
            method=model.method_name,
            fit_ms=0.0,
            bucket_ms=0.0,
            sort_ms=0.0,
        )

    model.fit(items)
    fit_ms = float(model.fit_ms or 0.0)

    bucket_start = perf_counter()
    predictions = model.predict(items)
    if len(predictions) != len(items):
        raise ValueError("model must return one prediction per input value")

    buckets: list[list[float]] = [[] for _ in range(bucket_count)]
    for value, predicted_rank in zip(items, predictions, strict=True):
        index = assign_learned_bucket_index(float(predicted_rank), bucket_count)
        buckets[index].append(float(value))
    bucket_ms = (perf_counter() - bucket_start) * 1000.0

    sort_start = perf_counter()
    sorted_values: list[float] = []
    for bucket in buckets:
        sorted_values.extend(sorted(bucket))
    sort_ms = (perf_counter() - sort_start) * 1000.0

    bucket_sizes = [len(bucket) for bucket in buckets]
    return LearnedSortResult(
        sorted_values=sorted_values,
        bucket_sizes=bucket_sizes,
        metrics=bucket_occupancy(bucket_sizes),
        method=model.method_name,
        fit_ms=fit_ms,
        bucket_ms=bucket_ms,
        sort_ms=sort_ms,
    )


def _validate_bucket_count(bucket_count: int) -> None:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
