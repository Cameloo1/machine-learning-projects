"""Analytic min-max bucket sort baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from learned_bucket_sort.metrics import BucketOccupancy, bucket_occupancy


@dataclass(frozen=True)
class BaselineSortResult:
    """Sorted output plus the bucket state used to produce it."""

    sorted_values: list[float]
    bucket_sizes: list[int]
    metrics: BucketOccupancy


def assign_bucket_index(
    value: float,
    min_value: float,
    max_value: float,
    bucket_count: int,
) -> int:
    """Assign a value to an analytic min-max bucket."""
    _validate_bucket_count(bucket_count)

    if max_value < min_value:
        raise ValueError("max_value must be greater than or equal to min_value")

    if max_value == min_value:
        return 0

    normalized = (value - min_value) / (max_value - min_value)
    raw_index = math.floor(bucket_count * normalized)
    return min(max(raw_index, 0), bucket_count - 1)


def analytic_bucket_sort(values: Iterable[float], bucket_count: int) -> BaselineSortResult:
    """Sort values using classical min-max bucket assignment."""
    _validate_bucket_count(bucket_count)

    items = list(values)
    _validate_finite(items)

    if not items:
        bucket_sizes = [0] * bucket_count
        return BaselineSortResult(
            sorted_values=[],
            bucket_sizes=bucket_sizes,
            metrics=bucket_occupancy(bucket_sizes),
        )

    min_value = min(items)
    max_value = max(items)
    buckets: list[list[float]] = [[] for _ in range(bucket_count)]

    for value in items:
        index = assign_bucket_index(value, min_value, max_value, bucket_count)
        buckets[index].append(value)

    sorted_values: list[float] = []
    for bucket in buckets:
        sorted_values.extend(sorted(bucket))

    bucket_sizes = [len(bucket) for bucket in buckets]
    return BaselineSortResult(
        sorted_values=sorted_values,
        bucket_sizes=bucket_sizes,
        metrics=bucket_occupancy(bucket_sizes),
    )


def _validate_bucket_count(bucket_count: int) -> None:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")


def _validate_finite(values: Iterable[float]) -> None:
    if any(not math.isfinite(value) for value in values):
        raise ValueError("values must be finite")
