"""Bucket occupancy and sorting metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class BucketOccupancy:
    """Inspectable summary of bucket fill balance."""

    bucket_count: int
    total_count: int
    min_bucket_size: int
    max_bucket_size: int
    empty_bucket_count: int
    spread: int
    mean_bucket_size: float
    variance: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def bucket_occupancy(bucket_sizes: Sequence[int]) -> BucketOccupancy:
    """Compute occupancy metrics from per-bucket item counts."""
    sizes = [int(size) for size in bucket_sizes]
    if any(size < 0 for size in sizes):
        raise ValueError("bucket sizes must be non-negative")

    bucket_count = len(sizes)
    total_count = sum(sizes)

    if bucket_count == 0:
        return BucketOccupancy(
            bucket_count=0,
            total_count=0,
            min_bucket_size=0,
            max_bucket_size=0,
            empty_bucket_count=0,
            spread=0,
            mean_bucket_size=0.0,
            variance=0.0,
        )

    min_bucket_size = min(sizes)
    max_bucket_size = max(sizes)
    mean_bucket_size = total_count / bucket_count
    variance = sum((size - mean_bucket_size) ** 2 for size in sizes) / bucket_count

    return BucketOccupancy(
        bucket_count=bucket_count,
        total_count=total_count,
        min_bucket_size=min_bucket_size,
        max_bucket_size=max_bucket_size,
        empty_bucket_count=sum(1 for size in sizes if size == 0),
        spread=max_bucket_size - min_bucket_size,
        mean_bucket_size=mean_bucket_size,
        variance=variance,
    )
