import pytest

from learned_bucket_sort.metrics import BucketOccupancy, bucket_occupancy


def test_bucket_occupancy_summarizes_bucket_sizes():
    metrics = bucket_occupancy([0, 3, 2])

    assert metrics == BucketOccupancy(
        bucket_count=3,
        total_count=5,
        min_bucket_size=0,
        max_bucket_size=3,
        empty_bucket_count=1,
        spread=3,
        mean_bucket_size=pytest.approx(5 / 3),
        variance=pytest.approx(14 / 9),
    )


def test_bucket_occupancy_handles_empty_sequence():
    metrics = bucket_occupancy([])

    assert metrics.bucket_count == 0
    assert metrics.total_count == 0
    assert metrics.empty_bucket_count == 0
    assert metrics.variance == 0.0


def test_bucket_occupancy_rejects_negative_sizes():
    with pytest.raises(ValueError, match="non-negative"):
        bucket_occupancy([1, -1, 2])


def test_bucket_occupancy_converts_to_dict():
    metrics = bucket_occupancy([1, 1])

    assert metrics.to_dict() == {
        "bucket_count": 2,
        "total_count": 2,
        "min_bucket_size": 1,
        "max_bucket_size": 1,
        "empty_bucket_count": 0,
        "spread": 0,
        "mean_bucket_size": 1.0,
        "variance": 0.0,
    }
