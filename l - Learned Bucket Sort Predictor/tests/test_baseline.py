import numpy as np
import pytest

from learned_bucket_sort.baseline import analytic_bucket_sort, assign_bucket_index
from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS, generate_data


def test_assign_bucket_index_uses_min_max_formula_and_clamps():
    assert assign_bucket_index(0.0, 0.0, 1.0, 10) == 0
    assert assign_bucket_index(0.5, 0.0, 1.0, 10) == 5
    assert assign_bucket_index(1.0, 0.0, 1.0, 10) == 9
    assert assign_bucket_index(-1.0, 0.0, 1.0, 10) == 0
    assert assign_bucket_index(2.0, 0.0, 1.0, 10) == 9


def test_assign_bucket_index_handles_all_equal_bounds():
    assert assign_bucket_index(3.0, 3.0, 3.0, 10) == 0


def test_assign_bucket_index_rejects_invalid_boundaries():
    with pytest.raises(ValueError, match="max_value"):
        assign_bucket_index(1.0, 2.0, 1.0, 10)


@pytest.mark.parametrize("bucket_count", [0, -1])
def test_assign_bucket_index_rejects_invalid_bucket_count(bucket_count):
    with pytest.raises(ValueError, match="bucket_count must be positive"):
        assign_bucket_index(1.0, 0.0, 2.0, bucket_count)


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_analytic_bucket_sort_matches_reference_sort(distribution):
    values = generate_data(distribution, n=500, seed=42)
    original = values.copy()

    result = analytic_bucket_sort(values, bucket_count=25)

    assert np.array_equal(values, original)
    assert np.allclose(result.sorted_values, np.sort(values))
    assert len(result.bucket_sizes) == 25
    assert sum(result.bucket_sizes) == len(values)
    assert result.metrics.total_count == len(values)


def test_analytic_bucket_sort_handles_empty_input():
    result = analytic_bucket_sort([], bucket_count=4)

    assert result.sorted_values == []
    assert result.bucket_sizes == [0, 0, 0, 0]
    assert result.metrics.empty_bucket_count == 4


def test_analytic_bucket_sort_handles_single_value():
    result = analytic_bucket_sort([2.5], bucket_count=4)

    assert result.sorted_values == [2.5]
    assert result.bucket_sizes == [1, 0, 0, 0]


def test_analytic_bucket_sort_handles_all_equal_values():
    result = analytic_bucket_sort([3.0, 3.0, 3.0], bucket_count=4)

    assert result.sorted_values == [3.0, 3.0, 3.0]
    assert result.bucket_sizes == [3, 0, 0, 0]
    assert result.metrics.empty_bucket_count == 3


def test_analytic_bucket_sort_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        analytic_bucket_sort([1.0, np.nan], bucket_count=4)


def test_baseline_occupancy_is_worse_on_skewed_data_than_uniform_data():
    uniform = generate_data("uniform", n=5_000, seed=11)
    lognormal = generate_data("lognormal", n=5_000, seed=11)

    uniform_metrics = analytic_bucket_sort(uniform, bucket_count=50).metrics
    lognormal_metrics = analytic_bucket_sort(lognormal, bucket_count=50).metrics

    assert lognormal_metrics.variance > uniform_metrics.variance * 5
    assert lognormal_metrics.max_bucket_size > uniform_metrics.max_bucket_size
