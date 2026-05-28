import numpy as np
import pytest

from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS, generate_data


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_generate_data_is_deterministic(distribution):
    first = generate_data(distribution, n=128, seed=123)
    second = generate_data(distribution, n=128, seed=123)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_generate_data_returns_1d_finite_float_array(distribution):
    values = generate_data(distribution, n=64, seed=7)

    assert values.shape == (64,)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()


def test_generate_data_allows_empty_data():
    values = generate_data("uniform", n=0, seed=1)

    assert values.shape == (0,)
    assert values.dtype == np.float64


def test_generate_data_rejects_negative_size():
    with pytest.raises(ValueError, match="n must be non-negative"):
        generate_data("uniform", n=-1, seed=1)


def test_generate_data_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="unsupported distribution"):
        generate_data("pareto", n=10, seed=1)
