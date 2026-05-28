import numpy as np
import pytest

from learned_bucket_sort.cdf_model import LinearCDFModel, clamp_cdf_predictions
from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS, generate_data
from learned_bucket_sort.learned_sort import assign_learned_bucket_index, learned_bucket_sort
from learned_bucket_sort.scenarios import SUPPORTED_SCENARIOS, generate_scenario_data


class FixedPredictionModel:
    method_name = "fixed_predictions"

    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.fit_ms = None
        self.fit_values = None

    def fit(self, values):
        self.fit_values = np.asarray(values, dtype=np.float64).copy()
        self.fit_ms = 0.0
        return self

    def predict(self, values):
        return clamp_cdf_predictions(self.predictions)


class WrongLengthModel:
    method_name = "wrong_length"

    def __init__(self):
        self.fit_ms = None

    def fit(self, values):
        self.fit_ms = 0.0
        return self

    def predict(self, values):
        return np.array([0.5], dtype=np.float64)


def test_assign_learned_bucket_index_clamps_rank_to_valid_bucket():
    assert assign_learned_bucket_index(-1.0, 10) == 0
    assert assign_learned_bucket_index(0.0, 10) == 0
    assert assign_learned_bucket_index(0.5, 10) == 5
    assert assign_learned_bucket_index(1.0, 10) == 9
    assert assign_learned_bucket_index(2.0, 10) == 9


def test_assign_learned_bucket_index_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="bucket_count must be positive"):
        assign_learned_bucket_index(0.5, 0)

    with pytest.raises(ValueError, match="finite"):
        assign_learned_bucket_index(np.nan, 10)


@pytest.mark.parametrize("distribution", SUPPORTED_DISTRIBUTIONS)
def test_learned_bucket_sort_matches_reference_sort_for_distributions(distribution):
    values = generate_data(distribution, n=250, seed=42)
    original = values.copy()

    result = learned_bucket_sort(values, bucket_count=20, model=LinearCDFModel())

    assert np.array_equal(values, original)
    assert np.allclose(result.sorted_values, np.sort(values))
    assert len(result.bucket_sizes) == 20
    assert sum(result.bucket_sizes) == len(values)
    assert result.metrics.total_count == len(values)
    assert result.method == "linear_cdf"
    assert result.fit_ms >= 0.0
    assert result.bucket_ms >= 0.0
    assert result.sort_ms >= 0.0


@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_learned_bucket_sort_matches_reference_sort_for_scenarios(scenario):
    values = generate_scenario_data(scenario, n=250, seed=42)

    result = learned_bucket_sort(values, bucket_count=20, model=LinearCDFModel())

    assert np.allclose(result.sorted_values, np.sort(values))
    assert result.metrics.total_count == len(values)


def test_learned_bucket_sort_handles_empty_input_without_fitting_model():
    model = FixedPredictionModel([])

    result = learned_bucket_sort([], bucket_count=4, model=model)

    assert result.sorted_values == []
    assert result.bucket_sizes == [0, 0, 0, 0]
    assert result.metrics.empty_bucket_count == 4
    assert result.method == "fixed_predictions"
    assert result.fit_ms == 0.0
    assert result.bucket_ms == 0.0
    assert result.sort_ms == 0.0
    assert model.fit_values is None


def test_learned_bucket_sort_handles_single_value():
    result = learned_bucket_sort([2.5], bucket_count=4, model=LinearCDFModel())

    assert result.sorted_values == [2.5]
    assert result.bucket_sizes == [1, 0, 0, 0]


def test_learned_bucket_sort_handles_all_equal_values():
    result = learned_bucket_sort([3.0, 3.0, 3.0], bucket_count=4, model=LinearCDFModel())

    assert result.sorted_values == [3.0, 3.0, 3.0]
    assert sum(result.bucket_sizes) == 3


def test_learned_bucket_sort_uses_model_predictions_for_bucket_assignment():
    model = FixedPredictionModel([-1.0, 0.2, 2.0])

    result = learned_bucket_sort([1.0, 2.0, 3.0], bucket_count=4, model=model)

    assert result.sorted_values == [1.0, 2.0, 3.0]
    assert result.bucket_sizes == [2, 0, 0, 1]
    assert np.array_equal(model.fit_values, np.array([1.0, 2.0, 3.0]))


def test_learned_bucket_sort_rejects_invalid_values_and_bucket_count():
    with pytest.raises(ValueError, match="finite"):
        learned_bucket_sort([1.0, np.nan], bucket_count=4, model=LinearCDFModel())

    with pytest.raises(ValueError, match="bucket_count must be positive"):
        learned_bucket_sort([1.0], bucket_count=0, model=LinearCDFModel())


def test_learned_bucket_sort_rejects_model_prediction_length_mismatch():
    with pytest.raises(ValueError, match="one prediction per input"):
        learned_bucket_sort([1.0, 2.0], bucket_count=4, model=WrongLengthModel())
