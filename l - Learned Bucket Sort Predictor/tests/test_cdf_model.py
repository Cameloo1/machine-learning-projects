import numpy as np
import pytest

from learned_bucket_sort.cdf_model import (
    CDFModel,
    LinearCDFModel,
    build_empirical_cdf_training_data,
    clamp_cdf_predictions,
    values_to_1d_float_array,
    values_to_model_input,
)


class DummyCDFModel:
    method_name = "dummy"

    def __init__(self):
        self.fit_ms = None

    def fit(self, values):
        self.fit_ms = 0.0
        return self

    def predict(self, values):
        return clamp_cdf_predictions([0.25 for _ in values])


def test_build_empirical_cdf_training_data_sorts_values_and_shapes_arrays():
    values = np.array([3.0, 1.0, 2.0], dtype=np.float64)

    training = build_empirical_cdf_training_data(values)

    assert training.x.shape == (3, 1)
    assert training.y.shape == (3,)
    assert training.x.dtype == np.float64
    assert training.y.dtype == np.float64
    assert np.array_equal(training.x[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(training.y, np.array([0.0, 0.5, 1.0]))


def test_build_empirical_cdf_training_data_does_not_mutate_input():
    values = np.array([3.0, 1.0, 2.0], dtype=np.float64)
    original = values.copy()

    build_empirical_cdf_training_data(values)

    assert np.array_equal(values, original)


def test_build_empirical_cdf_training_data_targets_are_bounded_and_monotonic():
    values = np.array([10.0, -2.0, 5.0, 1.0, 3.0], dtype=np.float64)

    training = build_empirical_cdf_training_data(values)

    assert np.all(training.y >= 0.0)
    assert np.all(training.y <= 1.0)
    assert np.all(np.diff(training.x[:, 0]) >= 0.0)
    assert np.all(np.diff(training.y) >= 0.0)


def test_build_empirical_cdf_training_data_uses_average_rank_for_duplicates():
    training = build_empirical_cdf_training_data([3.0, 1.0, 1.0, 2.0, 3.0])

    assert np.array_equal(training.x[:, 0], np.array([1.0, 1.0, 2.0, 3.0, 3.0]))
    assert np.allclose(training.y, np.array([0.125, 0.125, 0.5, 0.875, 0.875]))


def test_build_empirical_cdf_training_data_handles_empty_input():
    training = build_empirical_cdf_training_data([])

    assert training.x.shape == (0, 1)
    assert training.y.shape == (0,)


def test_build_empirical_cdf_training_data_handles_single_value():
    training = build_empirical_cdf_training_data([4.0])

    assert np.array_equal(training.x[:, 0], np.array([4.0]))
    assert np.array_equal(training.y, np.array([0.0]))


def test_build_empirical_cdf_training_data_handles_all_equal_values():
    training = build_empirical_cdf_training_data([2.0, 2.0, 2.0, 2.0])

    assert np.array_equal(training.x[:, 0], np.array([2.0, 2.0, 2.0, 2.0]))
    assert np.array_equal(training.y, np.array([0.5, 0.5, 0.5, 0.5]))


def test_build_empirical_cdf_training_data_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        build_empirical_cdf_training_data([1.0, np.nan])

    with pytest.raises(ValueError, match="finite"):
        build_empirical_cdf_training_data([1.0, np.inf])


def test_build_empirical_cdf_training_data_rejects_non_1d_values():
    with pytest.raises(ValueError, match="1D"):
        build_empirical_cdf_training_data(np.array([[1.0], [2.0]]))


def test_cdf_model_protocol_accepts_matching_model_shape():
    model = DummyCDFModel()

    assert isinstance(model, CDFModel)
    assert model.fit([1.0, 2.0]) is model
    assert model.fit_ms == 0.0
    assert np.array_equal(model.predict([1.0, 2.0]), np.array([0.25, 0.25]))


def test_values_to_1d_float_array_validates_and_converts_values():
    values = values_to_1d_float_array([1, 2, 3])

    assert values.dtype == np.float64
    assert np.array_equal(values, np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="finite"):
        values_to_1d_float_array([1.0, np.nan])

    with pytest.raises(ValueError, match="1D"):
        values_to_1d_float_array(np.array([[1.0], [2.0]]))


def test_values_to_model_input_returns_column_vector():
    values = values_to_model_input([1.0, 2.0, 3.0])

    assert values.shape == (3, 1)
    assert values.dtype == np.float64
    assert np.array_equal(values[:, 0], np.array([1.0, 2.0, 3.0]))


def test_clamp_cdf_predictions_bounds_rank_predictions():
    predictions = clamp_cdf_predictions([-0.5, 0.25, 1.5])

    assert predictions.dtype == np.float64
    assert np.array_equal(predictions, np.array([0.0, 0.25, 1.0]))


def test_clamp_cdf_predictions_rejects_invalid_predictions():
    with pytest.raises(ValueError, match="finite"):
        clamp_cdf_predictions([0.5, np.inf])

    with pytest.raises(ValueError, match="1D"):
        clamp_cdf_predictions(np.array([[0.1], [0.2]]))


def test_linear_cdf_model_fits_and_predicts_bounded_ranks():
    model = LinearCDFModel()

    assert isinstance(model, CDFModel)
    assert model.method_name == "linear_cdf"
    assert model.fit_ms is None
    assert model.fit([0.0, 1.0, 2.0, 3.0]) is model
    assert model.fit_ms is not None
    assert model.fit_ms >= 0.0

    predictions = model.predict([0.0, 1.5, 3.0])

    assert predictions.shape == (3,)
    assert predictions.dtype == np.float64
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_linear_cdf_model_predictions_are_deterministic_for_same_training_data():
    first = LinearCDFModel().fit([0.0, 1.0, 2.0, 3.0])
    second = LinearCDFModel().fit([0.0, 1.0, 2.0, 3.0])

    assert np.allclose(first.predict([0.25, 1.25, 2.25]), second.predict([0.25, 1.25, 2.25]))


def test_linear_cdf_model_clamps_out_of_range_predictions():
    model = LinearCDFModel().fit([0.0, 1.0])

    predictions = model.predict([-10.0, 0.5, 10.0])

    assert np.array_equal(predictions, np.array([0.0, 0.5, 1.0]))


def test_linear_cdf_model_handles_duplicate_training_values():
    model = LinearCDFModel().fit([1.0, 1.0, 2.0, 3.0, 3.0])

    predictions = model.predict([1.0, 2.0, 3.0])

    assert predictions.shape == (3,)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


def test_linear_cdf_model_handles_single_value_training_data():
    model = LinearCDFModel().fit([4.0])

    assert np.array_equal(model.predict([4.0]), np.array([0.0]))


def test_linear_cdf_model_predicts_empty_array_after_fit():
    model = LinearCDFModel().fit([0.0, 1.0])

    predictions = model.predict([])

    assert predictions.shape == (0,)
    assert predictions.dtype == np.float64


def test_linear_cdf_model_rejects_predict_before_fit():
    with pytest.raises(RuntimeError, match="fit before predict"):
        LinearCDFModel().predict([1.0])


def test_linear_cdf_model_rejects_empty_training_data():
    with pytest.raises(ValueError, match="empty values"):
        LinearCDFModel().fit([])


def test_linear_cdf_model_rejects_invalid_training_and_prediction_values():
    with pytest.raises(ValueError, match="finite"):
        LinearCDFModel().fit([1.0, np.nan])

    model = LinearCDFModel().fit([0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        model.predict([np.inf])
