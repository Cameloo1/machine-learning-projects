import numpy as np
import pytest

from learned_bucket_sort.benchmark import BASELINE_METHOD, LINEAR_METHOD, SUPPORTED_METHODS, run_benchmarks
from learned_bucket_sort.cdf_model import CDFModel, clamp_cdf_predictions
from learned_bucket_sort.learned_sort import learned_bucket_sort


class FutureMLPShapedModel:
    method_name = "future_mlp_cdf"

    def __init__(self):
        self.fit_ms = None
        self.seen_values = None
        self.device = "cpu"

    def fit(self, values):
        self.seen_values = np.asarray(values, dtype=np.float64)
        self.fit_ms = 0.0
        return self

    def predict(self, values):
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0:
            return np.array([], dtype=np.float64)
        minimum = float(np.min(self.seen_values))
        maximum = float(np.max(self.seen_values))
        if minimum == maximum:
            return np.zeros(len(array), dtype=np.float64)
        return clamp_cdf_predictions((array - minimum) / (maximum - minimum))


def test_part4_contract_keeps_existing_benchmark_method_surface_available():
    assert {"baseline", "linear", "mlp", "all"} <= set(SUPPORTED_METHODS)


def test_part4_contract_all_method_keeps_baseline_then_linear_as_controls(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", lambda config=None: FutureMLPShapedModel())

    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=50,
        bucket_count=5,
        seed=11,
        method="all",
    )

    assert [result.method for result in run.results[:2]] == [BASELINE_METHOD, LINEAR_METHOD]
    assert all(result.correct for result in run.results)


def test_part4_contract_cdf_model_interface_accepts_future_mlp_shape():
    model = FutureMLPShapedModel()

    assert isinstance(model, CDFModel)

    result = learned_bucket_sort([4.0, 1.0, 3.0, 2.0], bucket_count=4, model=model)

    assert result.method == "future_mlp_cdf"
    assert result.sorted_values == [1.0, 2.0, 3.0, 4.0]
    assert result.fit_ms == 0.0
    assert result.bucket_ms >= 0.0
    assert result.sort_ms >= 0.0


def test_part4_contract_model_prediction_count_remains_a_hard_boundary():
    class WrongLengthFutureModel(FutureMLPShapedModel):
        method_name = "wrong_length_future_mlp"

        def predict(self, values):
            return np.array([0.5], dtype=np.float64)

    with pytest.raises(ValueError, match="one prediction per input"):
        learned_bucket_sort([1.0, 2.0, 3.0], bucket_count=4, model=WrongLengthFutureModel())
