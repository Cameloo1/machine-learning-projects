import numpy as np
import pytest

from learned_bucket_sort.cdf_model import CDFModel
from learned_bucket_sort.learned_sort import learned_bucket_sort
from learned_bucket_sort.torch_mlp_cdf import (
    MLPCDFTrainingData,
    TorchMLPCDFConfig,
    TorchMLPCDFModel,
    TorchMLPTensors,
    TorchUnavailableError,
    build_mlp_cdf_training_data,
    build_standard_scaler,
    is_torch_installed,
    mlp_training_data_to_torch_tensors,
    resolve_torch_device,
    values_to_scaled_mlp_input,
)


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeTorch:
    def __init__(self, cuda_available):
        self.cuda = FakeCuda(cuda_available)
        self.float32 = "float32"
        self.tensor_calls = []

    def as_tensor(self, values, dtype=None, device=None):
        self.tensor_calls.append((values, dtype, device))
        return {
            "values": np.asarray(values),
            "dtype": dtype,
            "device": device,
        }


def test_torch_mlp_config_defaults_match_part4_design():
    config = TorchMLPCDFConfig()

    assert config.hidden_units == 32
    assert config.epochs == 100
    assert config.learning_rate == 0.001
    assert config.batch_size == 1024
    assert config.seed == 11
    assert config.device == "auto"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden_units": 0}, "hidden_units"),
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"batch_size": 0}, "batch_size"),
        ({"device": "tpu"}, "device"),
    ],
)
def test_torch_mlp_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TorchMLPCDFConfig(**kwargs)


def test_standard_scaler_centers_and_scales_values():
    scaler = build_standard_scaler([1.0, 2.0, 3.0])

    transformed = scaler.transform([1.0, 2.0, 3.0])

    assert transformed.dtype == np.float64
    assert transformed.shape == (3,)
    assert np.isclose(float(np.mean(transformed)), 0.0)
    assert np.isclose(float(np.std(transformed)), 1.0)


def test_standard_scaler_handles_all_equal_values_with_stable_std():
    scaler = build_standard_scaler([5.0, 5.0, 5.0])

    assert scaler.mean == 5.0
    assert scaler.std == 1.0
    assert np.array_equal(scaler.transform([5.0, 5.0]), np.array([0.0, 0.0]))


def test_standard_scaler_rejects_invalid_values():
    with pytest.raises(ValueError, match="empty"):
        build_standard_scaler([])

    with pytest.raises(ValueError, match="finite"):
        build_standard_scaler([1.0, np.nan])

    scaler = build_standard_scaler([1.0, 2.0])
    with pytest.raises(ValueError, match="1D"):
        scaler.transform(np.array([[1.0], [2.0]]))


def test_values_to_scaled_mlp_input_returns_float32_column_vector():
    scaler = build_standard_scaler([1.0, 2.0, 3.0])

    model_input = values_to_scaled_mlp_input([1.0, 2.0, 3.0], scaler)

    assert model_input.dtype == np.float32
    assert model_input.shape == (3, 1)
    assert np.isclose(float(np.mean(model_input[:, 0])), 0.0)


def test_build_mlp_cdf_training_data_reuses_empirical_cdf_targets_and_scales_inputs():
    training = build_mlp_cdf_training_data([3.0, 1.0, 2.0])

    assert isinstance(training, MLPCDFTrainingData)
    assert training.x.dtype == np.float32
    assert training.y.dtype == np.float32
    assert training.x.shape == (3, 1)
    assert training.y.shape == (3, 1)
    assert np.isclose(float(np.mean(training.x[:, 0])), 0.0)
    assert np.isclose(float(np.std(training.x[:, 0])), 1.0)
    assert np.array_equal(training.y[:, 0], np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_build_mlp_cdf_training_data_handles_duplicates_and_all_equal_values():
    duplicate_training = build_mlp_cdf_training_data([3.0, 1.0, 1.0, 2.0, 3.0])

    assert np.allclose(duplicate_training.y[:, 0], np.array([0.125, 0.125, 0.5, 0.875, 0.875]))

    equal_training = build_mlp_cdf_training_data([5.0, 5.0, 5.0])

    assert equal_training.scaler.mean == 5.0
    assert equal_training.scaler.std == 1.0
    assert np.array_equal(equal_training.x[:, 0], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.array_equal(equal_training.y[:, 0], np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_build_mlp_cdf_training_data_rejects_invalid_values():
    with pytest.raises(ValueError, match="empty"):
        build_mlp_cdf_training_data([])

    with pytest.raises(ValueError, match="finite"):
        build_mlp_cdf_training_data([1.0, np.inf])


def test_mlp_training_data_to_torch_tensors_uses_requested_device(monkeypatch):
    fake_torch = FakeTorch(cuda_available=False)
    training = build_mlp_cdf_training_data([1.0, 2.0, 3.0])
    monkeypatch.setattr("learned_bucket_sort.torch_mlp_cdf._import_torch", lambda: fake_torch)

    tensors = mlp_training_data_to_torch_tensors(training, device="cpu")

    assert isinstance(tensors, TorchMLPTensors)
    assert tensors.x["dtype"] == "float32"
    assert tensors.y["dtype"] == "float32"
    assert tensors.x["device"] == "cpu"
    assert tensors.y["device"] == "cpu"
    assert fake_torch.tensor_calls[0][0].shape == (3, 1)
    assert fake_torch.tensor_calls[1][0].shape == (3, 1)


def test_torch_availability_probe_returns_boolean():
    assert isinstance(is_torch_installed(), bool)


def test_resolve_torch_device_cpu_does_not_require_torch():
    assert resolve_torch_device("cpu") == "cpu"


def test_resolve_torch_device_auto_uses_cuda_when_fake_torch_reports_available(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.torch_mlp_cdf._import_torch", lambda: FakeTorch(cuda_available=True))

    assert resolve_torch_device("auto") == "cuda"


def test_resolve_torch_device_auto_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.torch_mlp_cdf._import_torch", lambda: FakeTorch(cuda_available=False))

    assert resolve_torch_device("auto") == "cpu"


def test_resolve_torch_device_cuda_request_requires_cuda(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.torch_mlp_cdf._import_torch", lambda: FakeTorch(cuda_available=False))

    with pytest.raises(TorchUnavailableError, match="CUDA was requested"):
        resolve_torch_device("cuda")


def test_torch_mlp_model_implements_cdf_model_contract_shape():
    model = TorchMLPCDFModel()

    assert isinstance(model, CDFModel)
    assert model.method_name == "mlp_cdf"
    assert model.fit_ms is None
    assert model.device is None
    assert model.scaler is None
    assert model.training_data is None


def test_torch_mlp_model_fit_fails_clearly_without_torch(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.torch_mlp_cdf.is_torch_installed", lambda: False)

    model = TorchMLPCDFModel()

    with pytest.raises(TorchUnavailableError, match="PyTorch is not installed"):
        model.fit([1.0, 2.0, 3.0])


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_torch_mlp_model_fit_trains_cpu_model_and_predicts_bounded_ranks():
    model = TorchMLPCDFModel(
        TorchMLPCDFConfig(
            hidden_units=8,
            epochs=30,
            learning_rate=0.01,
            batch_size=8,
            seed=123,
            device="cpu",
            loss_history=True,
        )
    )

    fitted = model.fit(np.linspace(-2.0, 2.0, 25))

    assert fitted is model
    assert model.device == "cpu"
    assert model.fit_ms is not None
    assert model.fit_ms >= 0.0
    assert model.scaler is not None
    assert model.training_data is not None
    assert model.training_summary is not None
    assert np.isfinite(model.training_summary.final_loss)
    assert len(model.training_summary.loss_history) == 30

    predictions = model.predict([-2.0, 0.0, 2.0])

    assert predictions.shape == (3,)
    assert predictions.dtype == np.float64
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
    assert np.all(np.diff(predictions) >= 0.0)


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_torch_mlp_model_predictions_are_deterministic_for_same_seed():
    values = np.linspace(0.0, 1.0, 20)
    config = TorchMLPCDFConfig(hidden_units=8, epochs=20, learning_rate=0.01, batch_size=10, seed=99, device="cpu")

    first = TorchMLPCDFModel(config).fit(values)
    second = TorchMLPCDFModel(config).fit(values)

    assert np.allclose(first.predict(values), second.predict(values))


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_torch_mlp_model_can_drive_learned_bucket_sort_correctly_on_simple_data():
    values = np.linspace(-1.0, 1.0, 30)
    model = TorchMLPCDFModel(TorchMLPCDFConfig(hidden_units=8, epochs=25, learning_rate=0.01, batch_size=10, seed=7, device="cpu"))

    result = learned_bucket_sort(values, bucket_count=5, model=model)

    assert result.method == "mlp_cdf"
    assert np.allclose(result.sorted_values, np.sort(values))
    assert result.fit_ms >= 0.0
    assert result.bucket_ms >= 0.0
    assert result.sort_ms >= 0.0


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_torch_mlp_model_cuda_request_reports_cuda_when_available():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = TorchMLPCDFModel(
        TorchMLPCDFConfig(hidden_units=4, epochs=2, learning_rate=0.01, batch_size=4, seed=5, device="cuda")
    ).fit([0.0, 1.0, 2.0, 3.0])

    assert model.device == "cuda"


def test_torch_mlp_model_predict_before_fit_fails():
    model = TorchMLPCDFModel()

    with pytest.raises(RuntimeError, match="fit before predict"):
        model.predict([1.0])
