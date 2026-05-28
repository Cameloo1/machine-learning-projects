"""Torch MLP CDF model design boundary.

This module defines the Part 4 MLP data and device contract without making
PyTorch a required project dependency yet. The training loop is implemented in
the next phase.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from learned_bucket_sort.cdf_model import CDFModel, build_empirical_cdf_training_data, clamp_cdf_predictions


DeviceRequest = Literal["auto", "cpu", "cuda"]


class TorchUnavailableError(RuntimeError):
    """Raised when the Torch MLP path is used without an available torch install."""


@dataclass(frozen=True)
class TorchMLPCDFConfig:
    """Configuration for the future Torch MLP CDF estimator."""

    hidden_units: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 1024
    seed: int = 11
    device: DeviceRequest = "auto"
    loss_history: bool = False

    def __post_init__(self) -> None:
        if self.hidden_units <= 0:
            raise ValueError("hidden_units must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be one of: auto, cpu, cuda")


@dataclass(frozen=True)
class StandardScaler1D:
    """Mean/std scaler for scalar model inputs."""

    mean: float
    std: float

    def transform(self, values: Iterable[float]) -> NDArray[np.float64]:
        array = np.asarray(list(values), dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("values must be a 1D sequence")
        if not np.isfinite(array).all():
            raise ValueError("values must be finite")
        return ((array - self.mean) / self.std).astype(np.float64, copy=False)


@dataclass(frozen=True)
class MLPCDFTrainingData:
    """Scaled MLP inputs and empirical CDF rank targets."""

    x: NDArray[np.float32]
    y: NDArray[np.float32]
    scaler: StandardScaler1D


@dataclass(frozen=True)
class TorchMLPTensors:
    """Torch tensor pair for the MLP training loop."""

    x: object
    y: object


@dataclass(frozen=True)
class TorchMLPTrainingSummary:
    """Training-loop summary for inspection and tests."""

    final_loss: float
    loss_history: tuple[float, ...] = field(default_factory=tuple)


def build_standard_scaler(values: Iterable[float]) -> StandardScaler1D:
    """Build a stable 1D standard scaler for future MLP inputs."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be a 1D sequence")
    if len(array) == 0:
        raise ValueError("cannot build scaler with empty values")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")

    std = float(np.std(array))
    if std == 0.0:
        std = 1.0
    return StandardScaler1D(mean=float(np.mean(array)), std=std)


def build_mlp_cdf_training_data(values: Iterable[float]) -> MLPCDFTrainingData:
    """Build scaled MLP inputs and empirical CDF targets."""
    empirical = build_empirical_cdf_training_data(values)
    if len(empirical.y) == 0:
        raise ValueError("cannot build MLP CDF training data with empty values")

    scalar_values = empirical.x[:, 0]
    scaler = build_standard_scaler(scalar_values)
    x_scaled = values_to_scaled_mlp_input(scalar_values, scaler)
    y = empirical.y.reshape(-1, 1).astype(np.float32, copy=False)
    return MLPCDFTrainingData(x=x_scaled, y=y, scaler=scaler)


def values_to_scaled_mlp_input(values: Iterable[float], scaler: StandardScaler1D) -> NDArray[np.float32]:
    """Scale scalar values into the `(n, 1)` float32 shape expected by the MLP."""
    return scaler.transform(values).reshape(-1, 1).astype(np.float32, copy=False)


def mlp_training_data_to_torch_tensors(training_data: MLPCDFTrainingData, device: str) -> TorchMLPTensors:
    """Convert MLP training arrays to Torch tensors on the requested device."""
    torch = _import_torch()
    return TorchMLPTensors(
        x=torch.as_tensor(training_data.x, dtype=torch.float32, device=device),
        y=torch.as_tensor(training_data.y, dtype=torch.float32, device=device),
    )


def is_torch_installed() -> bool:
    """Return whether the torch module can be imported in this environment."""
    return importlib.util.find_spec("torch") is not None


def resolve_torch_device(requested: DeviceRequest = "auto") -> str:
    """Resolve the requested Torch device without importing torch unless present."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")

    if requested == "cpu":
        return "cpu"

    torch = _import_torch()
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_available:
            raise TorchUnavailableError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"

    return "cuda" if cuda_available else "cpu"


class TorchMLPCDFModel:
    """Torch MLP empirical CDF estimator."""

    method_name = "mlp_cdf"

    def __init__(self, config: TorchMLPCDFConfig | None = None) -> None:
        self.config = config or TorchMLPCDFConfig()
        self.fit_ms: float | None = None
        self.device: str | None = None
        self.scaler: StandardScaler1D | None = None
        self.training_data: MLPCDFTrainingData | None = None
        self.training_summary: TorchMLPTrainingSummary | None = None
        self._network = None
        self._is_fit = False

    def fit(self, values: Iterable[float]) -> "TorchMLPCDFModel":
        """Fit the MLP CDF model."""
        _ensure_torch_available()
        torch = _import_torch()
        self.device = resolve_torch_device(self.config.device)
        self.training_data = build_mlp_cdf_training_data(values)
        self.scaler = self.training_data.scaler
        tensors = mlp_training_data_to_torch_tensors(self.training_data, device=self.device)

        _seed_torch(torch, self.config.seed, device=self.device)
        network = _build_network(torch, hidden_units=self.config.hidden_units).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        loss_fn = torch.nn.MSELoss()

        start = perf_counter()
        losses: list[float] = []
        n = int(self.training_data.x.shape[0])
        batch_size = min(self.config.batch_size, n)

        network.train()
        for _ in range(self.config.epochs):
            permutation = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            batch_count = 0

            for start_index in range(0, n, batch_size):
                batch_indexes = permutation[start_index : start_index + batch_size]
                batch_x = tensors.x.index_select(0, batch_indexes)
                batch_y = tensors.y.index_select(0, batch_indexes)

                optimizer.zero_grad(set_to_none=True)
                predictions = network(batch_x)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                batch_count += 1

            losses.append(epoch_loss / max(batch_count, 1))

        if self.device == "cuda":
            torch.cuda.synchronize()
        self.fit_ms = (perf_counter() - start) * 1000.0
        self.training_summary = TorchMLPTrainingSummary(
            final_loss=losses[-1],
            loss_history=tuple(losses) if self.config.loss_history else (),
        )
        self._network = network
        self._is_fit = True
        return self

    def predict(self, values: Iterable[float]) -> NDArray[np.float64]:
        if not self._is_fit:
            raise RuntimeError("TorchMLPCDFModel must be fit before predict")
        if self.scaler is None or self._network is None or self.device is None:
            raise RuntimeError("TorchMLPCDFModel fit state is incomplete")

        torch = _import_torch()
        model_input = values_to_scaled_mlp_input(values, self.scaler)
        if len(model_input) == 0:
            return np.empty((0,), dtype=np.float64)

        self._network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(model_input, dtype=torch.float32, device=self.device)
            predictions = self._network(tensor).detach().cpu().numpy().reshape(-1)
        return clamp_cdf_predictions(_monotonic_projection(model_input[:, 0], predictions))


def _build_network(torch, hidden_units: int):
    return torch.nn.Sequential(
        torch.nn.Linear(1, hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_units, hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_units, 1),
        torch.nn.Sigmoid(),
    )


def _seed_torch(torch, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def _monotonic_projection(values: NDArray[np.float32], predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(values) == 0:
        return np.empty((0,), dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_predictions = np.asarray(predictions, dtype=np.float64)[order]
    monotonic = np.maximum.accumulate(sorted_predictions)

    projected = np.empty_like(monotonic)
    projected[order] = monotonic
    return projected


def _ensure_torch_available() -> None:
    if not is_torch_installed():
        raise TorchUnavailableError(
            "PyTorch is not installed; complete the Part 4 install gate before using TorchMLPCDFModel"
        )


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise TorchUnavailableError(
            "PyTorch is not installed; complete the Part 4 install gate before using TorchMLPCDFModel"
        ) from exc
    return torch


assert isinstance(TorchMLPCDFModel(), CDFModel)
