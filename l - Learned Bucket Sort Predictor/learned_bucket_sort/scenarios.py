"""Realistic synthetic scenario dataset profiles."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Final, Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_DATASET_DIR: Final[Path] = Path("datasets/generated")


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    description: str
    unit: str
    shape_summary: str


@dataclass(frozen=True)
class ScenarioDatasetRecord:
    scenario: str
    n: int
    seed: int
    file: str
    dtype: str
    shape: list[int]
    unit: str
    description: str


@dataclass(frozen=True)
class ScenarioGenerationResult:
    data_files: list[Path]
    manifest_path: Path


ScenarioGenerator = Callable[[np.random.Generator, int], NDArray[np.float64]]


SCENARIO_PROFILES: Final[dict[str, ScenarioProfile]] = {
    "response_times": ScenarioProfile(
        name="response_times",
        description="Request latency samples with many fast responses and rare slow outliers.",
        unit="milliseconds",
        shape_summary="fast majority, slower tail, rare large outliers",
    ),
    "income_like_values": ScenarioProfile(
        name="income_like_values",
        description="Positive values shaped like income or account-size distributions.",
        unit="dollars",
        shape_summary="many moderate values, few high values, rare very high values",
    ),
    "file_sizes": ScenarioProfile(
        name="file_sizes",
        description="File-size samples with many small files and a small number of huge files.",
        unit="bytes",
        shape_summary="tiny majority, medium minority, rare massive files",
    ),
    "transaction_amounts": ScenarioProfile(
        name="transaction_amounts",
        description="Purchase-like values with many small transactions and a long high-value tail.",
        unit="dollars",
        shape_summary="small purchases, normal purchases, rare large transactions",
    ),
    "sensor_readings": ScenarioProfile(
        name="sensor_readings",
        description="Mixed-sign sensor signal with normal noise, drift, and spike events.",
        unit="signal_units",
        shape_summary="centered noise, negative drift cluster, positive spike cluster",
    ),
}

SUPPORTED_SCENARIOS: Final[tuple[str, ...]] = tuple(SCENARIO_PROFILES)

_GENERATORS: Final[dict[str, ScenarioGenerator]] = {
    "response_times": lambda rng, n: _response_times(rng, n),
    "income_like_values": lambda rng, n: _income_like_values(rng, n),
    "file_sizes": lambda rng, n: _file_sizes(rng, n),
    "transaction_amounts": lambda rng, n: _transaction_amounts(rng, n),
    "sensor_readings": lambda rng, n: _sensor_readings(rng, n),
}


def generate_scenario_data(scenario: str, n: int, seed: int) -> NDArray[np.float64]:
    """Generate a deterministic 1D scenario dataset."""
    if n < 0:
        raise ValueError("n must be non-negative")

    scenario_key = normalize_scenario(scenario)
    rng = np.random.default_rng(seed)
    values = _GENERATORS[scenario_key](rng, n)
    return np.asarray(values, dtype=np.float64)


def generate_scenario_dataset_files(
    scenario: str,
    n: int,
    seed: int,
    out_dir: str | Path = DEFAULT_DATASET_DIR,
    timestamp: datetime | None = None,
) -> ScenarioGenerationResult:
    """Generate scenario NPY files plus a manifest JSON file."""
    if n < 0:
        raise ValueError("n must be non-negative")

    scenario_names = SUPPORTED_SCENARIOS if scenario == "all" else (normalize_scenario(scenario),)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = timestamp or datetime.now(UTC)
    stamp = generated_at.strftime("%Y%m%d_%H%M%S")
    data_files: list[Path] = []
    records: list[ScenarioDatasetRecord] = []

    for scenario_name in scenario_names:
        values = generate_scenario_data(scenario_name, n=n, seed=seed)
        data_path = _unique_path(output_dir / f"{scenario_name}_n{n}_seed{seed}_{stamp}.npy")
        np.save(data_path, values)
        data_files.append(data_path)

        profile = SCENARIO_PROFILES[scenario_name]
        records.append(
            ScenarioDatasetRecord(
                scenario=scenario_name,
                n=n,
                seed=seed,
                file=data_path.name,
                dtype=str(values.dtype),
                shape=list(values.shape),
                unit=profile.unit,
                description=profile.description,
            )
        )

    manifest_path = _unique_path(output_dir / f"manifest_scenarios_n{n}_seed{seed}_{stamp}.json")
    manifest = {
        "generated_at": generated_at.isoformat(),
        "python_version": platform.python_version(),
        "scenarios": [asdict(record) for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return ScenarioGenerationResult(data_files=data_files, manifest_path=manifest_path)


def find_latest_scenario_file(
    scenario: str,
    n: int,
    seed: int,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
) -> Path | None:
    scenario_key = normalize_scenario(scenario)
    pattern = f"{scenario_key}_n{n}_seed{seed}_*.npy"
    matches = sorted(Path(dataset_dir).glob(pattern))
    return matches[-1] if matches else None


def load_manifest_records(manifest_path: str | Path) -> list[tuple[ScenarioDatasetRecord, Path]]:
    path = Path(manifest_path)
    if "<" in str(path) or ">" in str(path):
        raise ValueError(f"manifest path contains a placeholder; replace it with a real manifest file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"manifest file does not exist: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    records = []
    for raw_record in manifest.get("scenarios", []):
        record = ScenarioDatasetRecord(
            scenario=normalize_scenario(raw_record["scenario"]),
            n=int(raw_record["n"]),
            seed=int(raw_record["seed"]),
            file=str(raw_record["file"]),
            dtype=str(raw_record["dtype"]),
            shape=[int(value) for value in raw_record["shape"]],
            unit=str(raw_record["unit"]),
            description=str(raw_record["description"]),
        )
        records.append((record, base_dir / record.file))
    return records


def normalize_scenario(scenario: str) -> str:
    scenario_key = scenario.lower()
    if scenario_key not in SCENARIO_PROFILES:
        supported = ", ".join(SUPPORTED_SCENARIOS)
        raise ValueError(f"unsupported scenario '{scenario}'; expected one of: {supported}")
    return scenario_key


def _response_times(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    fast_count = int(n * 0.88)
    slow_count = int(n * 0.10)
    outlier_count = n - fast_count - slow_count
    fast = rng.lognormal(mean=3.55, sigma=0.25, size=fast_count)
    slow = rng.lognormal(mean=5.1, sigma=0.35, size=slow_count)
    outliers = rng.lognormal(mean=7.0, sigma=0.45, size=outlier_count)
    return _shuffled(rng, fast, slow, outliers)


def _income_like_values(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    base_count = int(n * 0.92)
    high_count = int(n * 0.07)
    rare_count = n - base_count - high_count
    base = rng.lognormal(mean=10.6, sigma=0.35, size=base_count)
    high = rng.lognormal(mean=11.7, sigma=0.30, size=high_count)
    rare = rng.lognormal(mean=13.0, sigma=0.35, size=rare_count)
    return _shuffled(rng, base, high, rare)


def _file_sizes(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    tiny_count = int(n * 0.78)
    medium_count = int(n * 0.18)
    huge_count = n - tiny_count - medium_count
    tiny = rng.lognormal(mean=8.0, sigma=0.75, size=tiny_count)
    medium = rng.lognormal(mean=13.0, sigma=0.65, size=medium_count)
    huge = rng.lognormal(mean=20.0, sigma=0.75, size=huge_count)
    return _shuffled(rng, tiny, medium, huge)


def _transaction_amounts(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    small_count = int(n * 0.82)
    normal_count = int(n * 0.16)
    large_count = n - small_count - normal_count
    small = rng.gamma(shape=2.0, scale=8.0, size=small_count)
    normal = rng.gamma(shape=3.0, scale=35.0, size=normal_count)
    large = rng.lognormal(mean=6.5, sigma=0.55, size=large_count)
    return _shuffled(rng, small, normal, large)


def _sensor_readings(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    normal_count = int(n * 0.86)
    drift_count = int(n * 0.10)
    spike_count = n - normal_count - drift_count
    normal = rng.normal(loc=0.0, scale=1.0, size=normal_count)
    drift = rng.normal(loc=-7.0, scale=1.2, size=drift_count)
    spikes = rng.normal(loc=18.0, scale=3.0, size=spike_count)
    return _shuffled(rng, normal, drift, spikes)


def _shuffled(rng: np.random.Generator, *parts: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.concatenate(parts) if parts else np.array([], dtype=np.float64)
    rng.shuffle(values)
    return values


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"could not create unique file path for {path}")
