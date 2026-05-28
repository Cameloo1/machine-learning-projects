import numpy as np
import pytest

from learned_bucket_sort.baseline import analytic_bucket_sort
from learned_bucket_sort.data import generate_data
from learned_bucket_sort.scenarios import (
    SCENARIO_PROFILES,
    SUPPORTED_SCENARIOS,
    find_latest_scenario_file,
    generate_scenario_data,
    generate_scenario_dataset_files,
    load_manifest_records,
)


def test_supported_scenario_names_are_stable():
    assert SUPPORTED_SCENARIOS == (
        "response_times",
        "income_like_values",
        "file_sizes",
        "transaction_amounts",
        "sensor_readings",
    )
    assert set(SCENARIO_PROFILES) == set(SUPPORTED_SCENARIOS)


@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_generate_scenario_data_is_deterministic(scenario):
    first = generate_scenario_data(scenario, n=128, seed=123)
    second = generate_scenario_data(scenario, n=128, seed=123)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_generate_scenario_data_returns_1d_finite_float_array(scenario):
    values = generate_scenario_data(scenario, n=64, seed=7)

    assert values.shape == (64,)
    assert values.dtype == np.float64
    assert np.isfinite(values).all()


@pytest.mark.parametrize(
    "scenario",
    ["response_times", "income_like_values", "file_sizes", "transaction_amounts"],
)
def test_positive_scenarios_use_positive_realistic_units(scenario):
    values = generate_scenario_data(scenario, n=256, seed=3)

    assert np.min(values) > 0.0


def test_sensor_readings_are_mixed_sign_signal_values():
    values = generate_scenario_data("sensor_readings", n=1_000, seed=3)

    assert np.min(values) < 0.0
    assert np.max(values) > 0.0


def test_generate_scenario_data_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n must be non-negative"):
        generate_scenario_data("response_times", n=-1, seed=1)

    with pytest.raises(ValueError, match="unsupported scenario"):
        generate_scenario_data("unknown", n=10, seed=1)


@pytest.mark.parametrize("scenario", SUPPORTED_SCENARIOS)
def test_scenario_occupancy_is_worse_than_uniform_baseline(scenario):
    scenario_values = generate_scenario_data(scenario, n=5_000, seed=11)
    uniform_values = generate_data("uniform", n=5_000, seed=11)

    scenario_metrics = analytic_bucket_sort(scenario_values, bucket_count=50).metrics
    uniform_metrics = analytic_bucket_sort(uniform_values, bucket_count=50).metrics

    assert scenario_metrics.variance > uniform_metrics.variance * 2


def test_generate_scenario_dataset_files_writes_npy_and_manifest():
    out_dir = _unique_test_dir("scenario-files")

    result = generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=out_dir)

    assert len(result.data_files) == 1
    assert result.data_files[0].name.startswith("response_times_n20_seed1_")
    assert result.data_files[0].suffix == ".npy"
    assert result.manifest_path.name.startswith("manifest_scenarios_n20_seed1_")
    assert result.manifest_path.suffix == ".json"

    values = np.load(result.data_files[0])
    assert values.shape == (20,)

    records = load_manifest_records(result.manifest_path)
    assert len(records) == 1
    record, path = records[0]
    assert record.scenario == "response_times"
    assert record.n == 20
    assert record.seed == 1
    assert record.dtype == "float64"
    assert record.shape == [20]
    assert path == result.data_files[0]


def test_generate_scenario_dataset_files_writes_all_scenarios():
    out_dir = _unique_test_dir("scenario-all-files")

    result = generate_scenario_dataset_files("all", n=20, seed=1, out_dir=out_dir)

    assert len(result.data_files) == len(SUPPORTED_SCENARIOS)
    assert len(load_manifest_records(result.manifest_path)) == len(SUPPORTED_SCENARIOS)


def test_generate_scenario_dataset_files_does_not_overwrite_existing_files():
    out_dir = _unique_test_dir("scenario-no-overwrite")

    first = generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=out_dir)
    second = generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=out_dir)

    assert first.data_files[0] != second.data_files[0]
    assert first.manifest_path != second.manifest_path
    assert first.data_files[0].exists()
    assert second.data_files[0].exists()


def test_find_latest_scenario_file_uses_latest_matching_name():
    out_dir = _unique_test_dir("scenario-latest-helper")
    first = generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=out_dir)
    second = generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=out_dir)

    assert find_latest_scenario_file("response_times", n=20, seed=1, dataset_dir=out_dir) == second.data_files[0]
    assert find_latest_scenario_file("response_times", n=999, seed=1, dataset_dir=out_dir) is None
    assert first.data_files[0].exists()


def _unique_test_dir(label):
    from pathlib import Path
    from uuid import uuid4

    path = Path(__file__).resolve().parents[1] / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
