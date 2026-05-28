import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from learned_bucket_sort.amortized_benchmark import (
    bucket_indexes_from_predictions,
    format_amortized_console_summary,
    main,
    run_amortized_benchmarks,
    should_use_color,
    sort_with_fitted_model,
    sort_with_fitted_model_python_reference,
    write_amortized_json_artifact,
)
from learned_bucket_sort.benchmark import BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD
from learned_bucket_sort.torch_mlp_cdf import TorchMLPCDFConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def unique_test_dir(label):
    path = PROJECT_ROOT / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class FastFakeMLPModel:
    method_name = MLP_METHOD

    def __init__(self, config=None):
        self.config = config
        self.fit_ms = None
        self.device = None
        self.minimum = None
        self.maximum = None

    def fit(self, values):
        array = np.asarray(list(values), dtype=np.float64)
        self.fit_ms = 0.123
        self.device = "cpu"
        self.minimum = float(np.min(array))
        self.maximum = float(np.max(array))
        return self

    def predict(self, values):
        array = np.asarray(list(values), dtype=np.float64)
        if len(array) == 0:
            return np.empty((0,), dtype=np.float64)
        span = self.maximum - self.minimum
        if span == 0.0:
            return np.zeros(len(array), dtype=np.float64)
        return np.clip((array - self.minimum) / span, 0.0, 1.0)


class TtyStream:
    def isatty(self):
        return True


class NonTtyStream:
    def isatty(self):
        return False


class WrongLengthModel(FastFakeMLPModel):
    method_name = "wrong_length"

    def predict(self, values):
        return np.array([0.5], dtype=np.float64)


def fast_mlp_config():
    return TorchMLPCDFConfig(hidden_units=4, epochs=2, learning_rate=0.01, batch_size=16, seed=1, device="cpu")


def test_sort_with_fitted_model_splits_predict_bucket_and_sort_timings():
    model = FastFakeMLPModel().fit([1.0, 2.0, 3.0, 4.0])

    result = sort_with_fitted_model([4.0, 1.0, 3.0, 2.0], bucket_count=4, model=model)

    assert result.sorted_values == [1.0, 2.0, 3.0, 4.0]
    assert result.predict_ms >= 0.0
    assert result.bucket_index_ms >= 0.0
    assert result.bucket_group_ms >= 0.0
    assert result.bucket_ms >= 0.0
    assert result.bucket_ms == pytest.approx(result.bucket_index_ms + result.bucket_group_ms)
    assert result.sort_ms >= 0.0
    assert result.metrics.total_count == 4


def test_vectorized_sort_matches_python_reference_for_same_fitted_model():
    values = [4.0, 1.0, 3.0, 2.0, 8.0, 7.0, 5.0, 6.0]
    vectorized_model = FastFakeMLPModel().fit(values)
    reference_model = FastFakeMLPModel().fit(values)

    vectorized = sort_with_fitted_model(values, bucket_count=4, model=vectorized_model)
    reference = sort_with_fitted_model_python_reference(values, bucket_count=4, model=reference_model)

    assert vectorized.sorted_values == reference.sorted_values
    assert vectorized.metrics.to_dict() == reference.metrics.to_dict()


def test_bucket_indexes_from_predictions_clamps_and_vectorizes_edges():
    indexes = bucket_indexes_from_predictions([-1.0, 0.0, 0.24, 0.25, 0.99, 1.0, 2.0], bucket_count=4)

    assert np.array_equal(indexes, np.array([0, 0, 0, 1, 3, 3, 3]))

    with pytest.raises(ValueError, match="finite"):
        bucket_indexes_from_predictions([0.0, np.nan], bucket_count=4)


def test_sort_with_fitted_model_rejects_wrong_prediction_count():
    model = WrongLengthModel().fit([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="one prediction per input"):
        sort_with_fitted_model([1.0, 2.0, 3.0], bucket_count=3, model=model)


def test_run_amortized_benchmarks_supports_all_distribution_methods(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)

    run = run_amortized_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=50,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="all",
    )

    assert [result.method for result in run.results] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    assert [result.device for result in run.results] == [None, "cpu", "cpu"]
    assert {result.train_seed for result in run.results} == {11}
    assert {result.eval_seed for result in run.results} == {12}
    assert all(result.correct for result in run.results)
    assert run.results[0].train_ms == 0.0
    assert run.results[0].predict_ms == 0.0
    assert run.results[1].train_ms >= 0.0
    assert run.results[2].train_ms == 0.123
    assert all(result.bucket_ms == pytest.approx(result.bucket_index_ms + result.bucket_group_ms) for result in run.results)
    assert all(result.sort_path_total_ms == pytest.approx(result.predict_ms + result.bucket_ms + result.sort_ms) for result in run.results)
    assert all(result.end_to_end_total_ms == pytest.approx(result.train_ms + result.sort_path_total_ms) for result in run.results)
    assert run.generated_files == []
    assert run.generated_manifests == []


def test_run_amortized_benchmarks_supports_single_scenario_and_materializes_train_eval_files():
    dataset_dir = unique_test_dir("amortized-scenario")

    run = run_amortized_benchmarks(
        distribution=None,
        scenario="response_times",
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="linear",
        dataset_dir=dataset_dir,
    )

    assert len(run.results) == 1
    assert run.results[0].method == LINEAR_METHOD
    assert run.results[0].distribution == "response_times"
    assert run.results[0].train_dataset_file == str(run.train_dataset_files[0])
    assert run.results[0].eval_dataset_file == str(run.eval_dataset_files[0])
    assert len(run.generated_files) == 2
    assert len(run.generated_manifests) == 2
    assert all(path.exists() for path in run.generated_files)
    assert all(result.correct for result in run.results)


def test_run_amortized_benchmarks_reuses_existing_scenario_files():
    dataset_dir = unique_test_dir("amortized-scenario-reuse")
    first = run_amortized_benchmarks(
        distribution=None,
        scenario="response_times",
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="baseline",
        dataset_dir=dataset_dir,
    )

    second = run_amortized_benchmarks(
        distribution=None,
        scenario="response_times",
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="baseline",
        dataset_dir=dataset_dir,
    )

    assert len(first.generated_files) == 2
    assert second.generated_files == []
    assert second.train_dataset_files == first.train_dataset_files
    assert second.eval_dataset_files == first.eval_dataset_files


def test_run_amortized_benchmarks_rejects_invalid_selection_and_seed_boundary():
    with pytest.raises(ValueError, match="choose exactly one"):
        run_amortized_benchmarks(
            distribution=None,
            scenario=None,
            n=20,
            bucket_count=5,
            train_seed=11,
            eval_seed=12,
        )

    with pytest.raises(ValueError, match="must differ"):
        run_amortized_benchmarks(
            distribution="uniform",
            scenario=None,
            n=20,
            bucket_count=5,
            train_seed=11,
            eval_seed=11,
        )


def test_format_amortized_console_summary_contains_reuse_cost_columns(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    run = run_amortized_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="all",
    )

    summary = format_amortized_console_summary(run.results)

    assert "train_ms" in summary
    assert "predict_ms" in summary
    assert "index_ms" in summary
    assert "group_ms" in summary
    assert "sort_path_ms" in summary
    assert "end_to_end_ms" in summary
    assert "analytic_baseline" in summary
    assert "linear_cdf" in summary
    assert "mlp_cdf" in summary


def test_format_amortized_console_summary_highlights_reuse_metrics(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    first = run_amortized_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="all",
    )
    second = run_amortized_benchmarks(
        distribution="bimodal",
        scenario=None,
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="all",
    )

    summary = format_amortized_console_summary([*first.results, *second.results], color=True)

    assert "\033[32m" in summary
    assert "\033[31m" in summary
    assert "\033[1;32m" in summary
    assert "\033[1;31m" in summary


def test_format_amortized_console_summary_plain_output_has_no_ansi_escapes(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    run = run_amortized_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="all",
    )

    summary = format_amortized_console_summary(run.results, color=False)

    assert "\033[" not in summary


def test_should_use_color_respects_tty_no_color_and_no_color_flag():
    assert should_use_color(no_color=False, stream=TtyStream(), environ={}) is True
    assert should_use_color(no_color=True, stream=TtyStream(), environ={}) is False
    assert should_use_color(no_color=False, stream=TtyStream(), environ={"NO_COLOR": ""}) is False
    assert should_use_color(no_color=False, stream=NonTtyStream(), environ={}) is False


def test_write_amortized_json_artifact_preserves_result_shape(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    run = run_amortized_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=20,
        bucket_count=5,
        train_seed=11,
        eval_seed=12,
        method="all",
    )

    artifact_path = write_amortized_json_artifact(
        run.results,
        out_dir=unique_test_dir("amortized-json"),
        config={"dist": "lognormal", "method": "all"},
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_path.name.startswith("amortized_benchmark_")
    assert payload["config"] == {"dist": "lognormal", "method": "all"}
    assert [row["method"] for row in payload["results"]] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    assert [row["device"] for row in payload["results"]] == [None, "cpu", "cpu"]
    assert {
        "train_ms",
        "predict_ms",
        "bucket_index_ms",
        "bucket_group_ms",
        "sort_path_total_ms",
        "end_to_end_total_ms",
    } <= set(payload["results"][0])
    assert "metrics" in payload["results"][0]


def test_main_prints_summary_and_writes_artifact(capsys, monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    out_dir = unique_test_dir("amortized-main-artifact")

    exit_code = main(
        [
            "--dist",
            "lognormal",
            "--n",
            "20",
            "--buckets",
            "5",
            "--train-seed",
            "11",
            "--eval-seed",
            "12",
            "--method",
            "all",
            "--no-color",
            "--out",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "sort_path_ms" in captured.out
    assert "mlp_cdf" in captured.out
    assert "\033[" not in captured.out
    assert "wrote artifact:" in captured.out
    assert len(list(out_dir.glob("amortized_benchmark_*.json"))) == 1


def test_main_prints_scenario_file_provenance(capsys):
    dataset_dir = unique_test_dir("amortized-main-scenario")

    exit_code = main(
        [
            "--scenario",
            "response_times",
            "--n",
            "20",
            "--buckets",
            "5",
            "--train-seed",
            "11",
            "--eval-seed",
            "12",
            "--method",
            "linear",
            "--dataset-dir",
            str(dataset_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "generated dataset:" in captured.out
    assert "train dataset file:" in captured.out
    assert "eval dataset file:" in captured.out
    assert "linear_cdf" in captured.out
