import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from learned_bucket_sort.benchmark import (
    BASELINE_METHOD,
    LINEAR_METHOD,
    MLP_METHOD,
    build_parser,
    format_console_summary,
    main,
    run_baseline_benchmark,
    run_benchmarks,
    should_use_color,
    write_json_artifact,
)
from learned_bucket_sort.metrics import BucketOccupancy
from learned_bucket_sort.scenarios import generate_scenario_dataset_files
from learned_bucket_sort.torch_mlp_cdf import TorchMLPCDFConfig, is_torch_installed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def unique_test_dir(label):
    path = PROJECT_ROOT / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TtyStream:
    def isatty(self):
        return True


class NonTtyStream:
    def isatty(self):
        return False


class FastFakeMLPModel:
    method_name = MLP_METHOD

    def __init__(self, config=None):
        self.config = config
        self.fit_ms = None
        self.device = None

    def fit(self, values):
        self.fit_ms = 0.123
        self.device = "cpu"
        return self

    def predict(self, values):
        array = np.asarray(list(values), dtype=np.float64)
        if len(array) == 0:
            return np.empty((0,), dtype=np.float64)
        span = float(np.max(array) - np.min(array))
        if span == 0.0:
            return np.zeros(len(array), dtype=np.float64)
        return (array - float(np.min(array))) / span


def fast_mlp_config():
    return TorchMLPCDFConfig(hidden_units=4, epochs=2, learning_rate=0.01, batch_size=16, seed=1, device="cpu")


def result_row(
    distribution,
    method,
    total_ms,
    variance,
    max_bucket,
    empty,
    correct=True,
    device=None,
):
    from learned_bucket_sort.benchmark import BenchmarkResult

    return BenchmarkResult(
        distribution=distribution,
        method=method,
        n=100,
        bucket_count=10,
        seed=1,
        fit_ms=0.0,
        bucket_ms=0.0,
        sort_ms=total_ms,
        total_ms=total_ms,
        correct=correct,
        metrics=BucketOccupancy(
            bucket_count=10,
            total_count=100,
            min_bucket_size=0 if empty else 1,
            max_bucket_size=max_bucket,
            empty_bucket_count=empty,
            spread=max_bucket if empty else max_bucket - 1,
            mean_bucket_size=10.0,
            variance=variance,
        ),
        device=device,
    )


def test_run_baseline_benchmark_returns_explicit_result_shape():
    result = run_baseline_benchmark("lognormal", n=100, bucket_count=10, seed=5)

    assert result.distribution == "lognormal"
    assert result.method == BASELINE_METHOD
    assert result.n == 100
    assert result.bucket_count == 10
    assert result.seed == 5
    assert result.fit_ms == 0.0
    assert result.bucket_ms == 0.0
    assert result.sort_ms >= 0.0
    assert result.total_ms == result.sort_ms
    assert result.correct is True
    assert result.dataset_file is None
    assert result.device is None
    assert result.metrics.total_count == 100
    assert result.metrics.bucket_count == 10


def test_run_benchmarks_supports_single_distribution():
    run = run_benchmarks(
        distribution="uniform",
        scenario=None,
        n=20,
        bucket_count=5,
        seed=1,
    )

    assert len(run.results) == 1
    assert run.results[0].distribution == "uniform"
    assert run.results[0].correct is True
    assert run.dataset_files == []
    assert run.generated_files == []


def test_run_benchmarks_supports_linear_distribution_method():
    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=100,
        bucket_count=10,
        seed=1,
        method="linear",
    )

    assert len(run.results) == 1
    result = run.results[0]
    assert result.distribution == "lognormal"
    assert result.method == LINEAR_METHOD
    assert result.correct is True
    assert result.fit_ms >= 0.0
    assert result.bucket_ms >= 0.0
    assert result.sort_ms >= 0.0
    assert result.total_ms == pytest.approx(result.fit_ms + result.bucket_ms + result.sort_ms)
    assert result.device == "cpu"


@pytest.mark.skipif(not is_torch_installed(), reason="torch is not installed")
def test_run_benchmarks_supports_mlp_distribution_method():
    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=30,
        bucket_count=5,
        seed=1,
        method="mlp",
        mlp_config=fast_mlp_config(),
    )

    assert len(run.results) == 1
    result = run.results[0]
    assert result.distribution == "lognormal"
    assert result.method == MLP_METHOD
    assert result.correct is True
    assert result.fit_ms >= 0.0
    assert result.bucket_ms >= 0.0
    assert result.sort_ms >= 0.0
    assert result.total_ms == pytest.approx(result.fit_ms + result.bucket_ms + result.sort_ms)
    assert result.device == "cpu"


def test_run_benchmarks_supports_all_methods_on_same_distribution_input(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)

    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=100,
        bucket_count=10,
        seed=1,
        method="all",
    )

    assert [result.method for result in run.results] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    assert {result.distribution for result in run.results} == {"lognormal"}
    assert {result.n for result in run.results} == {100}
    assert all(result.correct for result in run.results)
    assert [result.device for result in run.results] == [None, "cpu", "cpu"]


def test_run_benchmarks_rejects_invalid_distribution():
    with pytest.raises(ValueError, match="unsupported distribution"):
        run_benchmarks(
            distribution="pareto",
            scenario=None,
            n=20,
            bucket_count=5,
            seed=1,
        )


def test_run_benchmarks_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="n must be non-negative"):
        run_benchmarks(
            distribution="uniform",
            scenario=None,
            n=-1,
            bucket_count=5,
            seed=1,
        )

    with pytest.raises(ValueError, match="bucket_count must be positive"):
        run_benchmarks(
            distribution="uniform",
            scenario=None,
            n=10,
            bucket_count=0,
            seed=1,
        )


def test_run_benchmarks_rejects_invalid_method():
    with pytest.raises(ValueError, match="unsupported method"):
        run_benchmarks(
            distribution="uniform",
            scenario=None,
            n=20,
            bucket_count=5,
            seed=1,
            method="neural",
        )


def test_run_benchmarks_rejects_missing_or_multiple_selectors():
    with pytest.raises(ValueError, match="choose exactly one"):
        run_benchmarks(
            distribution=None,
            scenario=None,
            n=20,
            bucket_count=5,
            seed=1,
        )

    with pytest.raises(ValueError, match="choose exactly one"):
        run_benchmarks(
            distribution="uniform",
            scenario="response_times",
            n=20,
            bucket_count=5,
            seed=1,
        )


def test_run_benchmarks_rejects_dist_all():
    with pytest.raises(ValueError, match="unsupported distribution"):
        run_benchmarks(
            distribution="all",
            scenario=None,
            n=20,
            bucket_count=5,
            seed=1,
        )


def test_run_benchmarks_reads_latest_matching_scenario_file():
    dataset_dir = unique_test_dir("scenario-latest")
    generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=dataset_dir)
    second = generate_scenario_dataset_files("response_times", n=20, seed=1, out_dir=dataset_dir)

    run = run_benchmarks(
        distribution=None,
        scenario="response_times",
        n=20,
        bucket_count=5,
        seed=1,
        dataset_dir=dataset_dir,
    )

    assert run.generated_files == []
    assert run.dataset_files == second.data_files
    assert run.results[0].distribution == "response_times"
    assert run.results[0].dataset_file == str(second.data_files[0])
    assert run.results[0].correct is True


def test_run_benchmarks_supports_linear_scenario_method_with_file_provenance():
    dataset_dir = unique_test_dir("scenario-linear")
    generated = generate_scenario_dataset_files("response_times", n=50, seed=1, out_dir=dataset_dir)

    run = run_benchmarks(
        distribution=None,
        scenario="response_times",
        n=50,
        bucket_count=5,
        seed=1,
        method="linear",
        dataset_dir=dataset_dir,
    )

    assert run.generated_files == []
    assert run.dataset_files == generated.data_files
    assert len(run.results) == 1
    assert run.results[0].method == LINEAR_METHOD
    assert run.results[0].dataset_file == str(generated.data_files[0])
    assert run.results[0].correct is True


def test_run_benchmarks_supports_all_methods_on_same_scenario_file(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    dataset_dir = unique_test_dir("scenario-all-methods")
    generated = generate_scenario_dataset_files("response_times", n=50, seed=1, out_dir=dataset_dir)

    run = run_benchmarks(
        distribution=None,
        scenario="response_times",
        n=50,
        bucket_count=5,
        seed=1,
        method="all",
        dataset_dir=dataset_dir,
    )

    assert [result.method for result in run.results] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    assert {result.dataset_file for result in run.results} == {str(generated.data_files[0])}
    assert [result.device for result in run.results] == [None, "cpu", "cpu"]
    assert all(result.correct for result in run.results)


def test_run_benchmarks_auto_generates_missing_single_scenario_file():
    dataset_dir = unique_test_dir("scenario-autogen")

    run = run_benchmarks(
        distribution=None,
        scenario="response_times",
        n=20,
        bucket_count=5,
        seed=1,
        dataset_dir=dataset_dir,
    )

    assert len(run.generated_files) == 1
    assert run.generated_manifest is not None
    assert run.dataset_files == run.generated_files
    assert run.results[0].dataset_file == str(run.generated_files[0])


def test_run_benchmarks_auto_generates_all_scenarios_without_manifest():
    dataset_dir = unique_test_dir("scenario-all-autogen")

    run = run_benchmarks(
        distribution=None,
        scenario="all",
        n=20,
        bucket_count=5,
        seed=1,
        dataset_dir=dataset_dir,
    )

    assert len(run.generated_files) == 5
    assert run.generated_manifest is not None
    assert len(run.dataset_files) == 5
    assert [result.distribution for result in run.results] == [
        "response_times",
        "income_like_values",
        "file_sizes",
        "transaction_amounts",
        "sensor_readings",
    ]
    assert all(result.correct for result in run.results)


def test_run_benchmarks_uses_manifest_for_all_scenarios():
    dataset_dir = unique_test_dir("scenario-all-manifest")
    generated = generate_scenario_dataset_files("all", n=20, seed=1, out_dir=dataset_dir)

    run = run_benchmarks(
        distribution=None,
        scenario="all",
        n=999,
        bucket_count=5,
        seed=999,
        dataset_dir=dataset_dir,
        manifest=generated.manifest_path,
    )

    assert run.generated_files == []
    assert run.generated_manifest is None
    assert run.dataset_files == generated.data_files
    assert {result.n for result in run.results} == {20}
    assert {result.seed for result in run.results} == {1}


def test_run_benchmarks_all_scenarios_with_all_methods_returns_three_rows_per_scenario(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    dataset_dir = unique_test_dir("scenario-all-methods-manifest")
    generated = generate_scenario_dataset_files("all", n=20, seed=1, out_dir=dataset_dir)

    run = run_benchmarks(
        distribution=None,
        scenario="all",
        n=20,
        bucket_count=5,
        seed=1,
        method="all",
        dataset_dir=dataset_dir,
        manifest=generated.manifest_path,
    )

    assert len(run.results) == 15
    assert [result.method for result in run.results[:3]] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    assert [result.device for result in run.results[:3]] == [None, "cpu", "cpu"]
    assert all(result.dataset_file is not None for result in run.results)
    assert all(result.correct for result in run.results)


def test_format_console_summary_contains_human_readable_columns():
    result = run_baseline_benchmark("uniform", n=20, bucket_count=5, seed=1)
    summary = format_console_summary([result])

    assert "distribution" in summary
    assert "analytic_baseline" in summary
    assert "device" in summary
    assert "fit_ms" in summary
    assert "total_ms" in summary
    assert "variance" in summary
    assert "True" in summary


def test_format_console_summary_renders_device_provenance():
    results = [
        result_row("a", BASELINE_METHOD, total_ms=10.0, variance=100.0, max_bucket=20, empty=5),
        result_row("a", LINEAR_METHOD, total_ms=8.0, variance=80.0, max_bucket=18, empty=3, device="cpu"),
        result_row("a", MLP_METHOD, total_ms=7.0, variance=20.0, max_bucket=8, empty=1, device="cuda"),
    ]

    summary = format_console_summary(results)
    lines = summary.splitlines()

    assert "device" in lines[0]
    assert "analytic_baseline" in lines[1]
    assert "     -" in lines[1]
    assert "linear_cdf" in lines[2]
    assert "   cpu" in lines[2]
    assert "mlp_cdf" in lines[3]
    assert "  cuda" in lines[3]


def test_format_console_summary_plain_output_has_no_ansi_escapes(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)

    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=100,
        bucket_count=10,
        seed=1,
        method="all",
    )

    summary = format_console_summary(run.results, color=False)

    assert "\033[" not in summary


def test_format_console_summary_single_row_has_no_metric_color():
    result = result_row("one", BASELINE_METHOD, total_ms=10.0, variance=100.0, max_bucket=20, empty=5)

    summary = format_console_summary([result], color=True)

    assert "\033[" not in summary


def test_format_console_summary_highlights_group_and_global_metric_extremes():
    results = [
        result_row("a", BASELINE_METHOD, total_ms=10.0, variance=100.0, max_bucket=20, empty=5),
        result_row("a", LINEAR_METHOD, total_ms=8.0, variance=80.0, max_bucket=18, empty=3),
        result_row("b", BASELINE_METHOD, total_ms=20.0, variance=300.0, max_bucket=50, empty=10),
        result_row("b", LINEAR_METHOD, total_ms=12.0, variance=200.0, max_bucket=40, empty=8),
    ]

    summary = format_console_summary(results, color=True)

    assert "\033[32m" in summary
    assert "\033[31m" in summary
    assert "\033[1;32m" in summary
    assert "\033[1;31m" in summary


def test_format_console_summary_does_not_highlight_tied_metric_values():
    results = [
        result_row("a", BASELINE_METHOD, total_ms=10.0, variance=100.0, max_bucket=20, empty=5),
        result_row("a", LINEAR_METHOD, total_ms=10.0, variance=100.0, max_bucket=20, empty=5),
    ]

    summary = format_console_summary(results, color=True)

    assert "\033[" not in summary


def test_format_console_summary_highlights_failed_correctness_only():
    result = result_row("one", BASELINE_METHOD, total_ms=10.0, variance=100.0, max_bucket=20, empty=5, correct=False)

    summary = format_console_summary([result], color=True)

    assert "\033[33m" in summary
    assert "\033[32m" not in summary
    assert "\033[31m" not in summary


def test_should_use_color_respects_tty_no_color_and_no_color_flag():
    assert should_use_color(no_color=False, stream=TtyStream(), environ={}) is True
    assert should_use_color(no_color=True, stream=TtyStream(), environ={}) is False
    assert should_use_color(no_color=False, stream=TtyStream(), environ={"NO_COLOR": ""}) is False
    assert should_use_color(no_color=False, stream=NonTtyStream(), environ={}) is False


def test_write_json_artifact_uses_result_payload():
    result = run_baseline_benchmark("uniform", n=20, bucket_count=5, seed=1)
    out_dir = unique_test_dir("json-artifact")

    artifact_path = write_json_artifact(
        [result],
        out_dir=out_dir,
        config={"dist": "uniform", "n": 20, "buckets": 5, "seed": 1},
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_path.name.startswith("benchmark_methods_")
    assert payload["config"] == {"dist": "uniform", "n": 20, "buckets": 5, "seed": 1}
    assert payload["python_version"]
    assert payload["generated_at"]
    assert payload["results"][0]["distribution"] == "uniform"
    assert payload["results"][0]["method"] == BASELINE_METHOD
    assert payload["results"][0]["correct"] is True
    assert payload["results"][0]["dataset_file"] is None
    assert payload["results"][0]["device"] is None
    assert payload["results"][0]["fit_ms"] == 0.0
    assert payload["results"][0]["bucket_ms"] == 0.0
    assert payload["results"][0]["total_ms"] == payload["results"][0]["sort_ms"]
    assert payload["results"][0]["metrics"]["total_count"] == 20
    assert "highlight" not in payload["results"][0]
    assert "color" not in payload["results"][0]


def test_json_artifact_preserves_device_for_all_methods(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=20,
        bucket_count=5,
        seed=1,
        method="all",
    )

    artifact_path = write_json_artifact(
        run.results,
        out_dir=unique_test_dir("json-device-artifact"),
        config={"dist": "lognormal", "method": "all"},
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert [row["method"] for row in payload["results"]] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    assert [row["device"] for row in payload["results"]] == [None, "cpu", "cpu"]


def test_json_artifact_includes_scenario_dataset_file():
    dataset_dir = unique_test_dir("json-scenario")
    run = run_benchmarks(
        distribution=None,
        scenario="response_times",
        n=20,
        bucket_count=5,
        seed=1,
        dataset_dir=dataset_dir,
    )
    artifact_path = write_json_artifact(
        run.results,
        out_dir=unique_test_dir("json-scenario-artifact"),
        config={"scenario": "response_times"},
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["dataset_file"] == str(run.dataset_files[0])


def test_main_prints_summary_without_artifact_when_out_is_omitted(capsys, monkeypatch):
    run_dir = unique_test_dir("no-artifact")
    monkeypatch.chdir(run_dir)

    exit_code = main(["--dist", "uniform", "--n", "20", "--buckets", "5", "--seed", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "analytic_baseline" in captured.out
    assert "fit_ms" in captured.out
    assert "\033[" not in captured.out
    assert "wrote artifact" not in captured.out
    assert list(run_dir.iterdir()) == []


def test_main_no_color_flag_disables_color_for_distribution(capsys, monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    exit_code = main(["--dist", "lognormal", "--n", "20", "--buckets", "5", "--seed", "1", "--method", "all", "--no-color"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "device" in captured.out
    assert "analytic_baseline" in captured.out
    assert "linear_cdf" in captured.out
    assert "mlp_cdf" in captured.out
    assert "\033[" not in captured.out


def test_main_no_color_flag_disables_color_for_scenario(capsys, monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    dataset_dir = unique_test_dir("main-no-color-scenario")
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    exit_code = main(
        [
            "--scenario",
            "response_times",
            "--n",
            "20",
            "--buckets",
            "5",
            "--seed",
            "1",
            "--method",
            "all",
            "--dataset-dir",
            str(dataset_dir),
            "--no-color",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "device" in captured.out
    assert "response_times" in captured.out
    assert "mlp_cdf" in captured.out
    assert "\033[" not in captured.out


def test_main_prints_generation_notice_for_missing_scenario(capsys):
    dataset_dir = unique_test_dir("main-scenario-autogen")

    exit_code = main(
        [
            "--scenario",
            "response_times",
            "--n",
            "20",
            "--buckets",
            "5",
            "--seed",
            "1",
            "--method",
            "linear",
            "--dataset-dir",
            str(dataset_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "generated dataset:" in captured.out
    assert "generated manifest:" in captured.out
    assert "dataset file:" in captured.out
    assert "response_times" in captured.out
    assert "linear_cdf" in captured.out


def test_main_writes_artifact_when_out_is_provided(capsys):
    out_dir = unique_test_dir("main-artifact") / "artifacts"

    exit_code = main(
        [
            "--dist",
            "uniform",
            "--n",
            "20",
            "--buckets",
            "5",
            "--seed",
            "1",
            "--out",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote artifact:" in captured.out
    assert len(list(out_dir.glob("benchmark_methods_*.json"))) == 1


def test_parser_rejects_invalid_cli_arguments():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(["--dist", "all"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--dist", "pareto"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--n", "-1"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--buckets", "0"])

    with pytest.raises(SystemExit):
        parser.parse_args(["--dist", "uniform", "--method", "neural"])

    parser.parse_args(["--dist", "uniform", "--no-color"])
    parser.parse_args(["--dist", "uniform", "--method", "mlp"])


def test_main_rejects_placeholder_manifest_path(capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "--scenario",
                "all",
                "--n",
                "20",
                "--buckets",
                "5",
                "--seed",
                "1",
                "--manifest",
                "datasets/generated/<manifest-file>.json",
            ]
        )

    captured = capsys.readouterr()
    assert "manifest path contains a placeholder" in captured.err
