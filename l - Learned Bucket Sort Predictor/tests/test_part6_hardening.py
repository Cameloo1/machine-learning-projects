import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from learned_bucket_sort.amortized_benchmark import run_amortized_benchmarks, write_amortized_json_artifact
from learned_bucket_sort.benchmark import BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD, run_benchmarks, write_json_artifact
from learned_bucket_sort.cdf_model import LinearCDFModel
from learned_bucket_sort.data import generate_data
from learned_bucket_sort.learned_sort import learned_bucket_sort
from learned_bucket_sort.part5_evidence import (
    PART5_AMORTIZED_ASSET,
    PART5_QUALITY_ASSET,
    PART5_SCALE_ASSET,
    Part5PlotPaths,
)
from learned_bucket_sort.scale_closure import run_scale_closure
from learned_bucket_sort.scenarios import generate_scenario_data, generate_scenario_dataset_files
from learned_bucket_sort.torch_mlp_cdf import TorchUnavailableError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FastFakeMLPModel:
    method_name = MLP_METHOD

    def __init__(self, config=None):
        self.config = config
        self.fit_ms = None
        self.device = None
        self._min = 0.0
        self._span = 1.0

    def fit(self, values):
        array = np.asarray(list(values), dtype=np.float64)
        self.fit_ms = 0.123
        self.device = "cpu"
        if len(array) > 0:
            self._min = float(np.min(array))
            span = float(np.max(array) - self._min)
            self._span = span if span > 0.0 else 1.0
        return self

    def predict(self, values):
        array = np.asarray(list(values), dtype=np.float64)
        return np.clip((array - self._min) / self._span, 0.0, 1.0)


def unique_test_dir(label):
    path = PROJECT_ROOT / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_generated_artifact_boundaries_are_ignored_but_keep_placeholders():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "artifacts/*" in gitignore
    assert "!artifacts/.gitkeep" in gitignore
    assert "datasets/generated/*" in gitignore
    assert "!datasets/generated/.gitkeep" in gitignore
    assert (PROJECT_ROOT / "artifacts" / ".gitkeep").exists()
    assert (PROJECT_ROOT / "datasets" / "generated" / ".gitkeep").exists()


def test_docs_reference_existing_public_commands_and_promoted_assets():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    walkthrough = (PROJECT_ROOT / "docs" / "ml-project-walkthrough.md").read_text(encoding="utf-8")

    for relative_path in [
        "scripts/generate_scenario_datasets.py",
        "scripts/run_scale_closure.py",
        "scripts/run_amortized_benchmark.py",
        "scripts/run_part5_evidence.py",
        "learned_bucket_sort/benchmark.py",
        "learned_bucket_sort/amortized_benchmark.py",
    ]:
        documented = (
            relative_path.replace("/", "\\") in readme
            or relative_path in readme
            or relative_path in walkthrough
            or Path(relative_path).name in readme
            or Path(relative_path).name in walkthrough
        )
        assert documented
        assert (PROJECT_ROOT / relative_path).exists()

    for asset in [
        "assets/benchmark-color-results.png",
        "assets/scale-closure-n250000.png",
        f"assets/{PART5_SCALE_ASSET}",
        f"assets/{PART5_QUALITY_ASSET}",
        f"assets/{PART5_AMORTIZED_ASSET}",
    ]:
        assert asset in walkthrough or Path(asset).name in readme
        assert (PROJECT_ROOT / asset).exists()


def test_benchmark_json_contract_is_stable_and_color_free(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    run = run_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=30,
        bucket_count=5,
        seed=1,
        method="all",
    )
    artifact = write_json_artifact(run.results, out_dir=unique_test_dir("part6-benchmark-json"), config={"method": "all"})

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    _assert_artifact_envelope(payload)
    assert [row["method"] for row in payload["results"]] == [BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD]
    for row in payload["results"]:
        assert {
            "distribution",
            "method",
            "n",
            "bucket_count",
            "seed",
            "fit_ms",
            "bucket_ms",
            "sort_ms",
            "total_ms",
            "correct",
            "metrics",
            "dataset_file",
            "device",
        } <= set(row)
        assert row["correct"] is True
        assert isinstance(row["metrics"]["max_bucket_size"], int)
        assert "color" not in row
        assert "highlight" not in row


def test_amortized_json_contract_preserves_reuse_timing_components(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    run = run_amortized_benchmarks(
        distribution="lognormal",
        scenario=None,
        n=30,
        bucket_count=5,
        train_seed=1,
        eval_seed=2,
        method="all",
    )
    artifact = write_amortized_json_artifact(
        run.results,
        out_dir=unique_test_dir("part6-amortized-json"),
        config={"method": "all"},
    )

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    _assert_artifact_envelope(payload)
    for row in payload["results"]:
        assert {
            "distribution",
            "method",
            "device",
            "n",
            "bucket_count",
            "train_seed",
            "eval_seed",
            "train_ms",
            "predict_ms",
            "bucket_index_ms",
            "bucket_group_ms",
            "bucket_ms",
            "sort_ms",
            "sort_path_total_ms",
            "end_to_end_total_ms",
            "correct",
            "metrics",
            "train_dataset_file",
            "eval_dataset_file",
        } <= set(row)
        assert row["correct"] is True
        assert row["bucket_ms"] == pytest.approx(row["bucket_index_ms"] + row["bucket_group_ms"])
        assert row["sort_path_total_ms"] == pytest.approx(row["predict_ms"] + row["bucket_ms"] + row["sort_ms"])
        assert row["end_to_end_total_ms"] == pytest.approx(row["train_ms"] + row["sort_path_total_ms"])


def test_scale_closure_json_contract_is_flat_and_machine_readable():
    run = run_scale_closure(
        n_values=(20,),
        seed=1,
        fixed_buckets=5,
        distributions=("uniform",),
        scenarios=("response_times",),
        bucket_strategies=("fixed",),
        dataset_dir=unique_test_dir("part6-scale-datasets"),
        out_dir=unique_test_dir("part6-scale-json"),
    )

    payload = json.loads(run.artifact_path.read_text(encoding="utf-8"))
    _assert_artifact_envelope(payload)
    for row in payload["results"]:
        assert {
            "dataset",
            "dataset_kind",
            "bucket_strategy",
            "n",
            "buckets",
            "seed",
            "method",
            "fit_ms",
            "bucket_ms",
            "sort_ms",
            "total_ms",
            "variance",
            "max_bucket",
            "empty",
            "ok",
            "dataset_file",
        } <= set(row)
        assert "metrics" not in row
        assert row["ok"] is True


def test_public_script_wrappers_run_with_tiny_isolated_inputs(monkeypatch, capsys):
    import scripts.run_amortized_benchmark as amortized_script
    import scripts.run_part5_evidence as part5_script
    import scripts.run_scale_closure as scale_script

    scale_dir = unique_test_dir("part6-scale-script")
    assert scale_script.main(
        [
            "--n",
            "20",
            "--seed",
            "1",
            "--fixed-buckets",
            "5",
            "--bucket-strategy",
            "fixed",
            "--dist",
            "uniform",
            "--scenario",
            "response_times",
            "--dataset-dir",
            str(scale_dir / "datasets"),
            "--out",
            str(scale_dir / "artifacts"),
        ]
    ) == 0

    monkeypatch.setattr("learned_bucket_sort.amortized_benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    assert amortized_script.main(
        [
            "--dist",
            "uniform",
            "--n",
            "20",
            "--buckets",
            "5",
            "--train-seed",
            "1",
            "--eval-seed",
            "2",
            "--method",
            "all",
            "--out",
            str(unique_test_dir("part6-amortized-script")),
            "--no-color",
        ]
    ) == 0

    def fake_part5_run(**kwargs):
        evidence_dir = Path(kwargs["evidence_root"]) / "20260102_030405"
        evidence_dir.mkdir(parents=True)
        manifest = evidence_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        plot_paths = Part5PlotPaths(
            generated={PART5_SCALE_ASSET: evidence_dir / PART5_SCALE_ASSET},
            promoted={PART5_SCALE_ASSET: Path(kwargs["assets_dir"]) / PART5_SCALE_ASSET},
        )
        return SimpleNamespace(
            evidence_dir=evidence_dir,
            scale_artifact=evidence_dir / "scale.json",
            benchmark_artifact=evidence_dir / "benchmark.json",
            amortized_artifact=evidence_dir / "amortized.json",
            plot_paths=plot_paths,
            manifest_path=manifest,
        )

    monkeypatch.setattr("scripts.run_part5_evidence.run_part5_evidence", fake_part5_run)
    assert part5_script.main(
        [
            "--evidence-root",
            str(unique_test_dir("part6-part5-script") / "evidence"),
            "--assets-dir",
            str(unique_test_dir("part6-part5-assets")),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "wrote artifact:" in captured.out
    assert "promoted plot:" in captured.out


def test_seeded_generators_and_linear_model_are_reproducible():
    first_distribution = generate_data("lognormal", n=50, seed=7)
    second_distribution = generate_data("lognormal", n=50, seed=7)
    first_scenario = generate_scenario_data("response_times", n=50, seed=7)
    second_scenario = generate_scenario_data("response_times", n=50, seed=7)

    assert np.array_equal(first_distribution, second_distribution)
    assert np.array_equal(first_scenario, second_scenario)

    first_model = LinearCDFModel().fit(first_distribution)
    second_model = LinearCDFModel().fit(second_distribution)

    assert np.allclose(first_model.predict(first_distribution), second_model.predict(second_distribution))


def test_timestamped_scenario_outputs_are_reproducible_without_overwrite():
    out_dir = unique_test_dir("part6-scenario-overwrite")
    first = generate_scenario_dataset_files("response_times", n=25, seed=3, out_dir=out_dir)
    second = generate_scenario_dataset_files("response_times", n=25, seed=3, out_dir=out_dir)

    assert first.data_files[0] != second.data_files[0]
    assert first.manifest_path != second.manifest_path
    assert np.array_equal(np.load(first.data_files[0]), np.load(second.data_files[0]))


def test_all_sorting_methods_keep_correctness_separate_from_speed(monkeypatch):
    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", FastFakeMLPModel)
    values = generate_data("bimodal", n=60, seed=11)

    baseline_run = run_benchmarks(distribution="bimodal", scenario=None, n=60, bucket_count=8, seed=11, method="all")
    learned_result = learned_bucket_sort(values, bucket_count=8, model=LinearCDFModel())

    assert all(result.correct for result in baseline_run.results)
    assert np.allclose(learned_result.sorted_values, np.sort(values))


def test_non_mlp_paths_do_not_require_torch(monkeypatch):
    def fail_if_mlp_is_used(config=None):
        raise AssertionError("TorchMLPCDFModel should not be constructed for linear-only benchmark")

    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", fail_if_mlp_is_used)
    run = run_benchmarks(distribution="uniform", scenario=None, n=20, bucket_count=5, seed=1, method="linear")

    assert [result.method for result in run.results] == [LINEAR_METHOD]
    assert run.results[0].device == "cpu"


def test_mlp_path_fails_clearly_when_torch_is_unavailable(monkeypatch):
    class UnavailableMLP:
        def __init__(self, config=None):
            raise TorchUnavailableError("PyTorch is not installed")

    monkeypatch.setattr("learned_bucket_sort.benchmark.TorchMLPCDFModel", UnavailableMLP)

    with pytest.raises(TorchUnavailableError, match="PyTorch is not installed"):
        run_benchmarks(distribution="uniform", scenario=None, n=20, bucket_count=5, seed=1, method="mlp")


def _assert_artifact_envelope(payload):
    assert isinstance(payload["generated_at"], str)
    assert isinstance(payload["python_version"], str)
    assert isinstance(payload["config"], dict)
    assert isinstance(payload["results"], list)
    assert payload["results"]
