import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from learned_bucket_sort.part5_evidence import (
    PART5_AMORTIZED_ASSET,
    PART5_QUALITY_ASSET,
    PART5_SCALE_ASSET,
    Part5PlotPaths,
    amortized_breakdown_points,
    bucket_quality_points,
    ensure_plot_backend,
    generate_part5_plots,
    load_amortized_artifact,
    load_benchmark_artifact,
    load_scale_closure_artifact,
    resolve_evidence_dir,
    run_part5_evidence,
    scale_total_ms_points,
    write_part5_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def unique_test_dir(label):
    path = PROJECT_ROOT / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_artifact_loaders_accept_expected_families_and_reject_wrong_family():
    out_dir = unique_test_dir("part5-loaders")
    scale_path = _write_json(out_dir / "scale_closure.json", _scale_payload())
    benchmark_path = _write_json(out_dir / "benchmark_methods.json", _benchmark_payload())
    amortized_path = _write_json(out_dir / "amortized_benchmark.json", _amortized_payload())

    assert load_scale_closure_artifact(scale_path)["results"]
    assert load_benchmark_artifact(benchmark_path)["results"]
    assert load_amortized_artifact(amortized_path)["results"]

    with pytest.raises(ValueError, match="not a benchmark_methods artifact"):
        load_benchmark_artifact(scale_path)
    with pytest.raises(FileNotFoundError, match="scale_closure artifact not found"):
        load_scale_closure_artifact(out_dir / "missing.json")


def test_scale_total_ms_points_group_by_n_method_and_bucket_strategy():
    points = scale_total_ms_points(_scale_payload())

    fixed_baseline = [point for point in points if point.bucket_strategy == "fixed" and point.method == "analytic_baseline"]
    scaled_linear = [point for point in points if point.bucket_strategy == "scaled" and point.method == "linear_cdf"]

    assert [point.n for point in fixed_baseline] == [10, 20]
    assert fixed_baseline[0].median_total_ms == 10.0
    assert scaled_linear[1].median_total_ms == pytest.approx(7.0)


def test_bucket_quality_points_compute_max_bucket_ratio():
    points = bucket_quality_points(_benchmark_payload())

    response_mlp = next(point for point in points if point.dataset == "response_times" and point.method == "mlp_cdf")

    assert response_mlp.max_bucket_ratio == pytest.approx(0.1)


def test_amortized_breakdown_points_separate_reused_path_components():
    points = amortized_breakdown_points(_amortized_payload())

    mlp_predict = next(point for point in points if point.method == "mlp_cdf" and point.component == "predict_ms")
    linear_total = next(point for point in points if point.method == "linear_cdf" and point.component == "sort_path_total_ms")

    assert mlp_predict.median_ms == pytest.approx(8.0)
    assert linear_total.median_ms == pytest.approx(5.0)


def test_transformations_reject_failed_rows_by_default():
    payload = _benchmark_payload()
    payload["results"][0]["correct"] = False

    with pytest.raises(ValueError, match="failed benchmark rows"):
        bucket_quality_points(payload)

    assert bucket_quality_points(payload, allow_failed=True)


def test_generate_part5_plots_uses_agg_backend_and_promotes_assets():
    out_dir = unique_test_dir("part5-plots")
    scale_path = _write_json(out_dir / "scale_closure.json", _scale_payload())
    benchmark_path = _write_json(out_dir / "benchmark_methods.json", _benchmark_payload())
    amortized_path = _write_json(out_dir / "amortized_benchmark.json", _amortized_payload())

    paths = generate_part5_plots(
        scale_artifact=scale_path,
        benchmark_artifact=benchmark_path,
        amortized_artifact=amortized_path,
        output_dir=out_dir / "plots",
        assets_dir=out_dir / "assets",
    )

    assert ensure_plot_backend().lower() == "agg"
    assert set(paths.generated) == {PART5_SCALE_ASSET, PART5_QUALITY_ASSET, PART5_AMORTIZED_ASSET}
    assert all(path.exists() for path in paths.generated.values())
    assert all(path.exists() for path in paths.promoted.values())


def test_manifest_contains_artifact_plot_asset_and_device_provenance():
    out_dir = unique_test_dir("part5-manifest")
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    plot_paths = Part5PlotPaths(
        generated={PART5_SCALE_ASSET: out_dir / "plots" / PART5_SCALE_ASSET},
        promoted={PART5_SCALE_ASSET: out_dir / "assets" / PART5_SCALE_ASSET},
    )
    for path in [*plot_paths.generated.values(), *plot_paths.promoted.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    manifest_path = write_part5_manifest(
        evidence_dir=out_dir,
        generated_at=timestamp,
        scale_artifact=out_dir / "scale.json",
        benchmark_artifact=out_dir / "benchmark.json",
        amortized_artifact=out_dir / "amortized.json",
        plot_paths=plot_paths,
        generated_files=[out_dir / "dataset.npy"],
        generated_manifests=[out_dir / "manifest_scenarios.json"],
        device_request="auto",
        resolved_devices=["cpu", "cuda"],
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["generated_at"] == "2026-01-02T03:04:05+00:00"
    assert payload["config"]["normal_n"] == 5000
    assert payload["config"]["amortized_n"] == 50000
    assert payload["config"]["device_request"] == "auto"
    assert payload["config"]["resolved_devices"] == ["cpu", "cuda"]
    assert "scale_closure" in payload["artifacts"]
    assert PART5_SCALE_ASSET in payload["plots"]
    assert PART5_SCALE_ASSET in payload["promoted_assets"]


def test_resolve_evidence_dir_uses_timestamp_and_avoids_overwrite():
    out_dir = unique_test_dir("part5-dir")
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    first = resolve_evidence_dir(out_dir, timestamp)
    first.mkdir()
    second = resolve_evidence_dir(out_dir, timestamp)

    assert first.name == "20260102_030405"
    assert second.name == "20260102_030405_01"


def test_run_part5_evidence_writes_manifest_without_running_heavy_benchmarks(monkeypatch):
    out_dir = unique_test_dir("part5-runner")

    def fake_scale_closure(**kwargs):
        artifact_path = kwargs["out_dir"] / "scale_closure_20260102_030405.json"
        _write_json(artifact_path, _scale_payload())
        return SimpleNamespace(
            artifact_path=artifact_path,
            generated_files=[out_dir / "scale.npy"],
            generated_manifests=[out_dir / "scale_manifest.json"],
        )

    def fake_run_benchmarks(**kwargs):
        return SimpleNamespace(
            results=[SimpleNamespace(device="cpu"), SimpleNamespace(device="cuda")],
            generated_files=[],
            generated_manifest=None,
        )

    def fake_write_json_artifact(results, out_dir, config):
        return _write_json(Path(out_dir) / "benchmark_methods_fake.json", _benchmark_payload())

    def fake_run_amortized_benchmarks(**kwargs):
        return SimpleNamespace(
            results=[SimpleNamespace(device="cpu"), SimpleNamespace(device="cuda")],
            generated_files=[],
            generated_manifests=[],
        )

    def fake_write_amortized_json_artifact(results, out_dir, config):
        return _write_json(Path(out_dir) / "amortized_benchmark_fake.json", _amortized_payload())

    def fake_generate_plots(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        assets_dir = Path(kwargs["assets_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)
        generated = {PART5_SCALE_ASSET: output_dir / PART5_SCALE_ASSET}
        promoted = {PART5_SCALE_ASSET: assets_dir / PART5_SCALE_ASSET}
        generated[PART5_SCALE_ASSET].write_bytes(b"png")
        promoted[PART5_SCALE_ASSET].write_bytes(b"png")
        return Part5PlotPaths(generated=generated, promoted=promoted)

    monkeypatch.setattr("learned_bucket_sort.part5_evidence.run_scale_closure", fake_scale_closure)
    monkeypatch.setattr("learned_bucket_sort.part5_evidence.run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr("learned_bucket_sort.part5_evidence.write_json_artifact", fake_write_json_artifact)
    monkeypatch.setattr("learned_bucket_sort.part5_evidence.run_amortized_benchmarks", fake_run_amortized_benchmarks)
    monkeypatch.setattr("learned_bucket_sort.part5_evidence.write_amortized_json_artifact", fake_write_amortized_json_artifact)
    monkeypatch.setattr("learned_bucket_sort.part5_evidence.generate_part5_plots", fake_generate_plots)

    run = run_part5_evidence(
        evidence_root=out_dir / "evidence",
        assets_dir=out_dir / "assets",
        dataset_dir=out_dir / "datasets",
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert run.evidence_dir.name == "20260102_030405"
    assert run.manifest_path.exists()
    payload = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert payload["config"]["resolved_devices"] == ["cpu", "cuda"]
    assert payload["promoted_assets"][PART5_SCALE_ASSET]


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _scale_payload():
    rows = []
    for strategy in ("fixed", "scaled"):
        for n in (10, 20):
            rows.extend(
                [
                    {
                        "dataset": "uniform",
                        "dataset_kind": "distribution",
                        "bucket_strategy": strategy,
                        "n": n,
                        "buckets": 5,
                        "method": "analytic_baseline",
                        "total_ms": float(n),
                        "ok": True,
                    },
                    {
                        "dataset": "uniform",
                        "dataset_kind": "distribution",
                        "bucket_strategy": strategy,
                        "n": n,
                        "buckets": 5,
                        "method": "linear_cdf",
                        "total_ms": float(n) / 2.0 if strategy == "fixed" else float(n) / 3.0 + 0.333333333,
                        "ok": True,
                    },
                ]
            )
    return {"generated_at": "2026-01-02T03:04:05+00:00", "results": rows}


def _benchmark_payload():
    return {
        "generated_at": "2026-01-02T03:04:05+00:00",
        "results": [
            _benchmark_row("response_times", "analytic_baseline", 100, 60),
            _benchmark_row("response_times", "linear_cdf", 100, 30),
            _benchmark_row("response_times", "mlp_cdf", 100, 10),
            _benchmark_row("lognormal", "analytic_baseline", 100, 55),
            _benchmark_row("lognormal", "linear_cdf", 100, 25),
            _benchmark_row("lognormal", "mlp_cdf", 100, 8),
        ],
    }


def _benchmark_row(dataset, method, n, max_bucket):
    return {
        "distribution": dataset,
        "method": method,
        "n": n,
        "bucket_count": 10,
        "fit_ms": 0.0,
        "bucket_ms": 1.0,
        "sort_ms": 2.0,
        "total_ms": 3.0,
        "correct": True,
        "metrics": {
            "bucket_count": 10,
            "total_count": n,
            "max_bucket_size": max_bucket,
            "empty_bucket_count": 0,
            "variance": 1.0,
        },
    }


def _amortized_payload():
    return {
        "generated_at": "2026-01-02T03:04:05+00:00",
        "results": [
            _amortized_row("analytic_baseline", 0.0, 0.0, 12.0, 12.0),
            _amortized_row("linear_cdf", 3.0, 1.0, 1.0, 5.0),
            _amortized_row("mlp_cdf", 8.0, 2.0, 1.5, 11.5),
        ],
    }


def _amortized_row(method, predict_ms, bucket_ms, sort_ms, sort_path_total_ms):
    return {
        "distribution": "response_times",
        "method": method,
        "device": None if method == "analytic_baseline" else "cpu",
        "n": 100,
        "bucket_count": 10,
        "train_seed": 11,
        "eval_seed": 12,
        "train_ms": 0.0,
        "predict_ms": predict_ms,
        "bucket_index_ms": 0.1,
        "bucket_group_ms": max(bucket_ms - 0.1, 0.0),
        "bucket_ms": bucket_ms,
        "sort_ms": sort_ms,
        "sort_path_total_ms": sort_path_total_ms,
        "end_to_end_total_ms": sort_path_total_ms,
        "correct": True,
        "metrics": {
            "bucket_count": 10,
            "total_count": 100,
            "max_bucket_size": 10,
            "empty_bucket_count": 0,
            "variance": 1.0,
        },
    }
