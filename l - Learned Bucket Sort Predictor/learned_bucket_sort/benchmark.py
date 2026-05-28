"""Repeatable benchmark harness for baseline and learned bucket sort."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Final, Sequence

import numpy as np

from learned_bucket_sort.baseline import analytic_bucket_sort
from learned_bucket_sort.cdf_model import LinearCDFModel
from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS, generate_data
from learned_bucket_sort.learned_sort import learned_bucket_sort
from learned_bucket_sort.metrics import BucketOccupancy
from learned_bucket_sort.scenarios import (
    DEFAULT_DATASET_DIR,
    SUPPORTED_SCENARIOS,
    find_latest_scenario_file,
    generate_scenario_dataset_files,
    load_manifest_records,
    normalize_scenario,
)
from learned_bucket_sort.torch_mlp_cdf import TorchMLPCDFConfig, TorchMLPCDFModel, TorchUnavailableError


BASELINE_METHOD: Final[str] = "analytic_baseline"
LINEAR_METHOD: Final[str] = "linear_cdf"
MLP_METHOD: Final[str] = "mlp_cdf"
SUPPORTED_METHODS: Final[tuple[str, ...]] = ("baseline", "linear", "mlp", "all")
_HIGHLIGHT_METRICS: Final[tuple[str, ...]] = ("total_ms", "variance", "max_bucket", "empty")
_ANSI_RESET: Final[str] = "\033[0m"
_ANSI_GREEN: Final[str] = "\033[32m"
_ANSI_RED: Final[str] = "\033[31m"
_ANSI_YELLOW: Final[str] = "\033[33m"
_ANSI_BRIGHT_GREEN: Final[str] = "\033[1;32m"
_ANSI_BRIGHT_RED: Final[str] = "\033[1;31m"


@dataclass(frozen=True)
class BenchmarkResult:
    """A single benchmark row with explicit timing, correctness, and provenance."""

    distribution: str
    method: str
    n: int
    bucket_count: int
    seed: int
    fit_ms: float
    bucket_ms: float
    sort_ms: float
    total_ms: float
    correct: bool
    metrics: BucketOccupancy
    dataset_file: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass(frozen=True)
class BenchmarkRun:
    """Benchmark results plus generated/read dataset file state."""

    results: list[BenchmarkResult]
    dataset_files: list[Path]
    generated_files: list[Path]
    generated_manifest: Path | None = None


def run_baseline_benchmark(
    distribution: str,
    n: int,
    bucket_count: int,
    seed: int,
) -> BenchmarkResult:
    """Run one in-memory analytic-baseline distribution benchmark."""
    _validate_n(n)
    _validate_bucket_count(bucket_count)
    distribution_key = _normalize_distribution(distribution)
    values = generate_data(distribution_key, n=n, seed=seed)
    return _benchmark_baseline_values(
        dataset_name=distribution_key,
        values=values,
        bucket_count=bucket_count,
        seed=seed,
        dataset_file=None,
    )


def run_benchmarks(
    *,
    distribution: str | None,
    scenario: str | None,
    n: int,
    bucket_count: int,
    seed: int,
    method: str = "baseline",
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    manifest: str | Path | None = None,
    mlp_config: TorchMLPCDFConfig | None = None,
) -> BenchmarkRun:
    """Run one benchmark selection."""
    _validate_n(n)
    _validate_bucket_count(bucket_count)
    method_key = _normalize_method(method)

    if (distribution is None) == (scenario is None):
        raise ValueError("choose exactly one of distribution or scenario")

    if distribution is not None:
        if manifest is not None:
            raise ValueError("--manifest is only valid with --scenario all")
        distribution_key = _normalize_distribution(distribution)
        values = generate_data(distribution_key, n=n, seed=seed)
        return BenchmarkRun(
            results=_benchmark_values_for_method(
                dataset_name=distribution_key,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                method=method_key,
                dataset_file=None,
                mlp_config=mlp_config,
            ),
            dataset_files=[],
            generated_files=[],
        )

    assert scenario is not None
    scenario_key = scenario.lower()

    if scenario_key == "all":
        return _run_all_scenarios(
            n=n,
            bucket_count=bucket_count,
            seed=seed,
            method=method_key,
            dataset_dir=dataset_dir,
            manifest=manifest,
            mlp_config=mlp_config,
        )

    if manifest is not None:
        raise ValueError("--manifest is only valid with --scenario all")

    scenario_name = normalize_scenario(scenario_key)
    dataset_path = find_latest_scenario_file(
        scenario_name,
        n=n,
        seed=seed,
        dataset_dir=dataset_dir,
    )
    generated_files: list[Path] = []
    generated_manifest: Path | None = None

    if dataset_path is None:
        generation = generate_scenario_dataset_files(
            scenario_name,
            n=n,
            seed=seed,
            out_dir=dataset_dir,
        )
        generated_files = generation.data_files
        generated_manifest = generation.manifest_path
        dataset_path = generation.data_files[0]

    results = _run_scenario_file(
        scenario_name=scenario_name,
        dataset_path=dataset_path,
        bucket_count=bucket_count,
        seed=seed,
        method=method_key,
        mlp_config=mlp_config,
    )
    return BenchmarkRun(
        results=results,
        dataset_files=[dataset_path],
        generated_files=generated_files,
        generated_manifest=generated_manifest,
    )


def format_console_summary(results: Sequence[BenchmarkResult], color: bool = False) -> str:
    """Render a compact human-readable benchmark table."""
    highlights = _build_highlights(results) if color else {}
    header = (
        f"{'distribution':<20} {'method':<18} {'device':>6} {'n':>8} {'buckets':>8} "
        f"{'fit_ms':>10} {'bucket_ms':>10} {'sort_ms':>10} {'total_ms':>10} "
        f"{'variance':>12} {'max_bucket':>11} {'empty':>7} {'ok':>4}"
    )
    rows = [header]
    for row_index, result in enumerate(results):
        rows.append(
            f"{result.distribution:<20} {result.method:<18} {(result.device or '-'):>6} {result.n:>8} "
            f"{result.bucket_count:>8} {result.fit_ms:>10.3f} {result.bucket_ms:>10.3f} "
            f"{result.sort_ms:>10.3f} "
            f"{_metric_cell(result.total_ms, row_index, 'total_ms', highlights, 10, '.3f')} "
            f"{_metric_cell(result.metrics.variance, row_index, 'variance', highlights, 12, '.3f')} "
            f"{_metric_cell(result.metrics.max_bucket_size, row_index, 'max_bucket', highlights, 11, 'd')} "
            f"{_metric_cell(result.metrics.empty_bucket_count, row_index, 'empty', highlights, 7, 'd')} "
            f"{_ok_cell(result.correct, color)}"
        )
    return "\n".join(rows)


def write_json_artifact(
    results: Sequence[BenchmarkResult],
    out_dir: str | Path,
    config: dict[str, object],
) -> Path:
    """Write a benchmark JSON artifact and return its path."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    artifact_path = output_dir / f"benchmark_methods_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": timestamp.isoformat(),
        "python_version": platform.python_version(),
        "config": config,
        "results": [result.to_dict() for result in results],
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark bucket-sort methods.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--dist",
        choices=SUPPORTED_DISTRIBUTIONS,
        help="Controlled synthetic distribution to benchmark.",
    )
    selector.add_argument(
        "--scenario",
        choices=("all", *SUPPORTED_SCENARIOS),
        help="File-backed realistic scenario dataset to benchmark.",
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="baseline", help="Method to benchmark.")
    parser.add_argument("--n", type=_non_negative_int, default=10_000, help="Number of values.")
    parser.add_argument("--buckets", type=_positive_int, default=100, help="Number of buckets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Scenario NPY directory.")
    parser.add_argument("--manifest", type=Path, default=None, help="Scenario manifest for --scenario all.")
    parser.add_argument("--out", type=Path, default=None, help="Optional directory for JSON output.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color in console output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run = run_benchmarks(
            distribution=args.dist,
            scenario=args.scenario,
            n=args.n,
            bucket_count=args.buckets,
            seed=args.seed,
            method=args.method,
            dataset_dir=args.dataset_dir,
            manifest=args.manifest,
        )
    except (FileNotFoundError, OSError, ValueError, TorchUnavailableError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    for path in run.generated_files:
        print(f"generated dataset: {path}")
    if run.generated_manifest is not None:
        print(f"generated manifest: {run.generated_manifest}")
    for path in run.dataset_files:
        print(f"dataset file: {path}")

    print(format_console_summary(run.results, color=should_use_color(no_color=args.no_color)))

    if args.out is not None:
        artifact_path = write_json_artifact(
            run.results,
            out_dir=args.out,
            config={
                "dist": args.dist,
                "scenario": args.scenario,
                "method": args.method,
                "n": args.n,
                "buckets": args.buckets,
                "seed": args.seed,
                "dataset_dir": str(args.dataset_dir),
                "manifest": str(args.manifest) if args.manifest is not None else None,
            },
        )
        print(f"wrote artifact: {artifact_path}")

    return 0


def should_use_color(no_color: bool, stream=None, environ: dict[str, str] | None = None) -> bool:
    """Return whether console output should use ANSI color."""
    if no_color:
        return False

    env = os.environ if environ is None else environ
    if "NO_COLOR" in env:
        return False

    output = sys.stdout if stream is None else stream
    isatty = getattr(output, "isatty", None)
    return bool(isatty and isatty())


def _build_highlights(results: Sequence[BenchmarkResult]) -> dict[tuple[int, str], str]:
    if len(results) <= 1:
        return {}

    highlights: dict[tuple[int, str], str] = {}
    groups: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        groups.setdefault(result.distribution, []).append(index)

    for metric in _HIGHLIGHT_METRICS:
        for indexes in groups.values():
            if len(indexes) > 1:
                _apply_metric_highlights(results, indexes, metric, highlights, _ANSI_GREEN, _ANSI_RED)

        _apply_metric_highlights(
            results,
            list(range(len(results))),
            metric,
            highlights,
            _ANSI_BRIGHT_GREEN,
            _ANSI_BRIGHT_RED,
        )

    return highlights


def _apply_metric_highlights(
    results: Sequence[BenchmarkResult],
    indexes: Sequence[int],
    metric: str,
    highlights: dict[tuple[int, str], str],
    best_style: str,
    worst_style: str,
) -> None:
    values = [_metric_value(results[index], metric) for index in indexes]
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return

    if values.count(min_value) == 1:
        best_index = indexes[values.index(min_value)]
        highlights[(best_index, metric)] = best_style
    if values.count(max_value) == 1:
        worst_index = indexes[values.index(max_value)]
        highlights[(worst_index, metric)] = worst_style


def _metric_value(result: BenchmarkResult, metric: str) -> float:
    if metric == "total_ms":
        return result.total_ms
    if metric == "variance":
        return result.metrics.variance
    if metric == "max_bucket":
        return float(result.metrics.max_bucket_size)
    if metric == "empty":
        return float(result.metrics.empty_bucket_count)
    raise ValueError(f"unsupported highlight metric: {metric}")


def _metric_cell(
    value: float | int,
    row_index: int,
    metric: str,
    highlights: dict[tuple[int, str], str],
    width: int,
    format_spec: str,
) -> str:
    cell = f"{value:{width}{format_spec}}"
    style = highlights.get((row_index, metric))
    return _colorize(cell, style) if style else cell


def _ok_cell(correct: bool, color: bool) -> str:
    cell = f"{str(correct):>4}"
    return _colorize(cell, _ANSI_YELLOW) if color and not correct else cell


def _colorize(text: str, style: str) -> str:
    return f"{style}{text}{_ANSI_RESET}"


def _run_all_scenarios(
    *,
    n: int,
    bucket_count: int,
    seed: int,
    method: str,
    dataset_dir: str | Path,
    manifest: str | Path | None,
    mlp_config: TorchMLPCDFConfig | None,
) -> BenchmarkRun:
    generated_files: list[Path] = []
    generated_manifest: Path | None = None

    if manifest is None:
        generation = generate_scenario_dataset_files(
            "all",
            n=n,
            seed=seed,
            out_dir=dataset_dir,
        )
        generated_files = generation.data_files
        generated_manifest = generation.manifest_path
        manifest = generation.manifest_path

    records = load_manifest_records(manifest)
    scenario_names = [record.scenario for record, _ in records]
    if set(scenario_names) != set(SUPPORTED_SCENARIOS):
        raise ValueError("manifest for --scenario all must contain one record for each supported scenario")

    ordered_records = sorted(records, key=lambda item: SUPPORTED_SCENARIOS.index(item[0].scenario))
    results: list[BenchmarkResult] = []
    for record, path in ordered_records:
        results.extend(
            _run_scenario_file(
                scenario_name=record.scenario,
                dataset_path=path,
                bucket_count=bucket_count,
                seed=record.seed,
                method=method,
                mlp_config=mlp_config,
            )
        )

    return BenchmarkRun(
        results=results,
        dataset_files=[path for _, path in ordered_records],
        generated_files=generated_files,
        generated_manifest=generated_manifest,
    )


def _run_scenario_file(
    *,
    scenario_name: str,
    dataset_path: Path,
    bucket_count: int,
    seed: int,
    method: str,
    mlp_config: TorchMLPCDFConfig | None,
) -> list[BenchmarkResult]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"scenario dataset file does not exist: {dataset_path}")

    values = np.load(dataset_path)
    if values.ndim != 1:
        raise ValueError(f"scenario dataset must be 1D: {dataset_path}")

    return _benchmark_values_for_method(
        dataset_name=scenario_name,
        values=np.asarray(values, dtype=np.float64),
        bucket_count=bucket_count,
        seed=seed,
        method=method,
        dataset_file=str(dataset_path),
        mlp_config=mlp_config,
    )


def _benchmark_values_for_method(
    *,
    dataset_name: str,
    values: np.ndarray,
    bucket_count: int,
    seed: int,
    method: str,
    dataset_file: str | None,
    mlp_config: TorchMLPCDFConfig | None,
) -> list[BenchmarkResult]:
    if method == "baseline":
        return [
            _benchmark_baseline_values(
                dataset_name=dataset_name,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                dataset_file=dataset_file,
            )
        ]
    if method == "linear":
        return [
            _benchmark_linear_values(
                dataset_name=dataset_name,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                dataset_file=dataset_file,
            )
        ]
    if method == "mlp":
        return [
            _benchmark_mlp_values(
                dataset_name=dataset_name,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                dataset_file=dataset_file,
                mlp_config=mlp_config,
            )
        ]
    if method == "all":
        return [
            _benchmark_baseline_values(
                dataset_name=dataset_name,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                dataset_file=dataset_file,
            ),
            _benchmark_linear_values(
                dataset_name=dataset_name,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                dataset_file=dataset_file,
            ),
            _benchmark_mlp_values(
                dataset_name=dataset_name,
                values=values,
                bucket_count=bucket_count,
                seed=seed,
                dataset_file=dataset_file,
                mlp_config=mlp_config,
            ),
        ]

    raise ValueError(f"unsupported method '{method}'")


def _benchmark_baseline_values(
    *,
    dataset_name: str,
    values: np.ndarray,
    bucket_count: int,
    seed: int,
    dataset_file: str | None,
) -> BenchmarkResult:
    start = perf_counter()
    result = analytic_bucket_sort(values, bucket_count=bucket_count)
    sort_ms = (perf_counter() - start) * 1000.0
    correct = np.allclose(result.sorted_values, np.sort(values))

    return BenchmarkResult(
        distribution=dataset_name,
        method=BASELINE_METHOD,
        n=int(len(values)),
        bucket_count=bucket_count,
        seed=seed,
        fit_ms=0.0,
        bucket_ms=0.0,
        sort_ms=sort_ms,
        total_ms=sort_ms,
        correct=bool(correct),
        metrics=result.metrics,
        dataset_file=dataset_file,
    )


def _benchmark_linear_values(
    *,
    dataset_name: str,
    values: np.ndarray,
    bucket_count: int,
    seed: int,
    dataset_file: str | None,
) -> BenchmarkResult:
    result = learned_bucket_sort(values, bucket_count=bucket_count, model=LinearCDFModel())
    correct = np.allclose(result.sorted_values, np.sort(values))
    total_ms = result.fit_ms + result.bucket_ms + result.sort_ms

    return BenchmarkResult(
        distribution=dataset_name,
        method=LINEAR_METHOD,
        n=int(len(values)),
        bucket_count=bucket_count,
        seed=seed,
        fit_ms=result.fit_ms,
        bucket_ms=result.bucket_ms,
        sort_ms=result.sort_ms,
        total_ms=total_ms,
        correct=bool(correct),
        metrics=result.metrics,
        dataset_file=dataset_file,
        device="cpu",
    )


def _benchmark_mlp_values(
    *,
    dataset_name: str,
    values: np.ndarray,
    bucket_count: int,
    seed: int,
    dataset_file: str | None,
    mlp_config: TorchMLPCDFConfig | None,
) -> BenchmarkResult:
    model = TorchMLPCDFModel(mlp_config)
    result = learned_bucket_sort(values, bucket_count=bucket_count, model=model)
    correct = np.allclose(result.sorted_values, np.sort(values))
    total_ms = result.fit_ms + result.bucket_ms + result.sort_ms

    return BenchmarkResult(
        distribution=dataset_name,
        method=MLP_METHOD,
        n=int(len(values)),
        bucket_count=bucket_count,
        seed=seed,
        fit_ms=result.fit_ms,
        bucket_ms=result.bucket_ms,
        sort_ms=result.sort_ms,
        total_ms=total_ms,
        correct=bool(correct),
        metrics=result.metrics,
        dataset_file=dataset_file,
        device=model.device,
    )


def _normalize_distribution(distribution: str) -> str:
    distribution_key = distribution.lower()
    if distribution_key not in SUPPORTED_DISTRIBUTIONS:
        supported = ", ".join(SUPPORTED_DISTRIBUTIONS)
        raise ValueError(f"unsupported distribution '{distribution}'; expected one of: {supported}")
    return distribution_key


def _normalize_method(method: str) -> str:
    method_key = method.lower()
    if method_key not in SUPPORTED_METHODS:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"unsupported method '{method}'; expected one of: {supported}")
    return method_key


def _validate_n(n: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")


def _validate_bucket_count(bucket_count: int) -> None:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    sys.exit(main())
