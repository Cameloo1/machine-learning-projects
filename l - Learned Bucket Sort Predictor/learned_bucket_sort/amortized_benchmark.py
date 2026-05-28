"""Train-once benchmark for reused learned CDF models."""

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
from learned_bucket_sort.benchmark import BASELINE_METHOD, LINEAR_METHOD, MLP_METHOD
from learned_bucket_sort.cdf_model import CDFModel, LinearCDFModel, values_to_1d_float_array
from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS, generate_data
from learned_bucket_sort.learned_sort import assign_learned_bucket_index
from learned_bucket_sort.metrics import BucketOccupancy, bucket_occupancy
from learned_bucket_sort.scenarios import (
    DEFAULT_DATASET_DIR,
    SUPPORTED_SCENARIOS,
    find_latest_scenario_file,
    generate_scenario_dataset_files,
    normalize_scenario,
)
from learned_bucket_sort.torch_mlp_cdf import TorchMLPCDFConfig, TorchMLPCDFModel, TorchUnavailableError


SUPPORTED_AMORTIZED_METHODS: Final[tuple[str, ...]] = ("baseline", "linear", "mlp", "all")
_HIGHLIGHT_METRICS: Final[tuple[str, ...]] = (
    "sort_path_total_ms",
    "end_to_end_total_ms",
    "variance",
    "max_bucket",
    "empty",
)
_ANSI_RESET: Final[str] = "\033[0m"
_ANSI_GREEN: Final[str] = "\033[32m"
_ANSI_RED: Final[str] = "\033[31m"
_ANSI_YELLOW: Final[str] = "\033[33m"
_ANSI_BRIGHT_GREEN: Final[str] = "\033[1;32m"
_ANSI_BRIGHT_RED: Final[str] = "\033[1;31m"


@dataclass(frozen=True)
class ReusedModelSortResult:
    """Sort result for an already-fit CDF model."""

    sorted_values: list[float]
    metrics: BucketOccupancy
    predict_ms: float
    bucket_index_ms: float
    bucket_group_ms: float
    bucket_ms: float
    sort_ms: float


@dataclass(frozen=True)
class AmortizedBenchmarkResult:
    """One amortized benchmark row with separated training and reuse costs."""

    distribution: str
    method: str
    device: str | None
    n: int
    bucket_count: int
    train_seed: int
    eval_seed: int
    train_ms: float
    predict_ms: float
    bucket_index_ms: float
    bucket_group_ms: float
    bucket_ms: float
    sort_ms: float
    sort_path_total_ms: float
    end_to_end_total_ms: float
    correct: bool
    metrics: BucketOccupancy
    train_dataset_file: str | None = None
    eval_dataset_file: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass(frozen=True)
class AmortizedBenchmarkRun:
    """Amortized results plus local scenario materialization state."""

    results: list[AmortizedBenchmarkResult]
    train_dataset_files: list[Path]
    eval_dataset_files: list[Path]
    generated_files: list[Path]
    generated_manifests: list[Path]


def run_amortized_benchmarks(
    *,
    distribution: str | None,
    scenario: str | None,
    n: int,
    bucket_count: int,
    train_seed: int = 11,
    eval_seed: int = 12,
    method: str = "all",
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    mlp_config: TorchMLPCDFConfig | None = None,
) -> AmortizedBenchmarkRun:
    """Train learned models once on one seed and sort/evaluate another seed."""
    _validate_positive_n(n)
    _validate_bucket_count(bucket_count)
    _validate_distinct_seeds(train_seed, eval_seed)
    method_key = _normalize_method(method)

    if (distribution is None) == (scenario is None):
        raise ValueError("choose exactly one of distribution or scenario")

    if distribution is not None:
        distribution_key = _normalize_distribution(distribution)
        train_values = generate_data(distribution_key, n=n, seed=train_seed)
        eval_values = generate_data(distribution_key, n=n, seed=eval_seed)
        return AmortizedBenchmarkRun(
            results=_benchmark_amortized_values_for_method(
                dataset_name=distribution_key,
                train_values=train_values,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                method=method_key,
                train_dataset_file=None,
                eval_dataset_file=None,
                mlp_config=mlp_config,
            ),
            train_dataset_files=[],
            eval_dataset_files=[],
            generated_files=[],
            generated_manifests=[],
        )

    assert scenario is not None
    scenario_name = normalize_scenario(scenario)
    train_path, train_generated_files, train_manifest = _scenario_file_for_seed(
        scenario_name,
        n=n,
        seed=train_seed,
        dataset_dir=dataset_dir,
    )
    eval_path, eval_generated_files, eval_manifest = _scenario_file_for_seed(
        scenario_name,
        n=n,
        seed=eval_seed,
        dataset_dir=dataset_dir,
    )

    train_values = _load_scenario_values(train_path)
    eval_values = _load_scenario_values(eval_path)
    return AmortizedBenchmarkRun(
        results=_benchmark_amortized_values_for_method(
            dataset_name=scenario_name,
            train_values=train_values,
            eval_values=eval_values,
            bucket_count=bucket_count,
            train_seed=train_seed,
            eval_seed=eval_seed,
            method=method_key,
            train_dataset_file=str(train_path),
            eval_dataset_file=str(eval_path),
            mlp_config=mlp_config,
        ),
        train_dataset_files=[train_path],
        eval_dataset_files=[eval_path],
        generated_files=[*train_generated_files, *eval_generated_files],
        generated_manifests=[path for path in (train_manifest, eval_manifest) if path is not None],
    )


def format_amortized_console_summary(results: Sequence[AmortizedBenchmarkResult], color: bool = False) -> str:
    """Render a compact amortized benchmark table."""
    highlights = _build_highlights(results) if color else {}
    header = (
        f"{'distribution':<20} {'method':<18} {'device':>6} {'n':>8} {'buckets':>8} "
        f"{'train_ms':>10} {'predict_ms':>11} {'index_ms':>9} {'group_ms':>9} "
        f"{'bucket_ms':>10} {'sort_ms':>10} "
        f"{'sort_path_ms':>12} {'end_to_end_ms':>13} {'variance':>12} "
        f"{'max_bucket':>11} {'empty':>7} {'ok':>4}"
    )
    rows = [header]
    for row_index, result in enumerate(results):
        rows.append(
            f"{result.distribution:<20} {result.method:<18} {(result.device or '-'):>6} "
            f"{result.n:>8} {result.bucket_count:>8} {result.train_ms:>10.3f} "
            f"{result.predict_ms:>11.3f} {result.bucket_index_ms:>9.3f} "
            f"{result.bucket_group_ms:>9.3f} {result.bucket_ms:>10.3f} {result.sort_ms:>10.3f} "
            f"{_metric_cell(result.sort_path_total_ms, row_index, 'sort_path_total_ms', highlights, 12, '.3f')} "
            f"{_metric_cell(result.end_to_end_total_ms, row_index, 'end_to_end_total_ms', highlights, 13, '.3f')} "
            f"{_metric_cell(result.metrics.variance, row_index, 'variance', highlights, 12, '.3f')} "
            f"{_metric_cell(result.metrics.max_bucket_size, row_index, 'max_bucket', highlights, 11, 'd')} "
            f"{_metric_cell(result.metrics.empty_bucket_count, row_index, 'empty', highlights, 7, 'd')} "
            f"{_ok_cell(result.correct, color)}"
        )
    return "\n".join(rows)


def write_amortized_json_artifact(
    results: Sequence[AmortizedBenchmarkResult],
    out_dir: str | Path,
    config: dict[str, object],
) -> Path:
    """Write an amortized benchmark JSON artifact."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    artifact_path = output_dir / f"amortized_benchmark_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": timestamp.isoformat(),
        "python_version": platform.python_version(),
        "config": config,
        "results": [result.to_dict() for result in results],
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run train-once amortized bucket-sort benchmarks.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--dist",
        choices=SUPPORTED_DISTRIBUTIONS,
        help="Controlled synthetic distribution to benchmark.",
    )
    selector.add_argument(
        "--scenario",
        choices=SUPPORTED_SCENARIOS,
        help="File-backed realistic scenario dataset to benchmark.",
    )
    parser.add_argument("--method", choices=SUPPORTED_AMORTIZED_METHODS, default="all", help="Method to benchmark.")
    parser.add_argument("--n", type=_positive_int, default=10_000, help="Number of values.")
    parser.add_argument("--buckets", type=_positive_int, default=100, help="Number of buckets.")
    parser.add_argument("--train-seed", type=int, default=11, help="Seed used to train learned CDF models.")
    parser.add_argument("--eval-seed", type=int, default=12, help="Seed used to evaluate sorting.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Scenario NPY directory.")
    parser.add_argument("--out", type=Path, default=None, help="Optional directory for JSON output.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color in console output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run = run_amortized_benchmarks(
            distribution=args.dist,
            scenario=args.scenario,
            n=args.n,
            bucket_count=args.buckets,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
            method=args.method,
            dataset_dir=args.dataset_dir,
        )
    except (FileNotFoundError, OSError, ValueError, TorchUnavailableError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    for path in run.generated_files:
        print(f"generated dataset: {path}")
    for path in run.generated_manifests:
        print(f"generated manifest: {path}")
    for path in run.train_dataset_files:
        print(f"train dataset file: {path}")
    for path in run.eval_dataset_files:
        print(f"eval dataset file: {path}")

    print(format_amortized_console_summary(run.results, color=should_use_color(no_color=args.no_color)))

    if args.out is not None:
        artifact_path = write_amortized_json_artifact(
            run.results,
            out_dir=args.out,
            config={
                "dist": args.dist,
                "scenario": args.scenario,
                "method": args.method,
                "n": args.n,
                "buckets": args.buckets,
                "train_seed": args.train_seed,
                "eval_seed": args.eval_seed,
                "dataset_dir": str(args.dataset_dir),
            },
        )
        print(f"wrote artifact: {artifact_path}")

    return 0


def should_use_color(no_color: bool, stream=None, environ: dict[str, str] | None = None) -> bool:
    """Return whether amortized console output should use ANSI color."""
    if no_color:
        return False

    env = os.environ if environ is None else environ
    if "NO_COLOR" in env:
        return False

    output = sys.stdout if stream is None else stream
    isatty = getattr(output, "isatty", None)
    return bool(isatty and isatty())


def _build_highlights(results: Sequence[AmortizedBenchmarkResult]) -> dict[tuple[int, str], str]:
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
    results: Sequence[AmortizedBenchmarkResult],
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


def _metric_value(result: AmortizedBenchmarkResult, metric: str) -> float:
    if metric == "sort_path_total_ms":
        return result.sort_path_total_ms
    if metric == "end_to_end_total_ms":
        return result.end_to_end_total_ms
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


def sort_with_fitted_model(
    values: Sequence[float] | np.ndarray,
    *,
    bucket_count: int,
    model: CDFModel,
) -> ReusedModelSortResult:
    """Sort values using an already-fit CDF model."""
    _validate_bucket_count(bucket_count)
    items = values_to_1d_float_array(values)
    if len(items) == 0:
        bucket_sizes = [0] * bucket_count
        return ReusedModelSortResult(
            sorted_values=[],
            metrics=bucket_occupancy(bucket_sizes),
            predict_ms=0.0,
            bucket_index_ms=0.0,
            bucket_group_ms=0.0,
            bucket_ms=0.0,
            sort_ms=0.0,
        )

    predict_start = perf_counter()
    predictions = model.predict(items)
    predict_ms = (perf_counter() - predict_start) * 1000.0
    if len(predictions) != len(items):
        raise ValueError("model must return one prediction per input value")

    bucket_index_start = perf_counter()
    bucket_indexes = bucket_indexes_from_predictions(predictions, bucket_count)
    bucket_index_ms = (perf_counter() - bucket_index_start) * 1000.0

    bucket_group_start = perf_counter()
    grouped_values, bucket_sizes = _group_values_by_bucket_indexes(items, bucket_indexes, bucket_count)
    bucket_group_ms = (perf_counter() - bucket_group_start) * 1000.0

    sort_start = perf_counter()
    sorted_values = _sort_grouped_values(grouped_values)
    sort_ms = (perf_counter() - sort_start) * 1000.0

    return ReusedModelSortResult(
        sorted_values=sorted_values,
        metrics=bucket_occupancy(bucket_sizes),
        predict_ms=predict_ms,
        bucket_index_ms=bucket_index_ms,
        bucket_group_ms=bucket_group_ms,
        bucket_ms=bucket_index_ms + bucket_group_ms,
        sort_ms=sort_ms,
    )


def sort_with_fitted_model_python_reference(
    values: Sequence[float] | np.ndarray,
    *,
    bucket_count: int,
    model: CDFModel,
) -> ReusedModelSortResult:
    """Sort with the original Python bucket loop for reference tests."""
    _validate_bucket_count(bucket_count)
    items = values_to_1d_float_array(values)
    if len(items) == 0:
        bucket_sizes = [0] * bucket_count
        return ReusedModelSortResult(
            sorted_values=[],
            metrics=bucket_occupancy(bucket_sizes),
            predict_ms=0.0,
            bucket_index_ms=0.0,
            bucket_group_ms=0.0,
            bucket_ms=0.0,
            sort_ms=0.0,
        )

    predict_start = perf_counter()
    predictions = model.predict(items)
    predict_ms = (perf_counter() - predict_start) * 1000.0
    if len(predictions) != len(items):
        raise ValueError("model must return one prediction per input value")

    bucket_group_start = perf_counter()
    buckets: list[list[float]] = [[] for _ in range(bucket_count)]
    for value, predicted_rank in zip(items, predictions, strict=True):
        index = assign_learned_bucket_index(float(predicted_rank), bucket_count)
        buckets[index].append(float(value))
    bucket_group_ms = (perf_counter() - bucket_group_start) * 1000.0

    sort_start = perf_counter()
    sorted_values: list[float] = []
    for bucket in buckets:
        sorted_values.extend(sorted(bucket))
    sort_ms = (perf_counter() - sort_start) * 1000.0

    return ReusedModelSortResult(
        sorted_values=sorted_values,
        metrics=bucket_occupancy([len(bucket) for bucket in buckets]),
        predict_ms=predict_ms,
        bucket_index_ms=0.0,
        bucket_group_ms=bucket_group_ms,
        bucket_ms=bucket_group_ms,
        sort_ms=sort_ms,
    )


def bucket_indexes_from_predictions(predictions: Sequence[float] | np.ndarray, bucket_count: int) -> np.ndarray:
    """Vectorize CDF-rank predictions into clamped bucket indexes."""
    _validate_bucket_count(bucket_count)
    ranks = np.asarray(predictions, dtype=np.float64)
    if ranks.ndim != 1:
        raise ValueError("predictions must be a 1D sequence")
    if not np.isfinite(ranks).all():
        raise ValueError("predictions must be finite")

    indexes = np.floor(bucket_count * np.clip(ranks, 0.0, 1.0)).astype(np.int64, copy=False)
    return np.clip(indexes, 0, bucket_count - 1)


def _group_values_by_bucket_indexes(
    values: np.ndarray,
    bucket_indexes: np.ndarray,
    bucket_count: int,
) -> tuple[list[np.ndarray], list[int]]:
    if len(values) != len(bucket_indexes):
        raise ValueError("bucket index count must match value count")

    bucket_sizes = np.bincount(bucket_indexes, minlength=bucket_count).astype(int).tolist()
    if len(values) == 0:
        return [], bucket_sizes

    order = np.argsort(bucket_indexes, kind="stable")
    ordered_values = values[order]
    split_points = np.cumsum(bucket_sizes)[:-1]
    return list(np.split(ordered_values, split_points)), bucket_sizes


def _sort_grouped_values(grouped_values: Sequence[np.ndarray]) -> list[float]:
    sorted_parts = [np.sort(bucket) for bucket in grouped_values if len(bucket) > 0]
    return np.concatenate(sorted_parts).astype(np.float64, copy=False).tolist() if sorted_parts else []


def _benchmark_amortized_values_for_method(
    *,
    dataset_name: str,
    train_values: np.ndarray,
    eval_values: np.ndarray,
    bucket_count: int,
    train_seed: int,
    eval_seed: int,
    method: str,
    train_dataset_file: str | None,
    eval_dataset_file: str | None,
    mlp_config: TorchMLPCDFConfig | None,
) -> list[AmortizedBenchmarkResult]:
    if method == "baseline":
        return [
            _benchmark_amortized_baseline_values(
                dataset_name=dataset_name,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                train_dataset_file=train_dataset_file,
                eval_dataset_file=eval_dataset_file,
            )
        ]
    if method == "linear":
        return [
            _benchmark_amortized_model_values(
                dataset_name=dataset_name,
                train_values=train_values,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                model=LinearCDFModel(),
                device="cpu",
                train_dataset_file=train_dataset_file,
                eval_dataset_file=eval_dataset_file,
            )
        ]
    if method == "mlp":
        model = TorchMLPCDFModel(mlp_config)
        return [
            _benchmark_amortized_model_values(
                dataset_name=dataset_name,
                train_values=train_values,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                model=model,
                device=None,
                train_dataset_file=train_dataset_file,
                eval_dataset_file=eval_dataset_file,
            )
        ]
    if method == "all":
        return [
            *_benchmark_amortized_values_for_method(
                dataset_name=dataset_name,
                train_values=train_values,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                method="baseline",
                train_dataset_file=train_dataset_file,
                eval_dataset_file=eval_dataset_file,
                mlp_config=mlp_config,
            ),
            *_benchmark_amortized_values_for_method(
                dataset_name=dataset_name,
                train_values=train_values,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                method="linear",
                train_dataset_file=train_dataset_file,
                eval_dataset_file=eval_dataset_file,
                mlp_config=mlp_config,
            ),
            *_benchmark_amortized_values_for_method(
                dataset_name=dataset_name,
                train_values=train_values,
                eval_values=eval_values,
                bucket_count=bucket_count,
                train_seed=train_seed,
                eval_seed=eval_seed,
                method="mlp",
                train_dataset_file=train_dataset_file,
                eval_dataset_file=eval_dataset_file,
                mlp_config=mlp_config,
            ),
        ]

    raise ValueError(f"unsupported method '{method}'")


def _benchmark_amortized_baseline_values(
    *,
    dataset_name: str,
    eval_values: np.ndarray,
    bucket_count: int,
    train_seed: int,
    eval_seed: int,
    train_dataset_file: str | None,
    eval_dataset_file: str | None,
) -> AmortizedBenchmarkResult:
    start = perf_counter()
    result = analytic_bucket_sort(eval_values, bucket_count=bucket_count)
    sort_ms = (perf_counter() - start) * 1000.0
    correct = np.allclose(result.sorted_values, np.sort(eval_values))

    return AmortizedBenchmarkResult(
        distribution=dataset_name,
        method=BASELINE_METHOD,
        device=None,
        n=int(len(eval_values)),
        bucket_count=bucket_count,
        train_seed=train_seed,
        eval_seed=eval_seed,
        train_ms=0.0,
        predict_ms=0.0,
        bucket_index_ms=0.0,
        bucket_group_ms=0.0,
        bucket_ms=0.0,
        sort_ms=sort_ms,
        sort_path_total_ms=sort_ms,
        end_to_end_total_ms=sort_ms,
        correct=bool(correct),
        metrics=result.metrics,
        train_dataset_file=train_dataset_file,
        eval_dataset_file=eval_dataset_file,
    )


def _benchmark_amortized_model_values(
    *,
    dataset_name: str,
    train_values: np.ndarray,
    eval_values: np.ndarray,
    bucket_count: int,
    train_seed: int,
    eval_seed: int,
    model: CDFModel,
    device: str | None,
    train_dataset_file: str | None,
    eval_dataset_file: str | None,
) -> AmortizedBenchmarkResult:
    model.fit(train_values)
    train_ms = float(model.fit_ms or 0.0)
    result = sort_with_fitted_model(eval_values, bucket_count=bucket_count, model=model)
    correct = np.allclose(result.sorted_values, np.sort(eval_values))
    sort_path_total_ms = result.predict_ms + result.bucket_ms + result.sort_ms
    resolved_device = device if device is not None else getattr(model, "device", None)

    return AmortizedBenchmarkResult(
        distribution=dataset_name,
        method=model.method_name,
        device=resolved_device,
        n=int(len(eval_values)),
        bucket_count=bucket_count,
        train_seed=train_seed,
        eval_seed=eval_seed,
        train_ms=train_ms,
        predict_ms=result.predict_ms,
        bucket_index_ms=result.bucket_index_ms,
        bucket_group_ms=result.bucket_group_ms,
        bucket_ms=result.bucket_ms,
        sort_ms=result.sort_ms,
        sort_path_total_ms=sort_path_total_ms,
        end_to_end_total_ms=train_ms + sort_path_total_ms,
        correct=bool(correct),
        metrics=result.metrics,
        train_dataset_file=train_dataset_file,
        eval_dataset_file=eval_dataset_file,
    )


def _scenario_file_for_seed(
    scenario: str,
    *,
    n: int,
    seed: int,
    dataset_dir: str | Path,
) -> tuple[Path, list[Path], Path | None]:
    dataset_path = find_latest_scenario_file(scenario, n=n, seed=seed, dataset_dir=dataset_dir)
    if dataset_path is not None:
        return dataset_path, [], None

    generation = generate_scenario_dataset_files(scenario, n=n, seed=seed, out_dir=dataset_dir)
    return generation.data_files[0], generation.data_files, generation.manifest_path


def _load_scenario_values(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"scenario dataset file does not exist: {path}")
    values = np.load(path)
    if values.ndim != 1:
        raise ValueError(f"scenario dataset must be 1D: {path}")
    return np.asarray(values, dtype=np.float64)


def _normalize_distribution(distribution: str) -> str:
    distribution_key = distribution.lower()
    if distribution_key not in SUPPORTED_DISTRIBUTIONS:
        supported = ", ".join(SUPPORTED_DISTRIBUTIONS)
        raise ValueError(f"unsupported distribution '{distribution}'; expected one of: {supported}")
    return distribution_key


def _normalize_method(method: str) -> str:
    method_key = method.lower()
    if method_key not in SUPPORTED_AMORTIZED_METHODS:
        supported = ", ".join(SUPPORTED_AMORTIZED_METHODS)
        raise ValueError(f"unsupported method '{method}'; expected one of: {supported}")
    return method_key


def _validate_positive_n(n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")


def _validate_bucket_count(bucket_count: int) -> None:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")


def _validate_distinct_seeds(train_seed: int, eval_seed: int) -> None:
    if train_seed == eval_seed:
        raise ValueError("train_seed and eval_seed must differ")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    sys.exit(main())
