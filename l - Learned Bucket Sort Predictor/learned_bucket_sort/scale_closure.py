"""Part 3.8 scale-closure benchmark runner."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Sequence

from learned_bucket_sort.benchmark import BenchmarkResult, run_benchmarks
from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS
from learned_bucket_sort.scenarios import DEFAULT_DATASET_DIR, SUPPORTED_SCENARIOS


DEFAULT_SCALE_NS: Final[tuple[int, ...]] = (5_000, 50_000, 100_000)
STRETCH_SCALE_N: Final[int] = 250_000
DEFAULT_FIXED_BUCKETS: Final[int] = 50
DEFAULT_SCALED_ITEMS_PER_BUCKET: Final[int] = 1_000
DEFAULT_MIN_SCALED_BUCKETS: Final[int] = 50
FIXED_BUCKET_STRATEGY: Final[str] = "fixed"
SCALED_BUCKET_STRATEGY: Final[str] = "scaled"
DEFAULT_BUCKET_STRATEGIES: Final[tuple[str, ...]] = (FIXED_BUCKET_STRATEGY, SCALED_BUCKET_STRATEGY)
PART3_COMPARISON_METHODS: Final[tuple[str, ...]] = ("baseline", "linear")


@dataclass(frozen=True)
class ScaleClosureRow:
    """Flattened benchmark row for scale-closure artifacts."""

    dataset: str
    dataset_kind: str
    bucket_strategy: str
    n: int
    buckets: int
    seed: int
    method: str
    fit_ms: float
    bucket_ms: float
    sort_ms: float
    total_ms: float
    variance: float
    max_bucket: int
    empty: int
    ok: bool
    dataset_file: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleClosureRun:
    """Scale-closure rows plus generated local artifact state."""

    rows: list[ScaleClosureRow]
    generated_files: list[Path]
    generated_manifests: list[Path]
    artifact_path: Path | None


def scaled_bucket_count(
    n: int,
    *,
    items_per_bucket: int = DEFAULT_SCALED_ITEMS_PER_BUCKET,
    minimum: int = DEFAULT_MIN_SCALED_BUCKETS,
) -> int:
    """Return the scaled bucket count for a dataset size."""
    if n <= 0:
        raise ValueError("n must be positive")
    if items_per_bucket <= 0:
        raise ValueError("items_per_bucket must be positive")
    if minimum <= 0:
        raise ValueError("minimum must be positive")

    return max(minimum, n // items_per_bucket)


def bucket_count_for_strategy(
    strategy: str,
    *,
    n: int,
    fixed_buckets: int = DEFAULT_FIXED_BUCKETS,
    scaled_items_per_bucket: int = DEFAULT_SCALED_ITEMS_PER_BUCKET,
    min_scaled_buckets: int = DEFAULT_MIN_SCALED_BUCKETS,
) -> int:
    """Resolve a bucket strategy into a concrete bucket count."""
    strategy_key = strategy.lower()
    if fixed_buckets <= 0:
        raise ValueError("fixed_buckets must be positive")

    if strategy_key == FIXED_BUCKET_STRATEGY:
        return fixed_buckets
    if strategy_key == SCALED_BUCKET_STRATEGY:
        return scaled_bucket_count(
            n,
            items_per_bucket=scaled_items_per_bucket,
            minimum=min_scaled_buckets,
        )

    raise ValueError(f"unsupported bucket strategy '{strategy}'")


def run_scale_closure(
    *,
    n_values: Sequence[int] = DEFAULT_SCALE_NS,
    seed: int = 11,
    fixed_buckets: int = DEFAULT_FIXED_BUCKETS,
    scaled_items_per_bucket: int = DEFAULT_SCALED_ITEMS_PER_BUCKET,
    min_scaled_buckets: int = DEFAULT_MIN_SCALED_BUCKETS,
    distributions: Sequence[str] = SUPPORTED_DISTRIBUTIONS,
    scenarios: Sequence[str] = SUPPORTED_SCENARIOS,
    bucket_strategies: Sequence[str] = DEFAULT_BUCKET_STRATEGIES,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    out_dir: str | Path = "artifacts",
    write_artifact: bool = True,
    timestamp: datetime | None = None,
) -> ScaleClosureRun:
    """Run Part 3.8 scale closure across distributions and scenario datasets."""
    normalized_n_values = _validate_n_values(n_values)
    normalized_distributions = _normalize_names(distributions, SUPPORTED_DISTRIBUTIONS, "distribution")
    normalized_scenarios = _normalize_names(scenarios, SUPPORTED_SCENARIOS, "scenario")
    normalized_strategies = _normalize_bucket_strategies(bucket_strategies)
    if not normalized_distributions and not normalized_scenarios:
        raise ValueError("at least one distribution or scenario must be selected")

    rows: list[ScaleClosureRow] = []
    generated_files: list[Path] = []
    generated_manifests: list[Path] = []

    for n in normalized_n_values:
        for strategy in normalized_strategies:
            bucket_count = bucket_count_for_strategy(
                strategy,
                n=n,
                fixed_buckets=fixed_buckets,
                scaled_items_per_bucket=scaled_items_per_bucket,
                min_scaled_buckets=min_scaled_buckets,
            )

            for distribution in normalized_distributions:
                results, _, _ = _run_part3_comparison_methods(
                    distribution=distribution,
                    scenario=None,
                    n=n,
                    bucket_count=bucket_count,
                    seed=seed,
                    dataset_dir=dataset_dir,
                )
                rows.extend(_rows_from_results(results, dataset_kind="distribution", bucket_strategy=strategy))

            for scenario in normalized_scenarios:
                results, scenario_generated_files, scenario_generated_manifest = _run_part3_comparison_methods(
                    distribution=None,
                    scenario=scenario,
                    n=n,
                    bucket_count=bucket_count,
                    seed=seed,
                    dataset_dir=dataset_dir,
                )
                generated_files.extend(scenario_generated_files)
                generated_manifests.extend(scenario_generated_manifest)
                rows.extend(_rows_from_results(results, dataset_kind="scenario", bucket_strategy=strategy))

    artifact_path = None
    if write_artifact:
        artifact_path = write_scale_closure_artifact(
            rows,
            out_dir=out_dir,
            config={
                "n_values": list(normalized_n_values),
                "seed": seed,
                "fixed_buckets": fixed_buckets,
                "scaled_items_per_bucket": scaled_items_per_bucket,
                "min_scaled_buckets": min_scaled_buckets,
                "distributions": list(normalized_distributions),
                "scenarios": list(normalized_scenarios),
                "bucket_strategies": list(normalized_strategies),
                "dataset_dir": str(dataset_dir),
            },
            timestamp=timestamp,
        )

    return ScaleClosureRun(
        rows=rows,
        generated_files=generated_files,
        generated_manifests=generated_manifests,
        artifact_path=artifact_path,
    )


def write_scale_closure_artifact(
    rows: Sequence[ScaleClosureRow],
    *,
    out_dir: str | Path,
    config: dict[str, object],
    timestamp: datetime | None = None,
) -> Path:
    """Write a scale-closure JSON artifact."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = timestamp or datetime.now(UTC)
    artifact_path = output_dir / f"scale_closure_{generated_at.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": generated_at.isoformat(),
        "python_version": platform.python_version(),
        "config": config,
        "results": [row.to_dict() for row in rows],
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_path


def format_scale_closure_summary(rows: Sequence[ScaleClosureRow]) -> str:
    """Render a compact scale-closure table."""
    header = (
        f"{'kind':<12} {'dataset':<20} {'strategy':<8} {'n':>8} {'buckets':>8} "
        f"{'method':<18} {'sort_ms':>10} {'total_ms':>10} {'variance':>12} "
        f"{'max_bucket':>11} {'empty':>7} {'ok':>4}"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{row.dataset_kind:<12} {row.dataset:<20} {row.bucket_strategy:<8} "
            f"{row.n:>8} {row.buckets:>8} {row.method:<18} "
            f"{row.sort_ms:>10.3f} {row.total_ms:>10.3f} {row.variance:>12.3f} "
            f"{row.max_bucket:>11} {row.empty:>7} {str(row.ok):>4}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Part 3.8 scale-closure benchmarks.")
    parser.add_argument(
        "--n",
        nargs="+",
        type=_positive_int,
        default=list(DEFAULT_SCALE_NS),
        help="Dataset sizes to benchmark.",
    )
    parser.add_argument(
        "--include-stretch",
        action="store_true",
        help=f"Also include n={STRETCH_SCALE_N}.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    parser.add_argument("--fixed-buckets", type=_positive_int, default=DEFAULT_FIXED_BUCKETS)
    parser.add_argument("--scaled-items-per-bucket", type=_positive_int, default=DEFAULT_SCALED_ITEMS_PER_BUCKET)
    parser.add_argument("--min-scaled-buckets", type=_positive_int, default=DEFAULT_MIN_SCALED_BUCKETS)
    parser.add_argument(
        "--dist",
        nargs="+",
        choices=("all", *SUPPORTED_DISTRIBUTIONS),
        default=["all"],
        help="Controlled distributions to benchmark.",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=("all", *SUPPORTED_SCENARIOS),
        default=["all"],
        help="Realistic scenarios to benchmark.",
    )
    parser.add_argument(
        "--bucket-strategy",
        choices=("both", FIXED_BUCKET_STRATEGY, SCALED_BUCKET_STRATEGY),
        default="both",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-artifact", action="store_true", help="Print results without writing JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    n_values = tuple(args.n)
    if args.include_stretch and STRETCH_SCALE_N not in n_values:
        n_values = (*n_values, STRETCH_SCALE_N)

    try:
        run = run_scale_closure(
            n_values=n_values,
            seed=args.seed,
            fixed_buckets=args.fixed_buckets,
            scaled_items_per_bucket=args.scaled_items_per_bucket,
            min_scaled_buckets=args.min_scaled_buckets,
            distributions=_expand_all_selection(args.dist, SUPPORTED_DISTRIBUTIONS, "distribution"),
            scenarios=_expand_all_selection(args.scenario, SUPPORTED_SCENARIOS, "scenario"),
            bucket_strategies=_strategies_from_cli(args.bucket_strategy),
            dataset_dir=args.dataset_dir,
            out_dir=args.out,
            write_artifact=not args.no_artifact,
        )
    except ValueError as exc:
        parser.error(str(exc))

    for path in run.generated_files:
        print(f"generated dataset: {path}")
    for path in run.generated_manifests:
        print(f"generated manifest: {path}")
    print(format_scale_closure_summary(run.rows))
    if run.artifact_path is not None:
        print(f"wrote artifact: {run.artifact_path}")

    return 0


def _run_part3_comparison_methods(
    *,
    distribution: str | None,
    scenario: str | None,
    n: int,
    bucket_count: int,
    seed: int,
    dataset_dir: str | Path,
) -> tuple[list[BenchmarkResult], list[Path], list[Path]]:
    results: list[BenchmarkResult] = []
    generated_files: list[Path] = []
    generated_manifests: list[Path] = []

    for method in PART3_COMPARISON_METHODS:
        run = run_benchmarks(
            distribution=distribution,
            scenario=scenario,
            n=n,
            bucket_count=bucket_count,
            seed=seed,
            method=method,
            dataset_dir=dataset_dir,
        )
        results.extend(run.results)
        generated_files.extend(run.generated_files)
        if run.generated_manifest is not None:
            generated_manifests.append(run.generated_manifest)

    return results, generated_files, generated_manifests


def _rows_from_results(
    results: Sequence[BenchmarkResult],
    *,
    dataset_kind: str,
    bucket_strategy: str,
) -> list[ScaleClosureRow]:
    rows = []
    for result in results:
        rows.append(
            ScaleClosureRow(
                dataset=result.distribution,
                dataset_kind=dataset_kind,
                bucket_strategy=bucket_strategy,
                n=result.n,
                buckets=result.bucket_count,
                seed=result.seed,
                method=result.method,
                fit_ms=result.fit_ms,
                bucket_ms=result.bucket_ms,
                sort_ms=result.sort_ms,
                total_ms=result.total_ms,
                variance=result.metrics.variance,
                max_bucket=result.metrics.max_bucket_size,
                empty=result.metrics.empty_bucket_count,
                ok=result.correct,
                dataset_file=result.dataset_file,
            )
        )
    return rows


def _expand_all_selection(names: Sequence[str], supported: Sequence[str], label: str) -> tuple[str, ...]:
    lowered = tuple(name.lower() for name in names)
    if "all" in lowered:
        if len(lowered) > 1:
            raise ValueError(f"use either all {label}s or explicit {label} names")
        return tuple(supported)
    return _normalize_names(lowered, supported, label)


def _normalize_names(names: Sequence[str], supported: Sequence[str], label: str) -> tuple[str, ...]:
    supported_set = set(supported)
    normalized = tuple(name.lower() for name in names)
    invalid = [name for name in normalized if name not in supported_set]
    if invalid:
        supported_text = ", ".join(supported)
        raise ValueError(f"unsupported {label} '{invalid[0]}'; expected one of: {supported_text}")
    return normalized


def _strategies_from_cli(strategy: str) -> tuple[str, ...]:
    if strategy == "both":
        return DEFAULT_BUCKET_STRATEGIES
    return (strategy,)


def _normalize_bucket_strategies(strategies: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(strategy.lower() for strategy in strategies)
    invalid = [strategy for strategy in normalized if strategy not in DEFAULT_BUCKET_STRATEGIES]
    if invalid:
        supported = ", ".join(DEFAULT_BUCKET_STRATEGIES)
        raise ValueError(f"unsupported bucket strategy '{invalid[0]}'; expected one of: {supported}")
    return normalized


def _validate_n_values(n_values: Sequence[int]) -> tuple[int, ...]:
    if not n_values:
        raise ValueError("at least one n value is required")
    invalid = [n for n in n_values if n <= 0]
    if invalid:
        raise ValueError("n values must be positive")
    return tuple(int(n) for n in n_values)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    sys.exit(main())
