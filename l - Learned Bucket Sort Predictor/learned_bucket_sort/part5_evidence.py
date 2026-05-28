"""Part 5 evidence artifact and plotting workflow."""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Final, Iterable, Literal, Sequence

from learned_bucket_sort.amortized_benchmark import (
    AmortizedBenchmarkResult,
    run_amortized_benchmarks,
    write_amortized_json_artifact,
)
from learned_bucket_sort.benchmark import BenchmarkResult, run_benchmarks, write_json_artifact
from learned_bucket_sort.data import SUPPORTED_DISTRIBUTIONS
from learned_bucket_sort.scale_closure import (
    DEFAULT_BUCKET_STRATEGIES,
    FIXED_BUCKET_STRATEGY,
    SCALED_BUCKET_STRATEGY,
    run_scale_closure,
)
from learned_bucket_sort.scenarios import DEFAULT_DATASET_DIR, SUPPORTED_SCENARIOS
from learned_bucket_sort.torch_mlp_cdf import DeviceRequest, TorchMLPCDFConfig


PART5_SCALE_NS: Final[tuple[int, ...]] = (5_000, 50_000, 100_000, 250_000)
PART5_NORMAL_N: Final[int] = 5_000
PART5_NORMAL_BUCKETS: Final[int] = 50
PART5_AMORTIZED_N: Final[int] = 50_000
PART5_AMORTIZED_BUCKETS: Final[int] = 100
PART5_SEED: Final[int] = 11
PART5_EVAL_SEED: Final[int] = 12
PART5_SCALE_ASSET: Final[str] = "part5-scale-closure-total-ms.png"
PART5_QUALITY_ASSET: Final[str] = "part5-bucket-quality.png"
PART5_AMORTIZED_ASSET: Final[str] = "part5-amortized-runtime-breakdown.png"
PROMOTED_PART5_ASSETS: Final[tuple[str, ...]] = (
    PART5_SCALE_ASSET,
    PART5_QUALITY_ASSET,
    PART5_AMORTIZED_ASSET,
)


ArtifactFamily = Literal["scale_closure", "benchmark_methods", "amortized_benchmark"]


@dataclass(frozen=True)
class ScaleTotalMsPoint:
    """Median scale-closure runtime point."""

    bucket_strategy: str
    n: int
    method: str
    median_total_ms: float


@dataclass(frozen=True)
class BucketQualityPoint:
    """Per-dataset normal benchmark bucket-quality point."""

    dataset: str
    method: str
    max_bucket_ratio: float


@dataclass(frozen=True)
class AmortizedBreakdownPoint:
    """Median reused-sort-path timing component."""

    method: str
    component: str
    median_ms: float


@dataclass(frozen=True)
class Part5PlotPaths:
    """Generated and promoted Part 5 plot paths."""

    generated: dict[str, Path]
    promoted: dict[str, Path]


@dataclass(frozen=True)
class Part5EvidenceRun:
    """Part 5 evidence bundle state."""

    evidence_dir: Path
    scale_artifact: Path
    benchmark_artifact: Path
    amortized_artifact: Path
    manifest_path: Path
    plot_paths: Part5PlotPaths
    generated_files: list[Path]
    generated_manifests: list[Path]


def run_part5_evidence(
    *,
    evidence_root: str | Path = Path("artifacts") / "evidence",
    assets_dir: str | Path = "assets",
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    timestamp: datetime | None = None,
    device_request: DeviceRequest = "auto",
) -> Part5EvidenceRun:
    """Regenerate Part 5 numeric evidence and promoted plots."""
    generated_at = timestamp or datetime.now(UTC)
    evidence_dir = resolve_evidence_dir(evidence_root, generated_at)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    plots_dir = evidence_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mlp_config = TorchMLPCDFConfig(device=device_request)
    scale_run = run_scale_closure(
        n_values=PART5_SCALE_NS,
        seed=PART5_SEED,
        distributions=SUPPORTED_DISTRIBUTIONS,
        scenarios=SUPPORTED_SCENARIOS,
        bucket_strategies=DEFAULT_BUCKET_STRATEGIES,
        dataset_dir=dataset_dir,
        out_dir=evidence_dir,
        timestamp=generated_at,
    )

    normal_results: list[BenchmarkResult] = []
    normal_generated_files: list[Path] = []
    normal_generated_manifests: list[Path] = []
    for distribution in SUPPORTED_DISTRIBUTIONS:
        run = run_benchmarks(
            distribution=distribution,
            scenario=None,
            n=PART5_NORMAL_N,
            bucket_count=PART5_NORMAL_BUCKETS,
            seed=PART5_SEED,
            method="all",
            dataset_dir=dataset_dir,
            mlp_config=mlp_config,
        )
        normal_results.extend(run.results)
        normal_generated_files.extend(run.generated_files)
        if run.generated_manifest is not None:
            normal_generated_manifests.append(run.generated_manifest)

    scenario_run = run_benchmarks(
        distribution=None,
        scenario="all",
        n=PART5_NORMAL_N,
        bucket_count=PART5_NORMAL_BUCKETS,
        seed=PART5_SEED,
        method="all",
        dataset_dir=dataset_dir,
        mlp_config=mlp_config,
    )
    normal_results.extend(scenario_run.results)
    normal_generated_files.extend(scenario_run.generated_files)
    if scenario_run.generated_manifest is not None:
        normal_generated_manifests.append(scenario_run.generated_manifest)

    benchmark_artifact = write_json_artifact(
        normal_results,
        out_dir=evidence_dir,
        config={
            "part": "5",
            "artifact_family": "benchmark_methods",
            "distributions": list(SUPPORTED_DISTRIBUTIONS),
            "scenarios": list(SUPPORTED_SCENARIOS),
            "method": "all",
            "n": PART5_NORMAL_N,
            "buckets": PART5_NORMAL_BUCKETS,
            "seed": PART5_SEED,
            "dataset_dir": str(dataset_dir),
            "device_request": device_request,
        },
    )

    amortized_results: list[AmortizedBenchmarkResult] = []
    amortized_generated_files: list[Path] = []
    amortized_generated_manifests: list[Path] = []
    for distribution in SUPPORTED_DISTRIBUTIONS:
        run = run_amortized_benchmarks(
            distribution=distribution,
            scenario=None,
            n=PART5_AMORTIZED_N,
            bucket_count=PART5_AMORTIZED_BUCKETS,
            train_seed=PART5_SEED,
            eval_seed=PART5_EVAL_SEED,
            method="all",
            dataset_dir=dataset_dir,
            mlp_config=mlp_config,
        )
        amortized_results.extend(run.results)

    for scenario in SUPPORTED_SCENARIOS:
        run = run_amortized_benchmarks(
            distribution=None,
            scenario=scenario,
            n=PART5_AMORTIZED_N,
            bucket_count=PART5_AMORTIZED_BUCKETS,
            train_seed=PART5_SEED,
            eval_seed=PART5_EVAL_SEED,
            method="all",
            dataset_dir=dataset_dir,
            mlp_config=mlp_config,
        )
        amortized_results.extend(run.results)
        amortized_generated_files.extend(run.generated_files)
        amortized_generated_manifests.extend(run.generated_manifests)

    amortized_artifact = write_amortized_json_artifact(
        amortized_results,
        out_dir=evidence_dir,
        config={
            "part": "5",
            "artifact_family": "amortized_benchmark",
            "distributions": list(SUPPORTED_DISTRIBUTIONS),
            "scenarios": list(SUPPORTED_SCENARIOS),
            "method": "all",
            "n": PART5_AMORTIZED_N,
            "buckets": PART5_AMORTIZED_BUCKETS,
            "train_seed": PART5_SEED,
            "eval_seed": PART5_EVAL_SEED,
            "dataset_dir": str(dataset_dir),
            "device_request": device_request,
        },
    )

    assert scale_run.artifact_path is not None
    plot_paths = generate_part5_plots(
        scale_artifact=scale_run.artifact_path,
        benchmark_artifact=benchmark_artifact,
        amortized_artifact=amortized_artifact,
        output_dir=plots_dir,
        assets_dir=assets_dir,
    )

    manifest_path = write_part5_manifest(
        evidence_dir=evidence_dir,
        generated_at=generated_at,
        scale_artifact=scale_run.artifact_path,
        benchmark_artifact=benchmark_artifact,
        amortized_artifact=amortized_artifact,
        plot_paths=plot_paths,
        generated_files=[
            *scale_run.generated_files,
            *normal_generated_files,
            *amortized_generated_files,
        ],
        generated_manifests=[
            *scale_run.generated_manifests,
            *normal_generated_manifests,
            *amortized_generated_manifests,
        ],
        device_request=device_request,
        resolved_devices=_resolved_devices([*normal_results, *amortized_results]),
    )

    return Part5EvidenceRun(
        evidence_dir=evidence_dir,
        scale_artifact=scale_run.artifact_path,
        benchmark_artifact=benchmark_artifact,
        amortized_artifact=amortized_artifact,
        manifest_path=manifest_path,
        plot_paths=plot_paths,
        generated_files=[*scale_run.generated_files, *normal_generated_files, *amortized_generated_files],
        generated_manifests=[
            *scale_run.generated_manifests,
            *normal_generated_manifests,
            *amortized_generated_manifests,
        ],
    )


def resolve_evidence_dir(evidence_root: str | Path, timestamp: datetime) -> Path:
    """Return a non-existing timestamped evidence directory."""
    root = Path(evidence_root)
    base = root / timestamp.strftime("%Y%m%d_%H%M%S")
    if not base.exists():
        return base

    for index in range(1, 100):
        candidate = root / f"{base.name}_{index:02d}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find an unused evidence directory under {root}")


def load_scale_closure_artifact(path: str | Path) -> dict[str, object]:
    """Load and validate a scale-closure artifact."""
    return _load_artifact(path, "scale_closure", required_fields={"dataset", "dataset_kind", "bucket_strategy", "total_ms"})


def load_benchmark_artifact(path: str | Path) -> dict[str, object]:
    """Load and validate a normal benchmark artifact."""
    return _load_artifact(path, "benchmark_methods", required_fields={"distribution", "method", "fit_ms", "total_ms", "metrics"})


def load_amortized_artifact(path: str | Path) -> dict[str, object]:
    """Load and validate an amortized benchmark artifact."""
    return _load_artifact(
        path,
        "amortized_benchmark",
        required_fields={"distribution", "method", "train_ms", "predict_ms", "sort_path_total_ms", "metrics"},
    )


def scale_total_ms_points(payload: dict[str, object], *, allow_failed: bool = False) -> list[ScaleTotalMsPoint]:
    """Group scale-closure rows by n, method, and bucket strategy."""
    rows = _valid_rows(payload, "scale closure", allow_failed=allow_failed)
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        key = (str(row["bucket_strategy"]), int(row["n"]), str(row["method"]))
        grouped.setdefault(key, []).append(float(row["total_ms"]))

    return [
        ScaleTotalMsPoint(
            bucket_strategy=strategy,
            n=n,
            method=method,
            median_total_ms=float(median(values)),
        )
        for (strategy, n, method), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
    ]


def bucket_quality_points(payload: dict[str, object], *, allow_failed: bool = False) -> list[BucketQualityPoint]:
    """Return per-dataset max-bucket ratio points for normal benchmark rows."""
    rows = _valid_rows(payload, "benchmark", allow_failed=allow_failed)
    points = []
    for row in rows:
        n = int(row["n"])
        if n <= 0:
            raise ValueError("benchmark rows must have positive n")
        points.append(
            BucketQualityPoint(
                dataset=str(row["distribution"]),
                method=str(row["method"]),
                max_bucket_ratio=_max_bucket(row) / n,
            )
        )
    return points


def amortized_breakdown_points(payload: dict[str, object], *, allow_failed: bool = False) -> list[AmortizedBreakdownPoint]:
    """Return median reused-sort-path component timings by method."""
    rows = _valid_rows(payload, "amortized benchmark", allow_failed=allow_failed)
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        method = str(row["method"])
        for component in ("predict_ms", "bucket_ms", "sort_ms", "sort_path_total_ms"):
            grouped.setdefault((method, component), []).append(float(row[component]))

    return [
        AmortizedBreakdownPoint(method=method, component=component, median_ms=float(median(values)))
        for (method, component), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def generate_part5_plots(
    *,
    scale_artifact: str | Path,
    benchmark_artifact: str | Path,
    amortized_artifact: str | Path,
    output_dir: str | Path,
    assets_dir: str | Path,
) -> Part5PlotPaths:
    """Generate Part 5 plots from numeric JSON artifacts and promote selected assets."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assets = Path(assets_dir)
    assets.mkdir(parents=True, exist_ok=True)

    scale_payload = load_scale_closure_artifact(scale_artifact)
    benchmark_payload = load_benchmark_artifact(benchmark_artifact)
    amortized_payload = load_amortized_artifact(amortized_artifact)

    generated = {
        PART5_SCALE_ASSET: output / PART5_SCALE_ASSET,
        PART5_QUALITY_ASSET: output / PART5_QUALITY_ASSET,
        PART5_AMORTIZED_ASSET: output / PART5_AMORTIZED_ASSET,
    }
    _plot_scale_total_ms(scale_total_ms_points(scale_payload), generated[PART5_SCALE_ASSET])
    _plot_bucket_quality(bucket_quality_points(benchmark_payload), generated[PART5_QUALITY_ASSET])
    _plot_amortized_breakdown(amortized_breakdown_points(amortized_payload), generated[PART5_AMORTIZED_ASSET])

    promoted: dict[str, Path] = {}
    for name, source in generated.items():
        destination = assets / name
        shutil.copyfile(source, destination)
        promoted[name] = destination

    return Part5PlotPaths(generated=generated, promoted=promoted)


def write_part5_manifest(
    *,
    evidence_dir: str | Path,
    generated_at: datetime,
    scale_artifact: str | Path,
    benchmark_artifact: str | Path,
    amortized_artifact: str | Path,
    plot_paths: Part5PlotPaths,
    generated_files: Sequence[Path],
    generated_manifests: Sequence[Path],
    device_request: DeviceRequest,
    resolved_devices: Sequence[str],
) -> Path:
    """Write the Part 5 evidence manifest."""
    output = Path(evidence_dir)
    manifest_path = output / "manifest.json"
    payload = {
        "generated_at": generated_at.isoformat(),
        "python_version": platform.python_version(),
        "config": {
            "scale_n_values": list(PART5_SCALE_NS),
            "normal_n": PART5_NORMAL_N,
            "normal_buckets": PART5_NORMAL_BUCKETS,
            "amortized_n": PART5_AMORTIZED_N,
            "amortized_buckets": PART5_AMORTIZED_BUCKETS,
            "seed": PART5_SEED,
            "eval_seed": PART5_EVAL_SEED,
            "distributions": list(SUPPORTED_DISTRIBUTIONS),
            "scenarios": list(SUPPORTED_SCENARIOS),
            "device_request": device_request,
            "resolved_devices": list(resolved_devices),
        },
        "equivalent_commands": [
            ".\\.venv\\Scripts\\python.exe scripts\\run_part5_evidence.py",
        ],
        "artifacts": {
            "scale_closure": _display_path(scale_artifact),
            "benchmark_methods": _display_path(benchmark_artifact),
            "amortized_benchmark": _display_path(amortized_artifact),
        },
        "plots": {name: _display_path(path) for name, path in plot_paths.generated.items()},
        "promoted_assets": {name: _display_path(path) for name, path in plot_paths.promoted.items()},
        "generated_dataset_files": [_display_path(path) for path in generated_files],
        "generated_dataset_manifests": [_display_path(path) for path in generated_manifests],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def ensure_plot_backend() -> str:
    """Force Matplotlib to a non-GUI backend and return the selected backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    return str(matplotlib.get_backend())


def _load_artifact(path: str | Path, family: ArtifactFamily, *, required_fields: set[str]) -> dict[str, object]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"{family} artifact not found: {artifact_path}")

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{artifact_path} is not a JSON object")

    results = payload.get("results")
    if not isinstance(results, list) or len(results) == 0:
        raise ValueError(f"{artifact_path} is not a {family} artifact: missing non-empty results")

    first = results[0]
    if not isinstance(first, dict) or not required_fields <= set(first):
        raise ValueError(f"{artifact_path} is not a {family} artifact")
    return payload


def _valid_rows(payload: dict[str, object], label: str, *, allow_failed: bool) -> list[dict[str, object]]:
    rows = payload["results"]
    if not isinstance(rows, list):
        raise ValueError(f"{label} artifact has invalid results")
    typed_rows = [row for row in rows if isinstance(row, dict)]
    if len(typed_rows) != len(rows):
        raise ValueError(f"{label} artifact contains invalid row data")
    failed = [row for row in typed_rows if row.get("ok") is not True and row.get("correct") is not True]
    if failed and not allow_failed:
        raise ValueError(f"{label} artifact contains failed benchmark rows")
    return typed_rows


def _max_bucket(row: dict[str, object]) -> int:
    if "max_bucket" in row:
        return int(row["max_bucket"])
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or "max_bucket_size" not in metrics:
        raise ValueError("row is missing max bucket data")
    return int(metrics["max_bucket_size"])


def _plot_scale_total_ms(points: Sequence[ScaleTotalMsPoint], output_path: Path) -> None:
    ensure_plot_backend()
    import matplotlib.pyplot as plt

    strategies = [FIXED_BUCKET_STRATEGY, SCALED_BUCKET_STRATEGY]
    method_labels = {"analytic_baseline": "baseline", "linear_cdf": "linear CDF"}
    colors = {"analytic_baseline": "#6f6f6f", "linear_cdf": "#1f77b4"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.patch.set_facecolor("white")
    for axis, strategy in zip(axes, strategies, strict=True):
        strategy_points = [point for point in points if point.bucket_strategy == strategy]
        n_values = sorted({point.n for point in strategy_points})
        x_positions = list(range(len(n_values)))
        for method in ("analytic_baseline", "linear_cdf"):
            values = [
                next(
                    point.median_total_ms
                    for point in strategy_points
                    if point.n == n and point.method == method
                )
                for n in n_values
            ]
            axis.plot(x_positions, values, marker="o", linewidth=2.5, color=colors[method], label=method_labels[method])
        axis.set_title(f"{strategy.title()} buckets")
        axis.set_xticks(x_positions, [f"{n:,}" for n in n_values], rotation=25, ha="right")
        axis.set_xlabel("n")
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median total runtime (ms)")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle("Scale closure: learned CDF runtime improves as overloaded buckets grow", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_bucket_quality(points: Sequence[BucketQualityPoint], output_path: Path) -> None:
    ensure_plot_backend()
    import matplotlib.pyplot as plt

    methods = ("analytic_baseline", "linear_cdf", "mlp_cdf")
    colors = {"analytic_baseline": "#8d8d8d", "linear_cdf": "#1f77b4", "mlp_cdf": "#2ca02c"}
    labels = {"analytic_baseline": "baseline", "linear_cdf": "linear CDF", "mlp_cdf": "MLP CDF"}
    datasets = sorted({point.dataset for point in points})
    positions = list(range(len(datasets)))
    width = 0.24

    fig, axis = plt.subplots(figsize=(13, 5.2))
    fig.patch.set_facecolor("white")
    for offset, method in zip((-width, 0.0, width), methods, strict=True):
        values = [
            next(point.max_bucket_ratio for point in points if point.dataset == dataset and point.method == method)
            for dataset in datasets
        ]
        axis.bar([position + offset for position in positions], values, width=width, color=colors[method], label=labels[method])

    axis.set_title("Bucket quality: MLP reduces worst-bucket share on hard shapes", fontsize=13, fontweight="bold")
    axis.set_ylabel("Largest bucket / n")
    axis.set_xticks(positions, datasets, rotation=35, ha="right")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_amortized_breakdown(points: Sequence[AmortizedBreakdownPoint], output_path: Path) -> None:
    ensure_plot_backend()
    import matplotlib.pyplot as plt

    methods = ("analytic_baseline", "linear_cdf", "mlp_cdf")
    labels = {"analytic_baseline": "baseline", "linear_cdf": "linear CDF", "mlp_cdf": "MLP CDF"}
    components = ("predict_ms", "bucket_ms", "sort_ms")
    component_labels = {"predict_ms": "predict", "bucket_ms": "bucket", "sort_ms": "sort"}
    colors = {"predict_ms": "#d62728", "bucket_ms": "#ffbf00", "sort_ms": "#1f77b4"}
    positions = list(range(len(methods)))
    bottoms = [0.0] * len(methods)

    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    fig.patch.set_facecolor("white")
    for component in components:
        values = [
            next(point.median_ms for point in points if point.method == method and point.component == component)
            for method in methods
        ]
        axis.bar(positions, values, bottom=bottoms, color=colors[component], label=component_labels[component], width=0.58)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]

    totals = {
        point.method: point.median_ms
        for point in points
        if point.component == "sort_path_total_ms"
    }
    for position, method in zip(positions, methods, strict=True):
        axis.text(position, bottoms[position] + max(bottoms) * 0.025, f"{totals[method]:.1f} ms", ha="center", va="bottom", fontsize=9)

    axis.set_title("Amortized path: MLP quality costs prediction time", fontsize=13, fontweight="bold")
    axis.set_ylabel("Median reused sort-path time (ms)")
    axis.set_xticks(positions, [labels[method] for method in methods])
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _resolved_devices(results: Iterable[BenchmarkResult | AmortizedBenchmarkResult]) -> list[str]:
    return sorted({str(result.device) for result in results if result.device is not None})


def _display_path(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return str(resolved)
