import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from learned_bucket_sort.benchmark import BASELINE_METHOD, LINEAR_METHOD
from learned_bucket_sort.scale_closure import (
    bucket_count_for_strategy,
    format_scale_closure_summary,
    main,
    run_scale_closure,
    scaled_bucket_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def unique_test_dir(label):
    path = PROJECT_ROOT / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_scaled_bucket_count_uses_minimum_until_scale_exceeds_it():
    assert scaled_bucket_count(5_000) == 50
    assert scaled_bucket_count(50_000) == 50
    assert scaled_bucket_count(100_000) == 100
    assert scaled_bucket_count(250_000) == 250


def test_bucket_count_for_strategy_resolves_fixed_and_scaled_counts():
    assert bucket_count_for_strategy("fixed", n=100_000, fixed_buckets=50) == 50
    assert bucket_count_for_strategy("scaled", n=100_000) == 100

    with pytest.raises(ValueError, match="unsupported bucket strategy"):
        bucket_count_for_strategy("random", n=100_000)


def test_run_scale_closure_writes_flat_artifact_and_materializes_scenarios():
    dataset_dir = unique_test_dir("scale-datasets")
    out_dir = unique_test_dir("scale-artifact")
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    run = run_scale_closure(
        n_values=(20,),
        seed=1,
        fixed_buckets=5,
        distributions=("uniform",),
        scenarios=("response_times",),
        bucket_strategies=("fixed",),
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        timestamp=timestamp,
    )

    assert len(run.rows) == 4
    assert len(run.generated_files) == 1
    assert len(run.generated_manifests) == 1
    assert run.artifact_path == out_dir / "scale_closure_20260102_030405.json"

    methods = [row.method for row in run.rows]
    assert methods == [BASELINE_METHOD, LINEAR_METHOD, BASELINE_METHOD, LINEAR_METHOD]
    assert {row.dataset_kind for row in run.rows} == {"distribution", "scenario"}
    assert {row.bucket_strategy for row in run.rows} == {"fixed"}
    assert all(row.ok for row in run.rows)

    distribution_rows = [row for row in run.rows if row.dataset_kind == "distribution"]
    scenario_rows = [row for row in run.rows if row.dataset_kind == "scenario"]
    assert all(row.dataset_file is None for row in distribution_rows)
    assert all(row.dataset_file is not None for row in scenario_rows)

    payload = json.loads(run.artifact_path.read_text(encoding="utf-8"))
    assert payload["generated_at"] == "2026-01-02T03:04:05+00:00"
    assert payload["config"]["n_values"] == [20]
    assert payload["config"]["bucket_strategies"] == ["fixed"]
    assert payload["results"][0]["dataset"] == "uniform"
    assert payload["results"][0]["dataset_kind"] == "distribution"
    assert payload["results"][0]["buckets"] == 5
    assert "metrics" not in payload["results"][0]
    assert {"variance", "max_bucket", "empty"} <= set(payload["results"][0])


def test_format_scale_closure_summary_is_plain_and_readable():
    run = run_scale_closure(
        n_values=(20,),
        seed=1,
        fixed_buckets=5,
        distributions=("uniform",),
        scenarios=(),
        bucket_strategies=("fixed",),
        write_artifact=False,
    )

    summary = format_scale_closure_summary(run.rows)

    assert "kind" in summary
    assert "strategy" in summary
    assert "uniform" in summary
    assert "linear_cdf" in summary
    assert "\033[" not in summary


def test_scale_closure_main_writes_artifact_for_small_selection(capsys):
    dataset_dir = unique_test_dir("scale-main-datasets")
    out_dir = unique_test_dir("scale-main-artifact")

    exit_code = main(
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
            str(dataset_dir),
            "--out",
            str(out_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "generated dataset:" in captured.out
    assert "generated manifest:" in captured.out
    assert "wrote artifact:" in captured.out
    assert "response_times" in captured.out
    assert len(list(out_dir.glob("scale_closure_*.json"))) == 1


def test_scale_closure_main_rejects_all_mixed_with_explicit_name(capsys):
    with pytest.raises(SystemExit):
        main(["--dist", "all", "uniform", "--scenario", "response_times", "--no-artifact"])

    captured = capsys.readouterr()
    assert "use either all distributions or explicit distribution names" in captured.err
