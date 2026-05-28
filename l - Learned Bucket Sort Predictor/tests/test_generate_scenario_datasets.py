import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from learned_bucket_sort.scenarios import SUPPORTED_SCENARIOS
from scripts.generate_scenario_datasets import build_parser, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def unique_test_dir(label):
    path = PROJECT_ROOT / ".test-runs" / f"{label}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_script_requires_explicit_scenario():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_script_writes_single_scenario_dataset_and_manifest(capsys):
    out_dir = unique_test_dir("script-single")

    exit_code = main(["--scenario", "response_times", "--n", "20", "--seed", "1", "--out", str(out_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "wrote dataset:" in captured.out
    assert "wrote manifest:" in captured.out
    assert len(list(out_dir.glob("response_times_n20_seed1_*.npy"))) == 1
    assert len(list(out_dir.glob("manifest_scenarios_n20_seed1_*.json"))) == 1


def test_script_writes_all_scenario_datasets_and_manifest():
    out_dir = unique_test_dir("script-all")

    exit_code = main(["--scenario", "all", "--n", "20", "--seed", "1", "--out", str(out_dir)])

    assert exit_code == 0
    assert len(list(out_dir.glob("*.npy"))) == len(SUPPORTED_SCENARIOS)
    manifests = list(out_dir.glob("manifest_scenarios_n20_seed1_*.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert [row["scenario"] for row in payload["scenarios"]] == list(SUPPORTED_SCENARIOS)


def test_script_timestamped_files_avoid_overwrite():
    out_dir = unique_test_dir("script-no-overwrite")

    main(["--scenario", "response_times", "--n", "20", "--seed", "1", "--out", str(out_dir)])
    main(["--scenario", "response_times", "--n", "20", "--seed", "1", "--out", str(out_dir)])

    assert len(list(out_dir.glob("response_times_n20_seed1_*.npy"))) == 2
    assert len(list(out_dir.glob("manifest_scenarios_n20_seed1_*.json"))) == 2


def test_script_runs_directly_from_project_root():
    out_dir = unique_test_dir("script-direct")
    script_path = PROJECT_ROOT / "scripts" / "generate_scenario_datasets.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--scenario",
            "response_times",
            "--n",
            "20",
            "--seed",
            "1",
            "--out",
            str(out_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "wrote dataset:" in completed.stdout
    assert len(list(out_dir.glob("response_times_n20_seed1_*.npy"))) == 1
