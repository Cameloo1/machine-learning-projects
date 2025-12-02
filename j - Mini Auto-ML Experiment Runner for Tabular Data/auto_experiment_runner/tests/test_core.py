from pathlib import Path

import numpy as np
import pandas as pd

from autotab.config import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelParams,
    OutputConfig,
    PreprocessingConfig,
    SplitConfig,
    TaskConfig,
    load_config,
)
from autotab.data import load_dataset, train_valid_split
from autotab.runner import build_leaderboard, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_demo_classification():
    """Classification demo config loads with expected metadata."""
    config_path = PROJECT_ROOT / "configs" / "demo_classification.yaml"
    config = load_config(str(config_path))
    assert config.task.problem_name == "titanic_survival"
    enabled_models = [m for m in config.models if m.enabled]
    assert len(enabled_models) == 2
    assert config.dataset.path.endswith("examples/data/titanic.csv")


def test_data_loading_and_split_shapes():
    """Dataset loading and splitting should preserve row counts."""
    config = load_config(str(PROJECT_ROOT / "configs" / "demo_classification.yaml"))
    dataset = load_dataset(config)
    split = train_valid_split(dataset, config)

    assert len(split.X_train) + len(split.X_valid) == len(dataset.X)
    assert len(split.y_train) + len(split.y_valid) == len(dataset.y)
    assert set(split.X_train.columns) == set(dataset.X.columns)


def test_run_experiment_smoke(tmp_path):
    """Running the experiment end-to-end should produce a leaderboard."""
    rng = np.random.default_rng(123)
    rows = 80
    df = pd.DataFrame(
        {
            "feature_num": rng.normal(0, 1, rows),
            "feature_cat": rng.choice(["A", "B", "C"], rows),
            "target": rng.integers(0, 2, rows),
        }
    )
    csv_path = tmp_path / "toy.csv"
    df.to_csv(csv_path, index=False)

    config = ExperimentConfig(
        dataset=DatasetConfig(path=str(csv_path), target_column="target", id_columns=[]),
        task=TaskConfig(type="classification", problem_name="smoke_test"),
        preprocessing=PreprocessingConfig(),
        models=[
            ModelParams(name="logistic_regression", enabled=True, params={"max_iter": 500}),
            ModelParams(name="random_forest", enabled=True, params={"n_estimators": 50}),
        ],
        evaluation=EvaluationConfig(
            primary_metric="f1_macro",
            additional_metrics=["accuracy"],
            split=SplitConfig(type="holdout", test_size=0.2, random_state=42, stratify=True),
        ),
        output=OutputConfig(),
    )

    results = run_experiment(config)
    assert results, "Expected at least one model to be trained"

    leaderboard = build_leaderboard(results, config.evaluation.primary_metric)
    assert not leaderboard.empty
