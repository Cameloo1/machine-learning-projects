"""Evaluation script for trained pipelines."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import config
from .utils_io import load_dataframe, save_json
from .utils_plotting import plot_confusion_matrix


def load_split(name: str) -> pd.DataFrame:
    """Load a processed split by name."""

    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed split {name} not found at {path}.")
    return load_dataframe(path)


def evaluate_pipeline(model_name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Evaluate a pipeline and persist metrics + plots."""

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "per_class": classification_report(
            y_test, y_pred, output_dict=True, target_names=list(config.LABEL_MAPPING.values())
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "avg_probability": np.mean(y_proba, axis=0).tolist(),
    }

    metrics_path = config.METRICS_DIR / f"{model_name}_metrics.json"
    save_json(metrics, metrics_path)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        output_path=config.PLOTS_DIR / f"confusion_matrix_{model_name}.png",
        class_labels=list(config.LABEL_MAPPING.values()),
    )
    print(f"Saved metrics and confusion matrix for {model_name}")


def majority_baseline(y: pd.Series) -> float:
    """Return accuracy of a majority-class baseline."""

    majority_class = y.value_counts().idxmax()
    return (y == majority_class).mean()


def main() -> None:
    """CLI entrypoint for evaluation."""

    config.ensure_directories()
    test_df = load_split("test")
    X_test = test_df[config.FEATURE_COLUMNS]
    y_test = test_df[config.TARGET_COLUMN]

    baseline_acc = majority_baseline(y_test)
    print(f"Majority-class baseline accuracy: {baseline_acc:.4f}")

    models = {
        "xgb": joblib.load(config.MODELS_DIR / "xgb_pipeline.pkl"),
        "lgbm": joblib.load(config.MODELS_DIR / "lgbm_pipeline.pkl"),
    }

    for name, pipeline in models.items():
        evaluate_pipeline(name, pipeline, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained pipelines.")
    _ = parser.parse_args()
    main()

