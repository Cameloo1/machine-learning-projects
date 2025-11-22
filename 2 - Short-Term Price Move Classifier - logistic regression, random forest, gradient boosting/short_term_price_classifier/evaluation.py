"""Evaluation helpers for classifier models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PLOTS_DIR = Path("artifacts/plots")


def _ensure_plot_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", title.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "plot"


def _save_current_plot(title: str) -> Path:
    directory = _ensure_plot_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{_slugify(title)}.png"
    path = directory / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {path}")
    return path


def evaluate_classifier(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, Optional[float]]:
    """Compute common classification metrics."""

    metrics: Dict[str, Optional[float]] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": None,
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def print_evaluation_report(name: str, metrics: Dict[str, Optional[float]]) -> None:
    """Pretty-print classifier metrics."""

    print(f"\n{name} Metrics")
    print("-" * (len(name) + 8))
    for metric_name, value in metrics.items():
        if value is None:
            display_value = "N/A"
        else:
            display_value = f"{value:.3f}"
        print(f"{metric_name.capitalize():<10}: {display_value}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str) -> None:
    """Plot ROC curve for predicted probabilities."""

    plt.figure(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true=y_true, y_pred=y_proba)
    plt.title(title)
    plt.tight_layout()
    _save_current_plot(title)
    plt.show()


def plot_feature_importances(
    feature_names: list[str], importances: np.ndarray, title: str
) -> None:
    """Bar plot for feature importances."""

    order = np.argsort(importances)[::-1]
    sorted_names = np.array(feature_names)[order]
    sorted_importances = importances[order]

    plt.figure(figsize=(8, 4))
    plt.barh(sorted_names, sorted_importances)
    plt.title(title)
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_current_plot(title)
    plt.show()


def plot_logistic_coefficients(
    feature_names: list[str], coefficients: np.ndarray, title: str
) -> None:
    """Bar plot for logistic regression coefficients."""

    order = np.argsort(np.abs(coefficients))[::-1]
    sorted_names = np.array(feature_names)[order]
    sorted_coeffs = coefficients[order]

    plt.figure(figsize=(8, 4))
    plt.barh(sorted_names, sorted_coeffs)
    plt.title(title)
    plt.xlabel("Coefficient")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save_current_plot(title)
    plt.show()

