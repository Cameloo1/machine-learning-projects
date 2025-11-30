"""Training and evaluation helpers for sentiment classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from .config import CLASS_LABELS, CONFUSION_MATRIX_FILES


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
    )
    model.fit(X_train, y_train)
    return model


def train_linear_svc(X_train, y_train) -> LinearSVC:
    """Train a linear SVM classifier."""
    model = LinearSVC(
        class_weight="balanced",
        max_iter=5000,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    figures_dir: str | Path,
) -> Dict[str, Any]:
    """Compute metrics and save confusion matrix plot."""
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)

    printable_report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        digits=3,
        zero_division=0,
    )
    print(
        f"\n=== {model_name} Classification Report ===\n"
        f"Train accuracy: {train_score:.3f}\n{printable_report}"
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = CONFUSION_MATRIX_FILES.get(
        model_name,
        figures_dir / f"cm_{model_name.lower().replace(' ', '_')}.png",
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"{model_name} – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    metrics = {
        "model_name": model_name,
        "accuracy": report_dict.get("accuracy"),
        "macro_f1": report_dict.get("macro avg", {}).get("f1-score"),
        "weighted_f1": report_dict.get("weighted avg", {}).get("f1-score"),
    }
    return metrics


