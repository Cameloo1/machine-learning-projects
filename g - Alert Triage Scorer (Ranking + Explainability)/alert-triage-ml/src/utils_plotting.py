"""Plotting helpers for artifacts and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from . import config
from .utils_io import ensure_parent_dir


sns.set_theme(style="whitegrid")


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str,
    output_path: Path = config.PLOTS_DIR / "class_distribution.png",
) -> None:
    """Plot and save class distribution."""

    counts = df[target_col].map(config.LABEL_MAPPING).value_counts().sort_index()
    plot_df = counts.reset_index()
    plot_df.columns = ["priority", "count"]
    ensure_parent_dir(output_path)
    plt.figure(figsize=(6, 4))
    sns.barplot(data=plot_df, x="priority", y="count", hue="priority", palette="viridis", legend=False)
    plt.title("Alert Priority Distribution")
    plt.xlabel("Priority")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    output_path: Path,
    class_labels: Iterable[str] = ("Low", "Medium", "High"),
) -> None:
    """Plot and save a confusion matrix heatmap."""

    ensure_parent_dir(output_path)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_numeric_histograms(
    df: pd.DataFrame,
    columns: Iterable[str],
    output_dir: Path = config.PLOTS_DIR,
) -> None:
    """Persist histograms for selected numeric columns."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for col in columns:
        plt.figure(figsize=(5, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png")
        plt.close()

