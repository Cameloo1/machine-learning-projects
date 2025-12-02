"""Visualization utilities for anomaly detection results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def plot_feature_correlations(
    correlations: pd.Series,
    output_path: str | Path,
    title: str = "Feature Correlations with Anomaly Score",
) -> None:
    """Plot feature correlations as a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if x > 0 else "#2ca02c" for x in correlations.values]
    bars = ax.barh(correlations.index, correlations.values, color=colors, alpha=0.7)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Correlation Coefficient", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, correlations.values)):
        width = bar.get_width()
        ax.text(
            width + (0.01 if width >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=9,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature correlations plot to {output_path}")


def plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    output_path: str | Path,
    model_name: str = "Model",
    bins: int = 50,
) -> None:
    """Plot histogram of anomaly scores, colored by true labels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    
    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        label="Normal",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.7,
        label="Anomaly",
        color="#d62728",
        edgecolor="black",
        linewidth=0.5,
    )
    
    ax.set_xlabel("Anomaly Score (higher = more anomalous)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"{model_name} - Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved score distribution plot to {output_path}")


def plot_confusion_matrix_heatmap(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    output_path: str | Path,
    model_name: str = "Model",
) -> None:
    """Plot confusion matrix as a heatmap."""
    cm = np.array([[tn, fp], [fn, tp]])
    cm_percent = cm / cm.sum() * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["Actual Normal", "Actual Anomaly"],
        ax=ax,
        cbar_kws={"label": "Count"},
        linewidths=1,
        linecolor="gray",
    )
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            text = ax.texts[i * 2 + j]
            text.set_text(f"{text.get_text()}\n({cm_percent[i, j]:.1f}%)")
    
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_model_comparison(
    iso_metrics: dict,
    ocsvm_metrics: dict,
    output_path: str | Path,
) -> None:
    """Create a comparison bar chart of model metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ["IsolationForest", "OneClassSVM"]
    roc_aucs = [iso_metrics["roc_auc"], ocsvm_metrics["roc_auc"]]
    precisions = [iso_metrics["precision_at_k"], ocsvm_metrics["precision_at_k"]]
    
    # ROC AUC comparison
    bars1 = axes[0].bar(models, roc_aucs, color=["#1f77b4", "#ff7f0e"], alpha=0.7, edgecolor="black", linewidth=1)
    axes[0].set_ylabel("ROC AUC Score", fontsize=12)
    axes[0].set_title("ROC AUC Comparison", fontsize=13, fontweight="bold")
    axes[0].set_ylim([0.9, 1.01])
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, roc_aucs):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    # Precision@k comparison
    bars2 = axes[1].bar(models, precisions, color=["#1f77b4", "#ff7f0e"], alpha=0.7, edgecolor="black", linewidth=1)
    axes[1].set_ylabel("Precision@k", fontsize=12)
    axes[1].set_title("Precision@k Comparison", fontsize=13, fontweight="bold")
    axes[1].set_ylim([0.9, 1.01])
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, precisions):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    
    plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot to {output_path}")


def plot_top_anomalies_features(
    explanations: list[dict],
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    """Visualize the most deviant features across top anomalies."""
    feature_counts: dict[str, int] = {}
    feature_avg_z: dict[str, list[float]] = {}
    
    for exp in explanations[:top_n]:
        for explanation_str in exp["explanations"]:
            # Parse "feature_name is +X.XX std from normal"
            parts = explanation_str.split(" is ")
            if len(parts) == 2:
                feature = parts[0]
                z_str = parts[1].split()[0]
                try:
                    z_score = float(z_str)
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
                    if feature not in feature_avg_z:
                        feature_avg_z[feature] = []
                    feature_avg_z[feature].append(abs(z_score))
                except ValueError:
                    continue
    
    # Calculate average absolute z-scores
    feature_avg = {k: np.mean(v) for k, v in feature_avg_z.items()}
    
    # Sort by frequency, then by average z-score
    sorted_features = sorted(
        feature_counts.keys(),
        key=lambda x: (feature_counts[x], feature_avg.get(x, 0)),
        reverse=True,
    )
    
    if not sorted_features:
        print("No features found in explanations for plotting.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Frequency plot
    freqs = [feature_counts[f] for f in sorted_features[:10]]
    colors1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(freqs)))
    ax1.barh(sorted_features[:10], freqs, color=colors1, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Frequency in Top Anomalies", fontsize=12)
    ax1.set_title("Most Common Deviant Features", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)
    
    # Average z-score plot
    z_scores = [feature_avg.get(f, 0) for f in sorted_features[:10]]
    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(z_scores)))
    ax2.barh(sorted_features[:10], z_scores, color=colors2, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Average |Z-Score|", fontsize=12)
    ax2.set_title("Average Deviation Magnitude", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    
    plt.suptitle("Top Anomalies - Feature Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved top anomalies feature analysis to {output_path}")

