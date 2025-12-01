import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from autotab.config import ExperimentConfig
from autotab.data import DatasetMetadata


def prepare_output_root(config: ExperimentConfig) -> Path:
    """
    Create a unique experiment directory under config.output.base_dir
    using problem_name + timestamp.
    Copy the original config file into this directory.

    Args:
        config: Experiment configuration
    
    Returns:
        Path to the created experiment directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    dir_name = f"{config.task.problem_name}_{timestamp}"
    
    # Create full path
    root_path = Path(config.output.base_dir) / dir_name
    root_path.mkdir(parents=True, exist_ok=True)

    # Copy the original config file when available for full reproducibility
    source_config_path = getattr(config, "_source_config_path", None)
    destination = root_path / "config.yaml"
    if source_config_path and Path(source_config_path).is_file():
        shutil.copy2(source_config_path, destination)
    else:
        _dump_config(config, destination)

    return root_path


def _dump_config(config: ExperimentConfig, destination: Path) -> None:
    """
    Persist the parsed config to YAML when the original file path is unavailable.
    """
    try:
        config_dict = config.model_dump()
    except AttributeError:
        config_dict = config.dict()
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False)


def get_model_output_dir(root: Path, model_name: str) -> Path:
    """
    Get (and create) the output directory for a specific model.
    
    Args:
        root: Root experiment directory
        model_name: Name of the model
    
    Returns:
        Path to the model's output directory
    """
    model_dir = root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_json(path: Path, obj: dict | list):
    """
    Save a dictionary or list as JSON.
    
    Args:
        path: Path to save the JSON file
        obj: Dictionary or list to save
    """
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def save_leaderboard(df: pd.DataFrame, root: Path):
    """
    Save leaderboard as both CSV and JSON.
    
    Args:
        df: Leaderboard DataFrame
        root: Root experiment directory
    """
    # Save as CSV
    df.to_csv(root / "leaderboard.csv", index=False)
    
    # Save as JSON
    leaderboard_dict = df.to_dict(orient='records')
    save_json(root / "leaderboard.json", leaderboard_dict)


def plot_confusion_matrix(cm: np.ndarray, model_dir: Path):
    """
    Plot and save confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        model_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate counts
    num_rows, num_cols = cm.shape
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_xticks(range(num_cols))
    ax.set_yticks(range(num_rows))
    fig.tight_layout()
    fig.savefig(model_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(
    feature_importance: list[tuple[str, float]],
    model_dir: Path,
    top_k: int = 20
):
    """
    Plot and save feature importance bar chart.
    
    Args:
        feature_importance: List of (feature_name, importance) tuples
        model_dir: Directory to save the plot
        top_k: Number of top features to display
    """
    # Take top K features
    top_features = feature_importance[:top_k]
    
    if not top_features:
        return
    
    # Reverse for plotting (highest at top)
    top_features = list(reversed(top_features))
    
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    plt.figure(figsize=(10, max(6, len(names) * 0.3)))
    plt.barh(names, values, color='steelblue')
    plt.xlabel('Importance')
    plt.title(f'Top {len(top_features)} Feature Importances')
    plt.tight_layout()
    plt.savefig(model_dir / "feature_importance.png", dpi=150)
    plt.close()


def plot_residuals(y_true, y_pred, model_dir: Path):
    """
    Plot and save residuals scatter plot for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_dir: Directory to save the plot
    """
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot: predicted vs actual
    axes[0].scatter(y_pred, y_true, alpha=0.6, edgecolors='k')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('True Values')
    axes[0].set_title('Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(model_dir / "residuals.png", dpi=150)
    plt.close()


def save_dataset_metadata(metadata: DatasetMetadata, root: Path):
    """
    Save dataset metadata as JSON.
    
    Args:
        metadata: Dataset metadata object
        root: Root experiment directory
    """
    metadata_dict = {
        'n_rows': metadata.n_rows,
        'n_cols': metadata.n_cols,
        'n_numeric': metadata.n_numeric,
        'n_categorical': metadata.n_categorical,
    }
    save_json(root / "metadata.json", metadata_dict)


def generate_report(
    config: ExperimentConfig,
    metadata: DatasetMetadata,
    leaderboard: pd.DataFrame,
    root: Path
) -> str:
    """
    Generate a comprehensive Markdown report.
    
    Args:
        config: Experiment configuration
        metadata: Dataset metadata
        leaderboard: Leaderboard DataFrame
        root: Root experiment directory
    
    Returns:
        Markdown report as string
    """
    report_lines = []
    
    # Title
    report_lines.append(f"# Experiment Report: {config.task.problem_name}")
    report_lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n---\n")
    
    # Dataset Summary
    report_lines.append("## Dataset Summary\n")
    report_lines.append(f"- **Total Rows**: {metadata.n_rows}")
    report_lines.append(f"- **Total Columns**: {metadata.n_cols}")
    report_lines.append(f"- **Numeric Features**: {metadata.n_numeric}")
    report_lines.append(f"- **Categorical Features**: {metadata.n_categorical}")
    report_lines.append(f"- **Target Column**: {config.dataset.target_column}")
    
    # Task Configuration
    report_lines.append(f"\n## Task Configuration\n")
    report_lines.append(f"- **Task Type**: {config.task.type}")
    report_lines.append(f"- **Primary Metric**: {config.evaluation.primary_metric}")
    additional_metrics = ", ".join(config.evaluation.additional_metrics) if config.evaluation.additional_metrics else "None"
    report_lines.append(f"- **Additional Metrics**: {additional_metrics}")
    report_lines.append(f"- **Train/Valid Split**: {config.evaluation.split.test_size * 100:.0f}% validation")
    
    # Leaderboard
    report_lines.append(f"\n## Leaderboard\n")
    if leaderboard.empty:
        report_lines.append("_No trained models to display._")
    else:
        metric_columns = [col for col in leaderboard.columns if col not in {"model", "train_time_sec"}]
        top_n = min(5, len(leaderboard))
        report_lines.append(f"Top {top_n} of {len(leaderboard)} models ranked by **{config.evaluation.primary_metric}**:\n")
        
        header_cells = ["Rank", "Model", "Train Time (s)"] + metric_columns
        report_lines.append("| " + " | ".join(header_cells) + " |")
        report_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
        
        display_df = leaderboard.head(top_n)
        for idx, row in display_df.iterrows():
            metric_values = []
            for col in metric_columns:
                val = row.get(col)
                metric_values.append(f"{val:.4f}" if pd.notna(val) else "N/A")
            row_cells = [
                str(idx + 1),
                row['model'],
                f"{row['train_time_sec']:.3f}",
            ] + metric_values
            report_lines.append("| " + " | ".join(row_cells) + " |")
    
    # Best Model Summary
    if len(leaderboard) > 0:
        best_model = leaderboard.iloc[0]
        report_lines.append(f"\n## Best Model: {best_model['model']}\n")
        report_lines.append(f"**Winner**: `{best_model['model']}`\n")
        
        # Metrics
        report_lines.append("### Performance Metrics\n")
        metric_columns = [col for col in leaderboard.columns if col not in {"model", "train_time_sec"}]
        for col in metric_columns:
            val = best_model[col]
            if pd.notna(val):
                report_lines.append(f"- **{col}**: {val:.4f}")
        
        report_lines.append(f"- **Training Time**: {best_model['train_time_sec']:.3f} seconds")
        
        # Artifacts
        report_lines.append(f"\n### Artifacts\n")
        model_dir_name = best_model['model']
        model_dir_path = root / model_dir_name

        def add_artifact(filename: str, label: str) -> None:
            artifact_path = model_dir_path / filename
            if artifact_path.exists():
                report_lines.append(f"- **{label}**: `{model_dir_name}/{filename}`")

        add_artifact("model.joblib", "Model")
        add_artifact("metrics.json", "Metrics")
        add_artifact("classification_report.json", "Classification Report")
        add_artifact("confusion_matrix.png", "Confusion Matrix")
        add_artifact("residuals.png", "Residual Plot")
        add_artifact("feature_importance.csv", "Feature Importance (CSV)")
        add_artifact("feature_importance.png", "Feature Importance (Plot)")
    
    # Footer
    report_lines.append(f"\n---\n")
    report_lines.append(f"\n*All artifacts saved to: `{root.absolute()}`*")
    
    report_content = "\n".join(report_lines)
    
    # Save report
    with open(root / "report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_content
