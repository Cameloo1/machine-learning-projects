"""CLI to train and evaluate unsupervised anomaly detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_anomaly.anomaly_detection import (
    FEATURE_COLS,
    LABEL_COL,
    compute_feature_stats,
    evaluate_model,
    explain_top_anomalies,
    global_feature_correlations,
    isolation_forest_scores,
    load_dataset,
    oneclass_svm_scores,
    prepare_train_test,
    train_isolation_forest,
    train_oneclass_svm,
)
from soc_anomaly.config import DEFAULT_RANDOM_STATE
from soc_anomaly.visualization import (
    plot_confusion_matrix_heatmap,
    plot_feature_correlations,
    plot_model_comparison,
    plot_score_distribution,
    plot_top_anomalies_features,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the anomaly detection pipeline."""
    parser = argparse.ArgumentParser(description="Run IsolationForest and OneClassSVM on SOC alerts.")
    parser.add_argument("--data-path", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--k", type=int, default=50, help="k for precision@k.")
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Percentile for threshold-based classification.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used during splitting and modeling.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory to save visualization plots.",
    )
    return parser.parse_args()


def print_correlations(series, top_n: int = 8) -> None:
    """Pretty-print the top feature correlations."""
    print("Top feature correlations with anomaly score:")
    for feature, value in series.head(top_n).items():
        print(f"  {feature:<20} {value:+.3f}")


def print_local_explanations(explanations: list[dict[str, object]], max_items: int = 10) -> None:
    """Pretty-print local explanations."""
    print(f"\nTop {max_items} scored events with z-score explanations:")
    for entry in explanations[:max_items]:
        truth = "ANOMALY" if entry["true_label"] == 1 else "normal"
        print(f"  idx={entry['index']:>4} | score={entry['score']:.3f} | true={truth}")
        for exp in entry["explanations"]:
            print(f"    - {exp}")


def main() -> None:
    """Entry point for model training/evaluation."""
    args = parse_args()
    dataset_path = Path(args.data_path)
    df = load_dataset(str(dataset_path))

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        X_train_df,
        X_test_df,
    ) = prepare_train_test(
        df,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        test_size=0.2,
        random_state=args.random_state,
    )

    iso_model = train_isolation_forest(
        X_train_scaled,
        contamination=0.01,
        random_state=args.random_state,
    )
    iso_scores = isolation_forest_scores(iso_model, X_test_scaled)
    iso_metrics = evaluate_model(
        "IsolationForest",
        y_test,
        iso_scores,
        k=args.k,
        threshold_percentile=args.threshold_percentile,
    )

    ocsvm_model = train_oneclass_svm(
        X_train_scaled,
        nu=0.01,
        kernel="rbf",
        gamma="scale",
    )
    ocsvm_scores = oneclass_svm_scores(ocsvm_model, X_test_scaled)
    ocsvm_metrics = evaluate_model(
        "OneClassSVM",
        y_test,
        ocsvm_scores,
        k=args.k,
        threshold_percentile=args.threshold_percentile,
    )

    iso_corr = global_feature_correlations(
        X_test_df=X_test_df,
        y_test=y_test,
        scores=iso_scores,
        feature_cols=FEATURE_COLS,
        score_col_name="iso_score",
    )
    print()
    print_correlations(iso_corr)

    means, stds = compute_feature_stats(X_train_df, FEATURE_COLS)
    local_explanations = explain_top_anomalies(
        X_test_df=X_test_df,
        y_test=y_test,
        scores=iso_scores,
        means=means,
        stds=stds,
        feature_cols=FEATURE_COLS,
        top_m=10,
    )
    print_local_explanations(local_explanations, max_items=10)
    
    # Generate visualizations
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")
    
    # Feature correlations plot
    plot_feature_correlations(
        iso_corr,
        plots_dir / "iso_feature_corr.png",
        title="IsolationForest - Feature Correlations with Anomaly Score",
    )
    
    # Score distribution plots
    plot_score_distribution(
        iso_scores,
        y_test,
        plots_dir / "iso_score_hist.png",
        model_name="IsolationForest",
    )
    
    plot_score_distribution(
        ocsvm_scores,
        y_test,
        plots_dir / "ocsvm_score_hist.png",
        model_name="OneClassSVM",
    )
    
    # Confusion matrix heatmaps
    iso_cm = iso_metrics["confusion_matrix"]
    plot_confusion_matrix_heatmap(
        iso_cm["tn"],
        iso_cm["fp"],
        iso_cm["fn"],
        iso_cm["tp"],
        plots_dir / "iso_confusion_matrix.png",
        model_name="IsolationForest",
    )
    
    ocsvm_cm = ocsvm_metrics["confusion_matrix"]
    plot_confusion_matrix_heatmap(
        ocsvm_cm["tn"],
        ocsvm_cm["fp"],
        ocsvm_cm["fn"],
        ocsvm_cm["tp"],
        plots_dir / "ocsvm_confusion_matrix.png",
        model_name="OneClassSVM",
    )
    
    # Model comparison
    plot_model_comparison(
        iso_metrics,
        ocsvm_metrics,
        plots_dir / "model_comparison.png",
    )
    
    # Top anomalies feature analysis
    plot_top_anomalies_features(
        local_explanations,
        plots_dir / "top_anomalies_features.png",
        top_n=10,
    )
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to {plots_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

