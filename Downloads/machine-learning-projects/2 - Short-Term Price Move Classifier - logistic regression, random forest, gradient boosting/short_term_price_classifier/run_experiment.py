"""Command-line entry point for running the full experiment."""

from __future__ import annotations

from typing import Dict, Optional

from .config import DATA_START, DEFAULT_TICKER, RANDOM_STATE, TARGET_COLUMN, TRAIN_FRACTION
from .data_loader import load_ohlcv
from .evaluation import (
    evaluate_classifier,
    plot_feature_importances,
    plot_logistic_coefficients,
    plot_roc_curve,
    print_evaluation_report,
)
from .features import build_feature_dataset
from .modeling import (
    compute_baseline_predictions,
    time_based_train_test_split,
    train_gradient_boosting,
    train_logistic_regression,
    train_random_forest,
)


def summarize_metrics(summary: Dict[str, Dict[str, Optional[float]]]) -> None:
    """Print a compact summary table."""

    print("\nOverall Comparison")
    print("------------------")
    header = f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'ROC AUC':>8}"
    print(header)
    print("-" * len(header))

    def _fmt(value: Optional[float], width: int) -> str:
        return f"{value:>{width}.3f}" if value is not None else f"{'N/A':>{width}}"

    for name, metrics in summary.items():
        print(
            f"{name:<20} "
            f"{_fmt(metrics.get('accuracy'), 9)} "
            f"{_fmt(metrics.get('precision'), 10)} "
            f"{_fmt(metrics.get('recall'), 8)} "
            f"{_fmt(metrics.get('roc_auc'), 8)}"
        )


def main() -> None:
    """Execute the end-to-end experiment."""

    print("Downloading data...")
    ohlcv = load_ohlcv(DEFAULT_TICKER, DATA_START)
    print(f"Loaded {len(ohlcv)} rows of data for {DEFAULT_TICKER}.")

    feature_df, feature_cols = build_feature_dataset(ohlcv)
    if feature_df.empty:
        raise RuntimeError("Feature dataset is empty after engineering.")

    print(f"Feature rows available after dropping NaNs: {len(feature_df)}")

    X_train, X_test, y_train, y_test = time_based_train_test_split(
        df=feature_df,
        feature_cols=feature_cols,
        target_col=TARGET_COLUMN,
        train_fraction=TRAIN_FRACTION,
    )

    if len(X_test) == 0:
        raise RuntimeError("Test set is empty. Adjust TRAIN_FRACTION or gather more data.")

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    metrics_summary: Dict[str, Dict[str, Optional[float]]] = {}

    # Baseline
    y_pred_baseline = compute_baseline_predictions(y_train, y_test)
    baseline_metrics = evaluate_classifier(y_test, y_pred_baseline)
    print_evaluation_report("Baseline (Majority Class)", baseline_metrics)
    metrics_summary["Baseline"] = baseline_metrics

    # Logistic Regression
    log_reg, scaler = train_logistic_regression(X_train, y_train, RANDOM_STATE)
    X_test_scaled = scaler.transform(X_test)
    log_pred = log_reg.predict(X_test_scaled)
    log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    log_metrics = evaluate_classifier(y_test, log_pred, log_proba)
    print_evaluation_report("Logistic Regression", log_metrics)
    metrics_summary["Logistic Regression"] = log_metrics
    plot_roc_curve(y_test, log_proba, "Logistic Regression ROC")
    plot_logistic_coefficients(feature_cols, log_reg.coef_[0], "Logistic Regression Coefficients")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train, RANDOM_STATE)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_classifier(y_test, rf_pred, rf_proba)
    print_evaluation_report("Random Forest", rf_metrics)
    metrics_summary["Random Forest"] = rf_metrics
    plot_roc_curve(y_test, rf_proba, "Random Forest ROC")
    plot_feature_importances(feature_cols, rf_model.feature_importances_, "Random Forest Feature Importance")

    # Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train, RANDOM_STATE)
    gb_pred = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]
    gb_metrics = evaluate_classifier(y_test, gb_pred, gb_proba)
    print_evaluation_report("Gradient Boosting", gb_metrics)
    metrics_summary["Gradient Boosting"] = gb_metrics
    plot_roc_curve(y_test, gb_proba, "Gradient Boosting ROC")
    plot_feature_importances(
        feature_cols,
        gb_model.feature_importances_,
        "Gradient Boosting Feature Importance",
    )

    summarize_metrics(metrics_summary)


if __name__ == "__main__":
    main()

