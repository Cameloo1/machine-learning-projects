"""Utility script to run the full training/evaluation pipeline end-to-end."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CLASS_LABELS,
    FIGURES_DIR,
    LABELED_DATA_PATH,
    METRICS_COMPARISON_FIG,
    MODEL_ARTIFACT_PATHS,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_MD,
    PER_CLASS_FIG,
    REPORTS_DIR,
)
from src.explain import (  # noqa: E402
    get_feature_names,
    get_top_tokens_per_class,
    save_top_tokens_csv,
    save_top_tokens_plot,
)
from src.preprocess import (  # noqa: E402
    build_vectorizers,
    clean_text,
    load_labeled_data,
    transform_features,
)
from src.train import (  # noqa: E402
    evaluate_model,
    train_linear_svc,
    train_logistic_regression,
)


def describe_confusion(cm: np.ndarray) -> tuple[str, str, int]:
    """Return the most common off-diagonal confusion."""
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    idx = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    return CLASS_LABELS[idx[0]], CLASS_LABELS[idx[1]], int(cm_copy[idx])


def plot_metric_comparison(metrics_df: pd.DataFrame, output_path: Path) -> Path:
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    plot_df = metrics_df.melt(id_vars="model_name", value_vars=metrics, var_name="metric", value_name="score")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model_name")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title("Model Metric Comparison")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_per_class_f1(report_map: dict[str, dict], output_path: Path) -> Path:
    rows = []
    for model_name, report in report_map.items():
        for cls in CLASS_LABELS:
            rows.append(
                {
                    "model_name": model_name,
                    "class": cls,
                    "f1": report.get(cls, {}).get("f1-score", 0.0),
                }
            )
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="class", y="f1", hue="model_name")
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.xlabel("Sentiment class")
    plt.title("Per-class F1 comparison")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def main() -> None:
    sns.set_theme(style="whitegrid")
    print(f"[pipeline] Loading labeled data from {LABELED_DATA_PATH}")
    df = load_labeled_data()
    print(f"[pipeline] Rows: {len(df)}")

    tfidf_vectorizer, ticker_encoder = build_vectorizers(df)
    X, y = transform_features(df, tfidf_vectorizer, ticker_encoder)
    print(f"[pipeline] Feature matrix: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logreg_model = train_logistic_regression(X_train, y_train)
    logreg_metrics = evaluate_model(
        logreg_model,
        X_train,
        y_train,
        X_test,
        y_test,
        "Logistic Regression",
        FIGURES_DIR,
    )

    linearsvc_model = train_linear_svc(X_train, y_train)
    linearsvc_metrics = evaluate_model(
        linearsvc_model,
        X_train,
        y_train,
        X_test,
        y_test,
        "Linear SVC",
        FIGURES_DIR,
    )

    feature_names = get_feature_names(tfidf_vectorizer, ticker_encoder)
    logreg_top = get_top_tokens_per_class(logreg_model, feature_names, top_k=15)
    linearsvc_top = get_top_tokens_per_class(linearsvc_model, feature_names, top_k=15)

    logreg_plot = save_top_tokens_plot(logreg_top, "Logistic Regression", FIGURES_DIR)
    linearsvc_plot = save_top_tokens_plot(linearsvc_top, "Linear SVC", FIGURES_DIR)
    logreg_csv = save_top_tokens_csv(logreg_top, "Logistic Regression", REPORTS_DIR)
    linearsvc_csv = save_top_tokens_csv(linearsvc_top, "Linear SVC", REPORTS_DIR)

    metrics_df = pd.DataFrame([logreg_metrics, linearsvc_metrics])
    metrics_df.to_csv(MODEL_COMPARISON_CSV, index=False)

    logreg_preds = logreg_model.predict(X_test)
    linearsvc_preds = linearsvc_model.predict(X_test)
    logreg_cm = confusion_matrix(y_test, logreg_preds, labels=CLASS_LABELS)
    linearsvc_cm = confusion_matrix(y_test, linearsvc_preds, labels=CLASS_LABELS)
    logreg_report = classification_report(
        y_test,
        logreg_preds,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    linearsvc_report = classification_report(
        y_test,
        linearsvc_preds,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    best_macro = metrics_df.loc[metrics_df["macro_f1"].idxmax()]
    best_weighted = metrics_df.loc[metrics_df["weighted_f1"].idxmax()]
    summary_lines = [
        f"- Macro F1 leader: {best_macro.model_name} ({best_macro.macro_f1:.3f}).",
        f"- Weighted F1 leader: {best_weighted.model_name} ({best_weighted.weighted_f1:.3f}).",
        f"- Largest confusion (LogReg): {describe_confusion(logreg_cm)}.",
        f"- Largest confusion (Linear SVC): {describe_confusion(linearsvc_cm)}.",
        "- Both models remain interpretable thanks to linear weights over TF-IDF + ticker features.",
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    def to_md_table(df: pd.DataFrame) -> str:
        headers = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = []
        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if isinstance(val, float):
                    cells.append(f"{val:.3f}")
                else:
                    cells.append(str(val))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([headers, separator, *rows])

    with open(MODEL_COMPARISON_MD, "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write(to_md_table(metrics_df) + "\n\n")
        f.write("## Highlights\n")
        for line in summary_lines:
            f.write(line + "\n")

    models_dir = MODEL_ARTIFACT_PATHS["logreg_model"].parent
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(logreg_model, MODEL_ARTIFACT_PATHS["logreg_model"])
    joblib.dump(linearsvc_model, MODEL_ARTIFACT_PATHS["linearsvc_model"])
    joblib.dump(tfidf_vectorizer, MODEL_ARTIFACT_PATHS["tfidf_vectorizer"])
    joblib.dump(ticker_encoder, MODEL_ARTIFACT_PATHS["ticker_encoder"])

    metrics_fig = plot_metric_comparison(metrics_df, METRICS_COMPARISON_FIG)
    per_class_fig = plot_per_class_f1(
        {"Logistic Regression": logreg_report, "Linear SVC": linearsvc_report},
        PER_CLASS_FIG,
    )

    print("[pipeline] Saved artifacts to", models_dir)
    print("[pipeline] Top token plots:", logreg_plot, linearsvc_plot)
    print("[pipeline] Top token CSVs:", logreg_csv, linearsvc_csv)
    print("[pipeline] Model comparison:", MODEL_COMPARISON_CSV, MODEL_COMPARISON_MD)
    print("[pipeline] Metric comparison figure:", metrics_fig)
    print("[pipeline] Per-class F1 figure:", per_class_fig)

    # Quick inference demo
    sample_inputs = [
        ("Apple shares jump after crushing earnings and boosting dividends", "AAPL"),
        ("Tesla faces recall and regulatory probe over autopilot crashes", "TSLA"),
        ("Microsoft trades sideways ahead of mixed macro data", "MSFT"),
    ]
    best_model_name = (
        best_macro["model_name"] if best_macro["macro_f1"] >= best_weighted["macro_f1"] else best_weighted["model_name"]
    )
    prod_model = logreg_model if best_model_name == "Logistic Regression" else linearsvc_model
    print(f"[pipeline] Using {best_model_name} for inference demo.")
    for headline, ticker in sample_inputs:
        cleaned = clean_text(headline)
        text_vec = tfidf_vectorizer.transform([cleaned])
        ticker_vec = ticker_encoder.transform([[ticker.upper()]])
        from scipy import sparse

        X_sample = sparse.hstack([text_vec, sparse.csr_matrix(ticker_vec)])
        pred = prod_model.predict(X_sample)[0]
        print(f"  {ticker} | {headline} -> {pred}")


if __name__ == "__main__":
    main()


