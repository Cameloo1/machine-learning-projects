"""Explainability utilities leveraging SHAP."""

from __future__ import annotations

import argparse
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from . import config
from .utils_io import load_dataframe, load_json, save_dataframe


def simplify_feature_name(feature_name: str) -> str:
    """Strip ColumnTransformer prefixes to get the base feature name."""

    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def feature_reason(feature_name: str, shap_value: float, row: pd.Series) -> str:
    """Map feature contributions to domain-friendly phrases."""

    base = simplify_feature_name(feature_name)
    positive = shap_value > 0
    base_value = row.get(base.split("_", 1)[0], None)

    if "user_risk_score" in base:
        return "user risk score is elevated" if positive else "user risk context is low"
    if "asset_criticality" in base:
        return "involves a highly critical asset" if positive else "asset appears low criticality"
    if "detection_confidence" in base:
        return "detection confidence is high" if positive else "detection confidence is weak"
    if "rule_historical_fpr" in base:
        return "rule is historically precise" if positive else "linked to a noisy rule"
    if "failed_login_ratio" in base:
        return "failed login ratio is unusual" if positive else "few failed logins observed"
    if "geo_distance_km" in base:
        return "activity occurs far from usual geo" if positive else "geo distance is normal"
    if "event_count_24h" in base:
        return "event volume spike detected" if positive else "event volume is calm"
    if "rule_severity" in base:
        return "rule severity is high" if positive else "rule severity is mild"
    if "is_known_fp_source" in base:
        return "originates from known noisy source" if not positive else "source is typically trustworthy"
    if "alert_type_" in base:
        alert = base.replace("alert_type_", "")
        return f"alert pattern '{alert}' signals elevated risk" if positive else f"alert pattern '{alert}' is usually benign"
    if "kill_chain_stage_" in base:
        stage = base.replace("kill_chain_stage_", "")
        return f"kill chain stage '{stage}' is critical" if positive else f"kill chain stage '{stage}' lessens urgency"
    if "hour_of_day" in base:
        return "occurs during off-hours" if positive else "happens at routine hours"

    return f"feature {base} influences the assessment"


def generate_explanation(
    original_row: pd.Series,
    shap_values_row: np.ndarray,
    feature_names: List[str],
    predicted_label: int,
    top_n: int = 3,
) -> str:
    """Generate a concise, human-readable explanation string."""

    label_text = config.LABEL_MAPPING.get(predicted_label, str(predicted_label))
    indices = np.argsort(np.abs(shap_values_row))[::-1][:top_n]
    phrases = []
    for idx in indices:
        phrase = feature_reason(feature_names[idx], shap_values_row[idx], original_row)
        if phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        phrases.append("model evidence was inconclusive")
    explanation = f"Priority {label_text} because " + ", ".join(phrases) + "."
    return explanation


def pick_best_model() -> str:
    """Determine the best-performing model based on macro F1."""

    scores = {}
    for name in ("xgb", "lgbm"):
        metrics_path = config.METRICS_DIR / f"{name}_metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            scores[name] = metrics["macro_f1"]
    if not scores:
        raise FileNotFoundError("No metrics files found. Run evaluation first.")
    best_model = max(scores.items(), key=lambda item: item[1])[0]
    print(f"Selected {best_model} as best model based on macro F1.")
    return best_model


def to_dense(matrix):
    """Convert sparse matrices to dense arrays."""

    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def main(sample_count: int = 200) -> None:
    """Generate SHAP explainability artifacts."""

    config.ensure_directories()
    best_model = pick_best_model()
    pipeline = joblib.load(config.MODELS_DIR / f"{best_model}_pipeline.pkl")

    train_df = load_dataframe(config.PROCESSED_DIR / "train.csv")
    test_df = load_dataframe(config.PROCESSED_DIR / "test.csv")

    X_train = train_df[config.FEATURE_COLUMNS]
    X_test = test_df[config.FEATURE_COLUMNS]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    background_df = X_train.sample(n=min(500, len(X_train)), random_state=config.RANDOM_STATE)
    explain_df = X_test.sample(n=min(sample_count, len(X_test)), random_state=config.RANDOM_STATE)

    background = to_dense(preprocessor.transform(background_df))
    explain_data = to_dense(preprocessor.transform(explain_df))
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(explain_data)

    class_index = list(config.LABEL_MAPPING.keys()).index(2)
    shap.summary_plot(
        shap_values[class_index],
        features=explain_data,
        feature_names=feature_names,
        show=False,
    )
    config.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.SHAP_DIR / "global_summary.png", bbox_inches="tight")
    plt.close()

    local_idx = 0
    local_values = shap_values[class_index][local_idx]
    ordered_idx = np.argsort(np.abs(local_values))[::-1][:15]
    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in ordered_idx][::-1],
        local_values[ordered_idx][::-1],
        color="steelblue",
    )
    plt.xlabel("SHAP value (impact on model output)")
    plt.title("Local explanation – High priority class")
    plt.tight_layout()
    plt.savefig(config.SHAP_DIR / "local_example_bar.png", bbox_inches="tight")
    plt.close()

    predictions = pipeline.predict(explain_df)
    probabilities = pipeline.predict_proba(explain_df)

    explanations = []
    for i, row in enumerate(explain_df.itertuples(index=False)):
        row_series = pd.Series(row._asdict())
        predicted_label = predictions[i]
        shap_row = shap_values[predicted_label][i]
        text = generate_explanation(
            original_row=row_series,
            shap_values_row=shap_row,
            feature_names=list(feature_names),
            predicted_label=predicted_label,
        )
        explanations.append(
            {
                **row_series.to_dict(),
                "predicted_priority": config.LABEL_MAPPING[predicted_label],
                "prob_low": probabilities[i][0],
                "prob_medium": probabilities[i][1],
                "prob_high": probabilities[i][2],
                "explanation": text,
            }
        )

    explanations_df = pd.DataFrame(explanations[:10])
    save_dataframe(explanations_df, config.EXPLANATIONS_PATH)
    print(f"Saved explanations sample to {config.EXPLANATIONS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--sample_count", type=int, default=200, help="Number of samples to explain.")
    args = parser.parse_args()
    main(sample_count=args.sample_count)

