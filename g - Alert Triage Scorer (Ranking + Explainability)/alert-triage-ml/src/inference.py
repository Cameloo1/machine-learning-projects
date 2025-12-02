"""Batch and single record inference utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import shap

from . import config
from .explain import generate_explanation


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that all required feature columns exist."""

    missing = set(config.FEATURE_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")
    return df[config.FEATURE_COLUMNS].copy()


def to_dense(matrix):
    """Convert sparse matrices to dense arrays."""

    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def load_pipeline(model_path: Path):
    """Load a serialized pipeline."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def compute_explanations(
    pipeline,
    X_subset: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values for a subset of records."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    transformed = to_dense(preprocessor.transform(X_subset))
    return explainer.shap_values(transformed)


def score_csv(
    input_path: str,
    output_path: str,
    model_path: str,
    include_explanations: bool = False,
    n_explanations: int = 0,
) -> None:
    """Score an entire CSV file and optionally attach explanations."""

    pipeline = load_pipeline(Path(model_path))
    raw_df = pd.read_csv(input_path)
    feature_df = ensure_features(raw_df)
    predictions = pipeline.predict(feature_df)
    probabilities = pipeline.predict_proba(feature_df)

    result_df = raw_df.copy()
    result_df["predicted_priority"] = [config.LABEL_MAPPING[p] for p in predictions]
    result_df["prob_low"] = probabilities[:, 0]
    result_df["prob_medium"] = probabilities[:, 1]
    result_df["prob_high"] = probabilities[:, 2]

    if include_explanations and n_explanations > 0:
        subset = feature_df.head(n_explanations)
        shap_values = compute_explanations(pipeline, subset)
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = list(preprocessor.get_feature_names_out())

        explanations = []
        for idx in range(len(subset)):
            label = predictions[idx]
            shap_row = shap_values[label][idx]
            explanation = generate_explanation(
                original_row=subset.iloc[idx],
                shap_values_row=shap_row,
                feature_names=feature_names,
                predicted_label=label,
            )
            explanations.append(explanation)
        explanations += [""] * (len(result_df) - len(explanations))
        result_df["explanation"] = explanations

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved scored CSV with {len(result_df)} rows to {output_path}")


def score_single(
    alert_features: Dict[str, Any],
    model_path: str,
    include_explanation: bool = True,
) -> Dict[str, Any]:
    """Score a single alert and optionally include explanation details."""

    pipeline = load_pipeline(Path(model_path))
    data = pd.DataFrame([alert_features])
    feature_df = ensure_features(data)
    prediction = pipeline.predict(feature_df)[0]
    probabilities = pipeline.predict_proba(feature_df)[0]

    response: Dict[str, Any] = {
        "predicted_priority": prediction,
        "predicted_priority_label": config.LABEL_MAPPING[prediction],
        "probabilities": {
            "low": float(probabilities[0]),
            "medium": float(probabilities[1]),
            "high": float(probabilities[2]),
        },
    }

    if include_explanation:
        shap_values = compute_explanations(pipeline, feature_df)
        preprocessor = pipeline.named_steps["preprocessor"]
        explanation_text = generate_explanation(
            original_row=feature_df.iloc[0],
            shap_values_row=shap_values[prediction][0],
            feature_names=list(preprocessor.get_feature_names_out()),
            predicted_label=prediction,
        )
        response["explanation"] = explanation_text
    return response


def main() -> None:
    """Simple CLI wrapper for batch scoring."""

    parser = argparse.ArgumentParser(description="Alert triage inference CLI.")
    parser.add_argument("--mode", choices=["csv", "single"], default="csv")
    parser.add_argument("--input", help="Input CSV path for batch mode.")
    parser.add_argument("--output", help="Output CSV path for batch mode.")
    parser.add_argument("--model_path", default=str(config.MODELS_DIR / "xgb_pipeline.pkl"))
    args = parser.parse_args()

    if args.mode == "csv":
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for csv mode.")
        score_csv(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model_path,
            include_explanations=True,
            n_explanations=5,
        )
    else:
        raise NotImplementedError("Single mode CLI example not implemented. Use score_single() programmatically.")


if __name__ == "__main__":
    main()

