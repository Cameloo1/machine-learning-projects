"""Model explainability utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TOP_TOKENS_CSV_FILES, TOP_TOKENS_FILES


def get_feature_names(tfidf_vectorizer, ticker_encoder) -> np.ndarray:
    """Concatenate text + ticker feature names for consistent ordering."""
    text_features = tfidf_vectorizer.get_feature_names_out()
    ticker_features = ticker_encoder.get_feature_names_out(["ticker"])
    return np.concatenate([text_features, ticker_features])


def get_top_tokens_per_class(
    model,
    all_feature_names: np.ndarray,
    top_k: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """Return top contributing features per class."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Model must expose `classes_` attribute to extract tokens.")
    weights = getattr(model, "coef_", None)
    if weights is None:
        raise ValueError("Model must expose `coef_` attribute to extract tokens.")

    top_tokens: Dict[str, List[Tuple[str, float]]] = {}
    for idx, class_name in enumerate(classes):
        class_weights = weights[idx]
        top_indices = np.argsort(class_weights)[-top_k:][::-1]
        top_tokens[class_name] = [
            (all_feature_names[i], float(class_weights[i])) for i in top_indices
        ]
    return top_tokens


def top_tokens_to_dataframe(
    top_tokens: Dict[str, List[Tuple[str, float]]],
    model_name: str,
) -> pd.DataFrame:
    """Convert token dictionary to a tidy DataFrame."""
    rows = []
    for class_label, tokens in top_tokens.items():
        for feature, weight in tokens:
            rows.append(
                {"model_name": model_name, "class_label": class_label, "feature": feature, "weight": weight}
            )
    return pd.DataFrame(rows)


def save_top_tokens_plot(
    top_tokens: Dict[str, List[Tuple[str, float]]],
    model_name: str,
    figures_dir: str | Path,
) -> Path:
    """Save horizontal bar plots of top tokens per class."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(top_tokens)

    fig, axes = plt.subplots(n_classes, 1, figsize=(8, 3 * n_classes), sharex=True)
    if n_classes == 1:
        axes = [axes]  # type: ignore[List|ndarray]

    for ax, (class_label, tokens) in zip(axes, top_tokens.items()):
        features, weights = zip(*tokens)
        ax.barh(features[::-1], weights[::-1], color="#2b8cbe")
        ax.set_title(f"{class_label} – top signals")
        ax.set_xlabel("Coefficient weight")
        ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()

    fig_path = TOP_TOKENS_FILES.get(
        model_name,
        figures_dir / f"top_tokens_{model_name.lower().replace(' ', '')}.png",
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_top_tokens_csv(
    top_tokens: Dict[str, List[Tuple[str, float]]],
    model_name: str,
    reports_dir: str | Path,
) -> Path:
    """Save the token table to CSV."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = top_tokens_to_dataframe(top_tokens, model_name)
    csv_path = TOP_TOKENS_CSV_FILES.get(
        model_name,
        reports_dir / f"top_tokens_{model_name.lower().replace(' ', '')}.csv",
    )
    df.to_csv(csv_path, index=False)
    return csv_path



