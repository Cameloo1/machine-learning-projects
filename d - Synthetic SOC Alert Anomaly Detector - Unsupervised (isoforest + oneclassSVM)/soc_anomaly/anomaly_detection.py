"""Unsupervised anomaly detection utilities for SOC events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .config import DEFAULT_RANDOM_STATE
from .utils import ensure_columns

FEATURE_COLS: list[str] = [
    "hour",
    "login_success",
    "bytes_out",
    "bytes_in",
    "geo_distance",
    "failed_logins_24h",
    "device_trust_score",
    "user_risk_score",
]
LABEL_COL: str = "is_anomaly"


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def prepare_train_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """Prepare scaled train/test splits using only normal data for training."""
    ensure_columns(df, feature_cols + [label_col])

    X = df[feature_cols].copy()
    y = df[label_col].astype(int).to_numpy()

    normal_mask = y == 0
    if normal_mask.sum() == 0:
        raise ValueError("Dataset does not contain any normal events for training.")

    X_normal = X[normal_mask]
    X_anomalies = X[~normal_mask]
    y_anomalies = y[~normal_mask]

    X_train_df, X_test_normal_df = train_test_split(
        X_normal,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_test_df = pd.concat([X_test_normal_df, X_anomalies], axis=0, ignore_index=True)
    y_test = np.concatenate(
        [
            np.zeros(len(X_test_normal_df), dtype=int),
            y_anomalies,
        ]
    )

    test_shuffle = pd.DataFrame({"__y": y_test})
    X_test_df = pd.concat([X_test_df, test_shuffle], axis=1)
    X_test_df = X_test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_test = X_test_df.pop("__y").to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    y_train = np.zeros(len(X_train_df), dtype=int)
    X_train_df = X_train_df.reset_index(drop=True)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_df, X_test_df


def train_isolation_forest(
    X_train_scaled: np.ndarray,
    contamination: float = 0.01,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> IsolationForest:
    """Train an IsolationForest model."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)
    return model


def isolation_forest_scores(model: IsolationForest, X_scaled: np.ndarray) -> np.ndarray:
    """Return IsolationForest anomaly scores (higher is more anomalous)."""
    return -model.decision_function(X_scaled)


def train_oneclass_svm(
    X_train_scaled: np.ndarray,
    nu: float = 0.01,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> OneClassSVM:
    """Train a OneClassSVM model."""
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_train_scaled)
    return model


def oneclass_svm_scores(model: OneClassSVM, X_scaled: np.ndarray) -> np.ndarray:
    """Return OneClassSVM anomaly scores (higher is more anomalous)."""
    return -model.decision_function(X_scaled)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute precision among the top-k scored events."""
    if len(scores) == 0 or k <= 0:
        return 0.0
    k = min(k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]
    return float(np.sum(y_true[top_indices]) / k)


def binarize_by_percentile(scores: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    """Binarize scores using a percentile-based threshold."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100].")
    threshold = float(np.percentile(scores, percentile))
    binary = (scores >= threshold).astype(int)
    return binary, threshold


def evaluate_model(
    name: str,
    y_test: np.ndarray,
    scores: np.ndarray,
    k: int = 50,
    threshold_percentile: float = 99.0,
) -> dict[str, Any]:
    """Evaluate an anomaly detector and print summary metrics."""
    metrics: dict[str, Any] = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["precision_at_k"] = precision_at_k(y_test, scores, k)
    preds, threshold = binarize_by_percentile(scores, threshold_percentile)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics.update(
        {
            "threshold": threshold,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    )

    print(
        f"[{name}] ROC AUC: {metrics['roc_auc']:.3f} | "
        f"Precision@{k}: {metrics['precision_at_k']:.3f} | "
        f"Threshold (p{threshold_percentile:.1f}): {threshold:.3f}"
    )
    print(f"[{name}] Confusion Matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    return metrics


def global_feature_correlations(
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    scores: np.ndarray,
    feature_cols: list[str],
    score_col_name: str,
) -> pd.Series:
    """Compute correlations between features and anomaly scores."""
    ensure_columns(X_test_df, feature_cols)
    if len(X_test_df) != len(y_test):
        raise ValueError("X_test_df and y_test lengths must match.")
    corr_df = X_test_df.copy()
    corr_df[score_col_name] = scores
    correlations = corr_df.corr(numeric_only=True)[score_col_name].drop(labels=[score_col_name])
    correlations = correlations.sort_values(ascending=False)
    return correlations


def compute_feature_stats(
    X_train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute feature-wise mean and std statistics."""
    ensure_columns(X_train_df, feature_cols)
    means = X_train_df[feature_cols].mean()
    stds = X_train_df[feature_cols].std(ddof=0)
    stds = stds.replace(0, 1e-6)
    return means, stds


def explain_event_zscore(
    event_row: pd.Series,
    means: pd.Series,
    stds: pd.Series,
    feature_cols: list[str],
    top_n: int = 3,
) -> list[str]:
    """Create human-readable explanations using z-scores."""
    explanations: list[str] = []
    z_scores = []
    for feature in feature_cols:
        if feature not in event_row:
            continue
        value = event_row[feature]
        mean = means.get(feature, 0.0)
        std = stds.get(feature, 1.0)
        if std == 0:
            z = 0.0
        else:
            z = (value - mean) / std
        z_scores.append((feature, z))
    top_features = sorted(z_scores, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    for feature, z in top_features:
        explanations.append(f"{feature} is {z:+.2f} std from normal")
    return explanations


def explain_top_anomalies(
    X_test_df: pd.DataFrame,
    y_test: np.ndarray,
    scores: np.ndarray,
    means: pd.Series,
    stds: pd.Series,
    feature_cols: list[str],
    top_m: int = 10,
) -> list[dict[str, Any]]:
    """Generate explanations for the highest-scoring anomalies."""
    ensure_columns(X_test_df, feature_cols)
    top_m = min(top_m, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_m]
    explanations: list[dict[str, Any]] = []
    for idx in top_indices:
        row = X_test_df.iloc[idx]
        explanations.append(
            {
                "index": int(idx),
                "true_label": int(y_test[idx]),
                "score": float(scores[idx]),
                "explanations": explain_event_zscore(row, means, stds, feature_cols),
            }
        )
    return explanations

