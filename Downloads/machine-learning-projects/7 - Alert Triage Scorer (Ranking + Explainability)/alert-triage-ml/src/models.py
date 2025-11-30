"""Model builders for the alert triage pipelines."""

from __future__ import annotations

from typing import Any, Dict

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config


def build_xgb_classifier(**overrides: Dict[str, Any]) -> XGBClassifier:
    """Instantiate a configured XGBoost classifier."""

    params: Dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": config.RANDOM_STATE,
        "n_jobs": config.N_JOBS,
        "use_label_encoder": False,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def build_lgbm_classifier(**overrides: Dict[str, Any]) -> LGBMClassifier:
    """Instantiate a configured LightGBM classifier."""

    params: Dict[str, Any] = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multiclass",
        "num_class": len(config.LABEL_MAPPING),
        "random_state": config.RANDOM_STATE,
        "n_jobs": config.N_JOBS,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


def build_pipeline_xgb(preprocessor: ColumnTransformer) -> Pipeline:
    """Create the end-to-end XGB pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_xgb_classifier()),
        ]
    )


def build_pipeline_lgbm(preprocessor: ColumnTransformer) -> Pipeline:
    """Create the end-to-end LGBM pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_lgbm_classifier()),
        ]
    )

