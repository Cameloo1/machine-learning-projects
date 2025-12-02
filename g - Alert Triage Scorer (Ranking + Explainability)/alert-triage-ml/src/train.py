"""Training script for the alert triage models."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config
from .data_loading import load_raw_alerts, save_processed_splits, train_val_test_split
from .models import build_pipeline_lgbm, build_pipeline_xgb
from .preprocessing import build_preprocessor


macro_f1 = make_scorer(f1_score, average="macro")


def get_search_spaces() -> Dict[str, Dict[str, list]]:
    """Define hyperparameter search spaces for each model."""

    return {
        "xgb": {
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__gamma": [0.0, 0.3, 0.7],
        },
        "lgbm": {
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 0.95, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__min_child_samples": [10, 20, 40],
        },
    }


def prepare_splits(random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data and perform train/val/test split."""

    df = load_raw_alerts(config.RAW_DATA_PATH)
    train_df, val_df, test_df = train_val_test_split(
        df,
        random_state=random_state,
    )
    save_processed_splits(train_df, val_df, test_df)
    return train_df, val_df, test_df


def fit_model(
    name: str,
    pipeline,
    X_train,
    y_train,
    search_space,
    random_state: int,
    search_iterations: int,
    cv_folds: int,
):
    """Run randomized search and return the best estimator."""

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=search_iterations,
        scoring=macro_f1,
        cv=cv,
        n_jobs=config.N_JOBS,
        verbose=1,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"{name} best score: {search.best_score_:.4f}")
    print(f"{name} best params: {search.best_params_}")
    return search.best_estimator_, search.best_score_, search.best_params_


def main(
    random_state: int = config.RANDOM_STATE,
    search_iterations: int = 15,
    cv_folds: int = config.CV_FOLDS,
) -> None:
    """Entry point for training both models."""

    config.ensure_directories()
    train_df, val_df, test_df = prepare_splits(random_state)

    X_train = train_df[config.FEATURE_COLUMNS]
    y_train = train_df[config.TARGET_COLUMN]
    X_val = val_df[config.FEATURE_COLUMNS]
    y_val = val_df[config.TARGET_COLUMN]
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    base_preprocessor = build_preprocessor()
    pipelines = {
        "xgb": build_pipeline_xgb(preprocessor=clone(base_preprocessor)),
        "lgbm": build_pipeline_lgbm(preprocessor=clone(base_preprocessor)),
    }
    search_spaces = get_search_spaces()

    trained_models = {}
    for name, pipeline in pipelines.items():
        estimator, score, params = fit_model(
            name=name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            search_space=search_spaces[name],
            random_state=random_state,
            search_iterations=search_iterations,
            cv_folds=cv_folds,
        )
        estimator.fit(X_train_val, y_train_val)
        trained_models[name] = estimator
        print(f"{name} refitted on train+val. Validation size {X_val.shape[0]}")

    model_paths = {
        "xgb": config.MODELS_DIR / "xgb_pipeline.pkl",
        "lgbm": config.MODELS_DIR / "lgbm_pipeline.pkl",
    }
    for name, estimator in trained_models.items():
        joblib.dump(estimator, model_paths[name])
        print(f"Saved {name} pipeline to {model_paths[name]}")

    print(f"Test set reserved with shape {test_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gradient boosting alert triage models.")
    parser.add_argument("--random_state", type=int, default=config.RANDOM_STATE, help="Random seed.")
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=15,
        help="RandomizedSearch iterations per model (lower for faster CI runs).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=config.CV_FOLDS,
        help="Number of CV folds used during hyperparameter search.",
    )
    args = parser.parse_args()
    main(
        random_state=args.random_state,
        search_iterations=args.search_iterations,
        cv_folds=args.cv_folds,
    )

