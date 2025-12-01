from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def build_model(
    model_name: str,
    task_type: str,
    params: dict | None = None
):
    """
    Build a scikit-learn or XGBoost model based on model_name and task_type.
    
    Args:
        model_name: Name of the model (e.g., 'logistic_regression', 'random_forest', 'xgboost')
        task_type: Type of task ('classification' or 'regression')
        params: Optional dictionary of hyperparameters to override defaults
    
    Returns:
        An instantiated sklearn/xgboost estimator
    
    Raises:
        ValueError: If the (model_name, task_type) combination is not supported
    """
    # Default parameters for each model
    defaults = {
        "logistic_regression": {"max_iter": 1000, "random_state": 42},
        "linear_regression": {},
        "random_forest": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "xgboost": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    }
    
    # Merge defaults with provided params
    if model_name not in defaults:
        raise ValueError(
            f"Unsupported model_name: '{model_name}'. "
            f"Supported models: {list(defaults.keys())}"
        )
    
    final_params = defaults[model_name].copy()
    if params:
        final_params.update(params)
    
    # Build the appropriate model based on task_type
    if task_type == "classification":
        if model_name == "logistic_regression":
            return LogisticRegression(**final_params)
        elif model_name == "random_forest":
            return RandomForestClassifier(**final_params)
        elif model_name == "xgboost":
            return XGBClassifier(**final_params)
        elif model_name == "linear_regression":
            raise ValueError(
                f"Model '{model_name}' is not supported for task_type '{task_type}'. "
                f"Use a classification model instead."
            )
    elif task_type == "regression":
        if model_name == "linear_regression":
            return LinearRegression(**final_params)
        elif model_name == "random_forest":
            return RandomForestRegressor(**final_params)
        elif model_name == "xgboost":
            return XGBRegressor(**final_params)
        elif model_name == "logistic_regression":
            raise ValueError(
                f"Model '{model_name}' is not supported for task_type '{task_type}'. "
                f"Use a regression model instead."
            )
    else:
        raise ValueError(
            f"Unsupported task_type: '{task_type}'. "
            f"Supported types: ['classification', 'regression']"
        )
    
    # If we reach here, it means the model_name is valid but not implemented for this task_type
    raise ValueError(
        f"Unsupported combination: model_name='{model_name}', task_type='{task_type}'"
    )


def build_full_pipeline(
    preprocessor: ColumnTransformer,
    estimator
) -> Pipeline:
    """
    Build a complete sklearn Pipeline combining preprocessing and model.
    
    Args:
        preprocessor: A fitted or unfitted ColumnTransformer for preprocessing
        estimator: A scikit-learn compatible estimator
    
    Returns:
        A Pipeline with two steps: preprocessor and model
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
