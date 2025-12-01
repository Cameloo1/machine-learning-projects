import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def compute_metrics(
    task_type: str,
    y_true,
    y_pred,
    y_proba,
    primary_metric: str,
    additional_metrics: list[str],
) -> dict:
    """
    Compute metrics for classification or regression tasks.
    
    Args:
        task_type: 'classification' or 'regression'
        y_true: True labels/values
        y_pred: Predicted labels/values
        y_proba: Predicted probabilities (for classification, can be None)
        primary_metric: Name of the primary metric
        additional_metrics: List of additional metric names
    
    Returns:
        Dictionary with metric names as keys and computed values as values
    
    Supported metrics:
        Classification: accuracy, f1_macro, roc_auc
        Regression: rmse, mae, r2
    """
    metrics = {}
    all_metrics = [primary_metric] + additional_metrics
    
    if task_type == "classification":
        for metric_name in all_metrics:
            if metric_name == "accuracy":
                metrics[metric_name] = accuracy_score(y_true, y_pred)
            elif metric_name == "f1_macro":
                metrics[metric_name] = f1_score(y_true, y_pred, average='macro')
            elif metric_name == "roc_auc":
                if y_proba is not None:
                    # Handle binary and multiclass cases
                    try:
                        if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                            # Binary classification
                            if y_proba.ndim == 2:
                                y_proba_binary = y_proba[:, 1]
                            else:
                                y_proba_binary = y_proba
                            metrics[metric_name] = roc_auc_score(y_true, y_proba_binary)
                        else:
                            # Multiclass
                            metrics[metric_name] = roc_auc_score(
                                y_true, y_proba, multi_class='ovr', average='macro'
                            )
                    except Exception as e:
                        metrics[metric_name] = None
                else:
                    metrics[metric_name] = None
            else:
                # Unknown metric, skip
                pass
    
    elif task_type == "regression":
        for metric_name in all_metrics:
            if metric_name == "rmse":
                metrics[metric_name] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric_name == "mae":
                metrics[metric_name] = mean_absolute_error(y_true, y_pred)
            elif metric_name == "r2":
                metrics[metric_name] = r2_score(y_true, y_pred)
            else:
                # Unknown metric, skip
                pass
    
    return metrics


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for classification tasks.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix as numpy array with shape (n_classes, n_classes)
    """
    return confusion_matrix(y_true, y_pred)


def compute_classification_report_dict(y_true, y_pred):
    """
    Compute classification report as a dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with per-class metrics and overall metrics
    """
    return classification_report(y_true, y_pred, output_dict=True)


def compute_residuals(y_true, y_pred):
    """
    Compute residuals for regression tasks.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Residuals as numpy array (y_true - y_pred)
    """
    return np.array(y_true) - np.array(y_pred)


def extract_feature_importance(
    pipeline,
    transformed_feature_names: list[str],
):
    """
    Extract feature importances from a fitted pipeline.
    
    Args:
        pipeline: Fitted sklearn Pipeline with 'model' step
        transformed_feature_names: List of feature names after preprocessing
    
    Returns:
        List of (feature_name, importance_value) tuples sorted by importance,
        or None if the model doesn't support feature importance
    
    Supports:
        - Tree-based models with feature_importances_ attribute
        - Linear models with coef_ attribute
    """
    # Get the model from the pipeline
    if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
        model = pipeline.named_steps['model']
    else:
        # If not a pipeline, assume it's the model itself
        model = pipeline
    
    # Try to extract feature importances
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (RandomForest, XGBoost, etc.)
        importances = model.feature_importances_
        feature_importance_list = list(zip(transformed_feature_names, importances))
        # Sort by importance (descending)
        feature_importance_list.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_importance_list
    
    elif hasattr(model, 'coef_'):
        # Linear models (LogisticRegression, LinearRegression, etc.)
        coef = model.coef_
        
        # Flatten if needed (e.g., for multiclass logistic regression)
        if coef.ndim > 1:
            # For multiclass, take the mean absolute coefficient across classes
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
        
        feature_importance_list = list(zip(transformed_feature_names, importances))
        # Sort by importance (descending)
        feature_importance_list.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_importance_list
    
    else:
        # Model doesn't support feature importance
        return None
