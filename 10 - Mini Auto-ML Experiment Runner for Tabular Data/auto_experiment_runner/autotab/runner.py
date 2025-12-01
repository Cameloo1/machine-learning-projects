import time
import joblib
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from autotab.config import ExperimentConfig
from autotab.data import load_dataset, get_dataset_metadata, train_valid_split
from autotab.preprocessing import build_preprocessor, get_preprocessed_feature_names
from autotab.models import build_model, build_full_pipeline
from autotab.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_classification_report_dict,
    extract_feature_importance,
)
from autotab.reporting import (
    prepare_output_root,
    get_model_output_dir,
    save_json,
    save_leaderboard,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_residuals,
    save_dataset_metadata,
    generate_report,
)


@dataclass
class ExperimentResult:
    """Results from training a single model."""
    model_name: str
    metrics: dict
    train_time_sec: float


def run_experiment(config: ExperimentConfig) -> list[ExperimentResult]:
    """
    Run the full experiment: train all enabled models and collect results.
    
    Args:
        config: Experiment configuration
    
    Returns:
        List of ExperimentResult objects, one per enabled model
    """
    results = []
    
    # Load dataset and metadata
    dataset = load_dataset(config)
    metadata = get_dataset_metadata(dataset)
    
    # Perform train/validation split
    split_data = train_valid_split(dataset, config)
    
    # Build and fit preprocessor on training data
    preprocessor = build_preprocessor(
        config,
        dataset.numeric_features,
        dataset.categorical_features
    )
    preprocessor.fit(split_data.X_train)
    
    # Get transformed feature names for potential use
    feature_names = get_preprocessed_feature_names(preprocessor)
    
    # Loop over all enabled models
    enabled_models = [m for m in config.models if m.enabled]
    
    for model_cfg in enabled_models:
        # Build the estimator
        estimator = build_model(
            model_name=model_cfg.name,
            task_type=config.task.type,
            params=model_cfg.params
        )
        
        # Build full pipeline (preprocessor + model)
        pipeline = build_full_pipeline(preprocessor, estimator)
        
        # Time the training
        start_time = time.time()
        pipeline.fit(split_data.X_train, split_data.y_train)
        train_time_sec = time.time() - start_time
        
        # Make predictions
        y_pred = pipeline.predict(split_data.X_valid)
        
        # Get predicted probabilities for classification (if available)
        y_proba = None
        if config.task.type == "classification" and hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(split_data.X_valid)
        
        # Compute metrics
        metrics = compute_metrics(
            task_type=config.task.type,
            y_true=split_data.y_valid,
            y_pred=y_pred,
            y_proba=y_proba,
            primary_metric=config.evaluation.primary_metric,
            additional_metrics=config.evaluation.additional_metrics,
        )
        
        # Create result object
        result = ExperimentResult(
            model_name=model_cfg.name,
            metrics=metrics,
            train_time_sec=train_time_sec
        )
        results.append(result)
    
    return results


def build_leaderboard(
    results: list[ExperimentResult],
    primary_metric: str
) -> pd.DataFrame:
    """
    Build a leaderboard DataFrame from experiment results.
    
    Args:
        results: List of ExperimentResult objects
        primary_metric: Name of the primary metric to sort by
    
    Returns:
        DataFrame with columns: model, train_time_sec, and all metrics
        Sorted by primary_metric in descending order (higher is better)
    
    Note:
        Currently assumes all metrics are "higher is better".
        TODO: Handle metrics like RMSE/MAE where lower is better.
    """
    if not results:
        return pd.DataFrame()
    
    # Build list of dictionaries for DataFrame construction
    rows = []
    for result in results:
        row = {
            'model': result.model_name,
            'train_time_sec': result.train_time_sec,
        }
        # Add all metrics
        row.update(result.metrics)
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by primary metric (descending - higher is better)
    # TODO: Handle metrics where lower is better (rmse, mae)
    if primary_metric in df.columns:
        df = df.sort_values(by=primary_metric, ascending=False)
    
    # Reset index for clean display
    df = df.reset_index(drop=True)
    
    return df


def run_and_save_experiment(config: ExperimentConfig) -> tuple[list[ExperimentResult], Path]:
    """
    Run the full experiment and save all artifacts to disk.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Tuple of (results list, output root path)
    """
    # Prepare output directory
    root = prepare_output_root(config)
    
    # Load dataset and metadata
    dataset = load_dataset(config)
    metadata = get_dataset_metadata(dataset)
    
    # Save metadata
    save_dataset_metadata(metadata, root)
    
    # Perform train/validation split
    split_data = train_valid_split(dataset, config)
    
    # Build and fit preprocessor
    preprocessor = build_preprocessor(
        config,
        dataset.numeric_features,
        dataset.categorical_features
    )
    preprocessor.fit(split_data.X_train)
    feature_names = get_preprocessed_feature_names(preprocessor)
    
    # Train all models and save artifacts
    results = []
    enabled_models = [m for m in config.models if m.enabled]
    
    for model_cfg in enabled_models:
        # Build and train model
        estimator = build_model(
            model_name=model_cfg.name,
            task_type=config.task.type,
            params=model_cfg.params
        )
        pipeline = build_full_pipeline(preprocessor, estimator)
        
        start_time = time.time()
        pipeline.fit(split_data.X_train, split_data.y_train)
        train_time_sec = time.time() - start_time
        
        # Make predictions
        y_pred = pipeline.predict(split_data.X_valid)
        y_proba = None
        if config.task.type == "classification" and hasattr(pipeline, 'predict_proba'):
            y_proba = pipeline.predict_proba(split_data.X_valid)
        
        # Compute metrics
        metrics = compute_metrics(
            task_type=config.task.type,
            y_true=split_data.y_valid,
            y_pred=y_pred,
            y_proba=y_proba,
            primary_metric=config.evaluation.primary_metric,
            additional_metrics=config.evaluation.additional_metrics,
        )
        
        # Create model output directory
        model_dir = get_model_output_dir(root, model_cfg.name)
        
        # Save model
        if config.output.save_models:
            joblib.dump(pipeline, model_dir / "model.joblib")
        
        # Save metrics
        save_json(model_dir / "metrics.json", metrics)
        
        # Save classification-specific artifacts
        if config.task.type == "classification":
            # Confusion matrix
            if config.output.save_confusion_matrix:
                cm = compute_confusion_matrix(split_data.y_valid, y_pred)
                plot_confusion_matrix(cm, model_dir)
            
            # Classification report
            if config.output.save_classification_report:
                report_dict = compute_classification_report_dict(split_data.y_valid, y_pred)
                save_json(model_dir / "classification_report.json", report_dict)
        
        # Save regression-specific artifacts
        if config.task.type == "regression":
            if config.output.save_residual_plot:
                plot_residuals(split_data.y_valid, y_pred, model_dir)
        
        # Save feature importance
        if config.output.save_feature_importance:
            feature_importance = extract_feature_importance(pipeline, feature_names)
            if feature_importance is not None:
                # Save as CSV
                import pandas as pd
                fi_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
                fi_df.to_csv(model_dir / "feature_importance.csv", index=False)
                
                # Save plot
                if config.output.save_plots:
                    plot_feature_importance(feature_importance, model_dir)
        
        # Collect result
        result = ExperimentResult(
            model_name=model_cfg.name,
            metrics=metrics,
            train_time_sec=train_time_sec
        )
        results.append(result)
    
    # Build and save leaderboard
    leaderboard = build_leaderboard(results, config.evaluation.primary_metric)
    save_leaderboard(leaderboard, root)
    
    # Generate and save report
    generate_report(config, metadata, leaderboard, root)
    
    return results, root
