import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import json
import os
import logging
from typing import Any

logger = logging.getLogger("market_regime.eval")

def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    model_name: str,
    output_dir: str = "results",
    timestamps: np.ndarray = None
):
    """
    Evaluate model and save metrics/plots.
    Model can be Keras model or Sklearn model.
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predict
    if hasattr(model, "predict_proba"):
        # Sklearn RF
        y_probs = model.predict_proba(X_test)
        y_pred = np.argmax(y_probs, axis=1)
    elif hasattr(model, "predict"):
        # Keras or Sklearn
        # Check if it's a Keras model to add verbose=0
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
             y_probs = model.predict(X_test, verbose=0)
        else:
             y_probs = model.predict(X_test)
        # If Keras returns probs
        if y_probs.ndim > 1 and y_probs.shape[1] > 1:
            y_pred = np.argmax(y_probs, axis=1)
        else:
            # Should not happen with softmax but handle binary if needed
            y_pred = (y_probs > 0.5).astype(int).flatten()
            
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "class_metrics": {
            str(i): {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s)
            } for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support))
        }
    }
    
    # Save JSON
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f"{model_name}.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"cm_{model_name}.png"), bbox_inches='tight')
    plt.close()
    
    # Time Series Plot (Predicted vs True)
    if timestamps is not None:
        plt.figure(figsize=(12, 6))
        # Sort timestamps just in case
        indices = np.argsort(timestamps)
        ts_sorted = timestamps[indices]
        y_true_sorted = y_test[indices]
        y_pred_sorted = y_pred[indices]
        
        # We can't plot all points if too many, let's plot last 200 or similar if huge
        # But let's try plotting all with steps
        plt.plot(ts_sorted, y_true_sorted, label="True", alpha=0.6, linestyle='-', marker='.', markersize=2)
        plt.plot(ts_sorted, y_pred_sorted, label="Predicted", alpha=0.6, linestyle='--', marker='.', markersize=2)
        plt.title(f'Regime Prediction over Time - {model_name}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"preds_{model_name}.png"), bbox_inches='tight')
        plt.close()
        
    return metrics

def compare_models(metrics_dict: dict, output_dir: str = "results/plots"):
    """Compare models based on Macro F1."""
    models = list(metrics_dict.keys())
    f1_scores = [m['macro_f1'] for m in metrics_dict.values()]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=f1_scores)
    plt.title("Model Comparison - Macro F1")
    plt.ylim(0, 1.0)
    plt.ylabel("Macro F1 Score")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), bbox_inches='tight')
    plt.close()

