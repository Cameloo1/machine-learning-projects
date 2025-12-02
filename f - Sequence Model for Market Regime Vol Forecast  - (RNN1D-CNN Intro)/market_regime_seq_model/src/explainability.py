import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from sklearn.metrics import f1_score

logger = logging.getLogger("market_regime.explain")

def occlusion_sensitivity(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    output_dir: str = "results/explainability"
):
    """
    Compute occlusion sensitivity (feature importance).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline performance
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    baseline_f1 = f1_score(y_test, y_pred, average='macro')
    
    importance = {}
    
    # X_test shape: (n_samples, seq_len, n_features)
    n_features = X_test.shape[2]
    
    logger.info(f"Calculating occlusion sensitivity for {model_name}...")
    
    for i in range(n_features):
        feature = feature_names[i]
        X_temp = X_test.copy()
        
        # Replace with mean of that feature across the dataset (or zero)
        # Calculating mean per feature across all timesteps and samples
        # Here we just replace with 0 for simplicity as "zero out" is common,
        # but mean is better if 0 is meaningful. 
        # Prompt says "Zero, or the feature's mean (decide and document)."
        # We will use Mean.
        
        # Compute global mean for this feature index
        feat_mean = np.mean(X_test[:, :, i])
        X_temp[:, :, i] = feat_mean
        
        y_probs_occ = model.predict(X_temp, verbose=0)
        y_pred_occ = np.argmax(y_probs_occ, axis=1)
        f1_occ = f1_score(y_test, y_pred_occ, average='macro')
        
        # Importance = Drop in F1
        drop = baseline_f1 - f1_occ
        importance[feature] = float(drop)
        
    # Save JSON
    with open(os.path.join(output_dir, f"occlusion_importance_{model_name}.json"), 'w') as f:
        json.dump(importance, f, indent=4)
        
    # Plot
    plt.figure(figsize=(10, 8))
    # Sort by importance
    sorted_imp = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    sns.barplot(x=list(sorted_imp.values()), y=list(sorted_imp.keys()))
    plt.title(f"Occlusion Importance (F1 Drop) - {model_name}")
    plt.xlabel("F1 Drop")
    
    plt.savefig(os.path.join(output_dir, f"occlusion_importance_{model_name}.png"), bbox_inches='tight')
    plt.close()

def compute_saliency(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    feature_names: list[str],
    model_name: str,
    output_dir: str = "results/explainability"
):
    """
    Compute saliency maps for a sample of sequences.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        probs = model(X_tensor)
        # We want gradients of the predicted class probability
        # But taking max prob is easier
        top_probs = tf.reduce_max(probs, axis=1)
        
    grads = tape.gradient(top_probs, X_tensor)
    grads = tf.abs(grads) # Magnitude
    
    # Convert to numpy
    grads_np = grads.numpy()
    
    # Plot for each sample
    for i in range(len(X_sample)):
        plt.figure(figsize=(12, 6))
        # Shape: (seq_len, n_features)
        sample_grads = grads_np[i]
        
        sns.heatmap(sample_grads.T, yticklabels=feature_names, cmap="viridis")
        plt.title(f"Saliency Map - {model_name} - Sample {i}")
        plt.xlabel("Time Step")
        
        plt.savefig(os.path.join(output_dir, f"saliency_{model_name}_sample_{i}.png"), bbox_inches='tight')
        plt.close()

def characterize_regimes(
    df_with_regime: pd.DataFrame,
    output_dir: str = "results/explainability"
):
    """
    Characterize regimes by feature averages.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select key features
    key_features = [
        'vol_5d', 'vol_20d', 'vol_acc', 'vol_z', 
        'bb_width', 'rsi_14', 'ret_1d', 'slope_10'
    ]
    # Ensure they exist
    available_features = [f for f in key_features if f in df_with_regime.columns]
    
    if 'regime' not in df_with_regime.columns:
        logger.warning("No regime column found for characterization.")
        return
        
    profile = df_with_regime.groupby('regime')[available_features].mean()
    
    # Save JSON
    profile.to_json(os.path.join(output_dir, "regime_profiles.json"), indent=4)
    
    # Plot (Heatmap or Bar)
    # Normalizing for better visualization (z-score per feature)
    profile_norm = (profile - profile.mean()) / profile.std()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(profile_norm.T, annot=True, cmap="RdBu_r", center=0)
    plt.title("Regime Feature Profiles (Normalized)")
    plt.savefig(os.path.join(output_dir, "regime_profiles.png"), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Regime characterization saved to {output_dir}")

