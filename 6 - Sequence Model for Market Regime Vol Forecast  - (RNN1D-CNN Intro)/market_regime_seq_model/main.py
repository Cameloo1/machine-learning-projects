import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
import tensorflow as tf

from src.utils import setup_directories, setup_logging, set_seeds
from src.data_ingestion import download_spy_data
from src.feature_engineering import engineer_features
from src.regime_loader import load_and_merge_regimes
from src.seq_builder import fit_scaler_on_train, build_sequences, time_series_split, save_processed_data
from src.models_lstm import train_lstm
from src.models_cnn import train_cnn
from src.baseline_rf import build_rf_features, train_random_forest
from src.evaluation import evaluate_model, compare_models
from src.explainability import occlusion_sensitivity, compute_saliency, characterize_regimes

def generate_dummy_regimes(features_path: str, output_path: str):
    """
    Generate dummy KMeans regimes if the file is missing,
    so the pipeline is runnable for the user.
    """
    print("Generating dummy regimes (KMeans=3) for demonstration...")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    # Simple KMeans on a few features
    cols = ['vol_5d', 'vol_20d', 'rsi_14']
    X = df[cols].fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    regime_df = pd.DataFrame({
        'Date': df.index,
        'regime': labels
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    regime_df.to_csv(output_path, index=False)
    print(f"Saved dummy regimes to {output_path}")

def main():
    # 0. Setup
    setup_directories()
    logger = setup_logging()
    set_seeds(42)
    logger.info("Pipeline started.")

    # 1. Data Ingestion
    logger.info("Step 1: Data Ingestion")
    spy_path = "data/raw/spy_merged.csv"
    if not os.path.exists(spy_path):
        # Download up to current date to match new regime file (2023-2025)
        df_spy = download_spy_data(start="2010-01-01", end=None)
        df_spy.to_csv(spy_path)
    else:
        logger.info("SPY data already exists.")

    # 2. Feature Engineering
    logger.info("Step 2: Feature Engineering")
    features_path = "data/features/features.csv"
    if not os.path.exists(features_path) or not os.path.exists(spy_path):
        engineer_features(input_path=spy_path, output_path=features_path)
    else:
        logger.info("Features already exist.")

    # 3. Regime Loading
    logger.info("Step 3: Regime Loading")
    regime_path = "data/regimes/regime_labels.csv"
    
    # Check if regime file exists, if not create dummy for runnability
    if not os.path.exists(regime_path):
        logger.warning("Regime file not found. Generating dummy regimes from features...")
        generate_dummy_regimes(features_path, regime_path)
        
    df_merged = load_and_merge_regimes(pd.read_csv(features_path, index_col=0, parse_dates=True), regime_path)
    
    # 4. Sequence Building & Splitting
    logger.info("Step 4: Sequence Building")
    
    # Feature columns (exclude target and non-numeric)
    target_col = "regime"
    # Drop target, date index is not a column
    feature_cols = [c for c in df_merged.columns if c != target_col]
    
    # Split Dates - Adjusted for the provided regime file (Dec 2023 - Nov 2025)
    # Train: Dec 2023 -> Mar 2025
    # Val: Apr 2025 -> Jul 2025
    # Test: Aug 2025 -> Nov 2025
    train_end = "2025-03-31"
    val_end = "2025-07-31"
    
    # Fit Scaler on Train ONLY
    # We need to identify train rows first
    train_mask = df_merged.index <= pd.Timestamp(train_end)
    train_df = df_merged[train_mask]
    scaler = fit_scaler_on_train(train_df, feature_cols)
    
    # Transform entire DF
    df_scaled = df_merged.copy()
    df_scaled[feature_cols] = scaler.transform(df_merged[feature_cols])
    
    # Build Sequences
    SEQ_LEN = 30
    X, y, timestamps = build_sequences(df_scaled, feature_cols, target_col, seq_len=SEQ_LEN)
    
    # Split
    splits = time_series_split(timestamps, X, y, train_end, val_end)
    save_processed_data(splits)
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Determine Num Classes Global
    num_classes = int(df_merged[target_col].max() + 1)
    logger.info(f"Detected {num_classes} regime classes.")

    # 5. Models
    logger.info("Step 5: Model Training")
    
    # 5.1 LSTM
    lstm_model = train_lstm(X_train, y_train, X_val, y_val, epochs=10, num_classes=num_classes) # 10 for speed in demo
    
    # 5.2 CNN
    cnn_model = train_cnn(X_train, y_train, X_val, y_val, epochs=10, num_classes=num_classes)
    
    # 5.3 RF Baseline
    # Prepare features
    logger.info("Building RF features...")
    X_train_rf = build_rf_features(X_train, feature_cols)
    X_val_rf = build_rf_features(X_val, feature_cols)
    X_test_rf = build_rf_features(X_test, feature_cols)
    
    rf_model = train_random_forest(X_train_rf, y_train, X_val_rf, y_val)
    
    # 6. Evaluation
    logger.info("Step 6: Evaluation")
    metrics_all = {}
    
    metrics_all['lstm'] = evaluate_model(lstm_model, X_test, y_test, "lstm", timestamps=splits['test_ts'])
    metrics_all['cnn'] = evaluate_model(cnn_model, X_test, y_test, "cnn", timestamps=splits['test_ts'])
    metrics_all['rf'] = evaluate_model(rf_model, X_test_rf, y_test, "rf", timestamps=splits['test_ts'])
    
    compare_models(metrics_all)
    
    # 7. Explainability
    logger.info("Step 7: Explainability")
    
    # Occlusion
    occlusion_sensitivity(lstm_model, X_test, y_test, feature_cols, "lstm")
    occlusion_sensitivity(cnn_model, X_test, y_test, feature_cols, "cnn")
    
    # Saliency (Sample of 5)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    X_sample = X_test[sample_indices]
    compute_saliency(lstm_model, X_sample, feature_cols, "lstm")
    compute_saliency(cnn_model, X_sample, feature_cols, "cnn")
    
    # Regime Profiles
    characterize_regimes(df_merged)
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

