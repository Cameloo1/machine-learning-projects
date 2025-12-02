# Sequence Model for Market Regime Forecasting

## Project Overview
This project implements a production-grade machine learning pipeline to forecast stock market regimes (e.g., "Low Volatility", "Crisis", "Bull Run") using daily OHLCV data. It leverages a combination of advanced feature engineering, sequence modeling (LSTM, 1D-CNN), and baseline comparisons (Random Forest) to predict future market states.

The pipeline is designed to be modular, reproducible, and robust, handling everything from data ingestion to explainability analysis.

### Key Features
*   **Dual Data Source**: Primary ingestion from Yahoo Finance with automatic fallback to Stooq.
*   **Advanced Feature Engineering**: Calculates Volatility (acceleration, z-scores), RSI, MACD, Bollinger Bands, and Momentum Bursts.
*   **Sequence Modeling**: Utilizes sliding window sequences (30-day lookback) to train Time-Series LSTMs and 1D-CNNs.
*   **Strict Validation**: Implements time-series splitting (no future leakage) and standardizes features based on training data only.
*   **Explainability**: Includes Occlusion Sensitivity, Saliency Maps, and Regime Profiling to interpret *why* a model predicts a specific regime.

---

## Project Structure

```text
market_regime_seq_model/
│
├── data/
│   ├── raw/              # Raw OHLCV data (spy_merged.csv)
│   ├── features/         # Engineered features (features.csv)
│   ├── regimes/          # Target labels (regime_labels.csv)
│   └── processed/        # Numpy arrays for model training (X_train.npy, etc.)
│
├── models/               # Saved models and scalers
│   ├── lstm/
│   ├── cnn/
│   └── baseline_rf/
│
├── results/              # Evaluation outputs
│   ├── metrics/          # JSON metrics (F1, Precision, Recall)
│   ├── plots/            # Confusion Matrices, Predictions, Comparisons
│   └── explainability/   # Feature importance plots and Saliency maps
│
├── src/                  # Source code modules
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── models_lstm.py
│   ├── seq_builder.py
│   └── ...
│
├── main.py               # Orchestration entry point
└── requirements.txt      # Dependencies
```

---

## Getting Started

### Prerequisites
*   Python 3.10+
*   An active internet connection (for data download)

### Installation

1.  **Clone the repository** (or navigate to the project folder):
    ```bash
    cd market_regime_seq_model
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Regime Labels**:
    *   Place your regime classification file at `data/regimes/regime_labels.csv`.
    *   Format: Must contain `Date` (YYYY-MM-DD) and `regime` (integer class 0, 1, 2...).
    *   *Note: If this file is missing, the pipeline will generate dummy K-Means clusters for demonstration.*

### Running the Pipeline
Execute the main script to run the full end-to-end pipeline:

```bash
python main.py
```

This will:
1.  Download SPY data (2010–Present).
2.  Generate technical indicators.
3.  Merge features with your regime labels.
4.  Train LSTM, CNN, and Random Forest models.
5.  Save performance metrics and explainability plots to the `results/` directory.

---

## Analysis: Synthetic vs. Real Data Results

The transition from synthetic (K-Means generated) labels to real-world market regime labels highlighted critical challenges in deep learning for finance.

### 1. Performance Comparison
*   **Synthetic Data (Previous Run)**: The models achieved **~62% accuracy** and healthy F1 scores. This was because the synthetic labels were mathematically derived from the features themselves (e.g., K-Means on Volatility), making the task "easy" for the models to reverse-engineer.
*   **Real Data (Current Run)**: The models achieved **High Accuracy (85%)** but **Failed F1 Scores (~0.46)**. They essentially stopped learning features and collapsed to guessing the majority class.

### 2. Why did the models fail on real data?
*   **Data Starvation**: The real-world dataset provided was small (~480 days total, ~280 for training). Deep learning models (LSTMs/CNNs) have thousands of parameters and require thousands of examples to generalize effective patterns.
*   **Class Imbalance**: The test period (Aug–Nov 2025) was dominated by "Regime 1" (85% of samples). The models optimized their loss functions by simply predicting "1" for everything, achieving high accuracy by ignoring the minority "Regime 0".

### 3. Visual Evidence
*   **Confusion Matrices**: Showed vertical lines (predicting only one class) rather than a diagonal spread.
*   **Prediction Plots**: Flat lines indicating the models were insensitive to changing market data inputs.
*   **Occlusion Plots**: Blank, because the models ignored input features entirely, so removing them had zero impact on the output.

### Recommendation
To deploy this system effectively with real labels:
1.  **Expand History**: Incorporate 10+ years of labeled regime data to capture diverse market conditions (2008, 2020, etc.).
2.  **Model Selection**: For small datasets (<1000 samples), prefer simpler models like Random Forest or Logistic Regression over Deep Learning.
3.  **Balance Training**: Implement class weighting or oversampling (SMOTE) to force models to learn minority regimes.

