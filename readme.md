# ML Projects Portfolio
Compact collection of self-contained ML projects (finance, cybersec, and others) with runnable scripts, notebooks, and saved artifacts.

![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)

## What this repo contains
- 10 + 1 W.I.P. independent subprojects, each in its own folder with its own dependencies and README.
- Runnable scripts and module entrypoints for training, evaluation, and inference.
- Notebooks for exploration and training workflows.
- Pre-generated artifacts and figures such as [artifacts/plots](<b - Short-Term Price Move Classifier - logistic regression, random forest, gradient boosting/artifacts/plots/>), [plots](<d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/plots/>), [results](<f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/results/>), [reports](<g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/reports/>), and [outputs](<j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/outputs/>).
- Synthetic datasets and sample CSVs for multiple projects (see Data).

### ML in 60 seconds
- A "model" here is a script or module that trains on CSVs or downloaded OHLCV data and writes metrics, plots, or predictions.
- Inputs are typically CSV files (tabular/synthetic) or market data pulled at runtime.
- Outputs are saved artifacts (plots, metrics JSON/CSV, trained model files) or an API response in the FastAPI service.

## Key ML components
- SPY Market Regime Clustering: unsupervised regime labeling with KMeans in [model.py](<a - SPY Market Regime Clustering - KMeans/model.py>) (inputs: SPY OHLCV; outputs: [spy_price_regimesSPY_last_2y.png](<a - SPY Market Regime Clustering - KMeans/spy_price_regimesSPY_last_2y.png>) and [spy_vol_scatterSPY_last_2y.png](<a - SPY Market Regime Clustering - KMeans/spy_vol_scatterSPY_last_2y.png>)).
- Short-Term Price Move Classifier: Logistic Regression, Random Forest, Gradient Boosting pipeline in [run_experiment.py](<b - Short-Term Price Move Classifier - logistic regression, random forest, gradient boosting/short_term_price_classifier/run_experiment.py>) (inputs: SPY OHLCV; outputs: plots in [artifacts/plots](<b - Short-Term Price Move Classifier - logistic regression, random forest, gradient boosting/artifacts/plots/>)).
- Next-Day Realized Volatility Forecasting: Linear Regression + Random Forest in [vol_forecasting.py](<c - Volatility Forecasting - Regression + Evaluation Discipline/vol_forecasting.py>) (inputs: SPY OHLCV; outputs: [SPY_volatility_forecast_comparison.png](<c - Volatility Forecasting - Regression + Evaluation Discipline/SPY_volatility_forecast_comparison.png>)).
- Synthetic SOC Alert Anomaly Detector: IsolationForest + OneClassSVM in [scripts/](<d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/scripts/>) (inputs: generated SOC events; outputs: [plots](<d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/plots/>) + CSV).
- News Headline Sentiment: TF-IDF + Logistic Regression / LinearSVC in [src/train.py](<e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/src/train.py>) and [scripts/run_end_to_end.py](<e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/scripts/run_end_to_end.py>) (inputs: [data/raw_headlines.csv](<e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/data/raw_headlines.csv>) and [data/labeled_headlines.csv](<e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/data/labeled_headlines.csv>); outputs: [reports/figures](<e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/reports/figures/>)).
- Market Regime Sequence Model: LSTM, 1D-CNN, and RF baseline in [main.py](<f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/main.py>) (inputs: OHLCV + [data/regimes/regime_labels.csv](<f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/data/regimes/regime_labels.csv>); outputs: [results/metrics](<f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/results/metrics/>), [results/plots](<f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/results/plots/>), [results/explainability](<f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/results/explainability/>)).
- Alert Triage Scorer: XGBoost/LightGBM + SHAP in [src/](<g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/src/>) (inputs: [data/raw/alerts_synthetic.csv](<g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/data/raw/alerts_synthetic.csv>); outputs: [artifacts](<g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/artifacts/>) and [reports](<g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/reports/>)).
- Mini RL Trading Sandbox: DQN agent in [training/train.py](<h - Mini-RL Trading Sandbox (RL Fundamentals)/rl_trading_sandbox/rl_trading/training/train.py>) (inputs: intraday SPY CSV; outputs: [experiments](<h - Mini-RL Trading Sandbox (RL Fundamentals)/rl_trading_sandbox/experiments/>)).
- Alert Risk Score API: HistGradientBoostingClassifier served by FastAPI in [app/main.py](<i - Risk Score API (Model as service)/risk-score-api/app/main.py>) (inputs: JSON alert features; outputs: label + probabilities).
- Mini Auto-ML Experiment Runner: CLI-driven tabular experiments in [autotab/cli.py](<j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/autotab/cli.py>) (inputs: CSV + YAML config; outputs: [outputs](<j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/outputs/>) with reports, metrics, models).
- Backtest & Research Lab

## Repo layout
- `a - SPY Market Regime Clustering - KMeans/` KMeans clustering on SPY with saved plots.
- `b - Short-Term Price Move Classifier - logistic regression, random forest, gradient boosting/` Supervised short-horizon classifier + notebook.
- `c - Volatility Forecasting - Regression + Evaluation Discipline/` Regression baseline for realized volatility.
- `d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/` Synthetic SOC generator + anomaly detection scripts.
- `e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/` NLP pipeline with reports and figures.
- `f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/` Sequence models and explainability artifacts.
- `g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/` Gradient-boosting ranking and SHAP explainability.
- `h - Mini-RL Trading Sandbox (RL Fundamentals)/rl_trading_sandbox/` RL environment, training, and experiments.
- `i - Risk Score API (Model as service)/risk-score-api/` FastAPI service with model artifacts and tests.
- `j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/` CLI-based AutoML runner with configs and outputs.

## Quickstart
Example end-to-end pipeline (train + evaluate + explain + infer) using the alert triage scorer:

```bash
cd "g - Alert Triage Scorer (Ranking + Explainability)\alert-triage-ml"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generation --n_samples 6000
python -m src.train
python -m src.evaluate
python -m src.explain
python -m src.inference --mode csv --input data/raw/alerts_synthetic.csv --output artifacts/scored_alerts.csv --model_path models/xgb_pipeline.pkl
```

Other entrypoints (one-liners):
- `a - SPY Market Regime Clustering - KMeans/model.py`: `python model.py`
- `b - Short-Term Price Move Classifier - logistic regression, random forest, gradient boosting/short_term_price_classifier/run_experiment.py`: `python -m short_term_price_classifier.run_experiment`
- `c - Volatility Forecasting - Regression + Evaluation Discipline/vol_forecasting.py`: `python vol_forecasting.py`
- `d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/scripts/generate_data.py`: `python scripts/generate_data.py`
- `d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/scripts/run_anomaly_detection.py`: `python scripts/run_anomaly_detection.py --data-path data/soc_synthetic.csv`
- `e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/scripts/run_end_to_end.py`: `python scripts/run_end_to_end.py`
- `f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/main.py`: `python main.py`
- `h - Mini-RL Trading Sandbox (RL Fundamentals)/rl_trading_sandbox/rl_trading/training/train.py`: `python -m rl_trading.training.train`
- `i - Risk Score API (Model as service)/risk-score-api/app/main.py`: `uvicorn app.main:app --reload`
- `j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner`: `pip install -e . && autotab --config configs/demo_classification.yaml`

## Configuration
- YAML configs for Auto-ML experiments live in `j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/configs/` (example: `configs/demo_classification.yaml` with `dataset.path`, `task.type`, `models.*`, `evaluation.split`, and `output.*`).
- RL training and data configs are defined as dataclasses in `h - Mini-RL Trading Sandbox (RL Fundamentals)/rl_trading_sandbox/rl_trading/config.py` (`DataConfig`, `TrainingConfig`, `FeatureConfig`).

## Data
- Finance projects (`a`, `b`, `c`, `f`) pull OHLCV data from yfinance with fallbacks; `f` also expects regime labels at `f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/data/regimes/regime_labels.csv`.
- SOC anomaly detection (`d`) uses generated data saved to `d - Synthetic SOC Alert Anomaly Detector - Unsupervised (isoforest + oneclassSVM)/data/soc_synthetic.csv`.
- News sentiment (`e`) includes `e - Text & Tabular News Headline Sentiment for Tickers (NLP Intro)/data/raw_headlines.csv` and `data/labeled_headlines.csv`.
- Alert triage (`g`) includes `g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/data/raw/alerts_synthetic.csv` plus processed splits.
- RL trading (`h`) expects an intraday CSV under `h - Mini-RL Trading Sandbox (RL Fundamentals)/rl_trading_sandbox/data/` and provides `scripts/download_data.py`.
- Risk score API (`i`) ships `i - Risk Score API (Model as service)/risk-score-api/data/alerts_sample.csv`.
- Auto-ML (`j`) includes example CSVs at `j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/examples/data/`.

## Reproducibility and engineering notes
- Seeds are set or configurable in multiple projects, e.g. `f - Sequence Model for Market Regime Vol Forecast  - (RNN1D-CNN Intro)/market_regime_seq_model/src/utils.py` and `g - Alert Triage Scorer (Ranking + Explainability)/alert-triage-ml/src/config.py`.
- Tests exist in `i - Risk Score API (Model as service)/risk-score-api/tests/` and `j - Mini Auto-ML Experiment Runner for Tabular Data/auto_experiment_runner/tests/`.

## Roadmap / next improvements
- Add a root-level manifest and a shared environment spec to standardize setup across projects.
- Introduce CI to run unit tests for `risk-score-api` and `auto_experiment_runner`.
- Add minimal data schema docs for each project to make inputs explicit without reading code.
- Consolidate common utilities (data download, plotting) to reduce duplication.
- Provide small, deterministic sample runs for each project to validate end-to-end execution.

## Contributing
- Each subproject is self-contained; install its local dependencies first.
- Tests: `cd "i - Risk Score API (Model as service)\risk-score-api" && pytest tests/` or `cd "j - Mini Auto-ML Experiment Runner for Tabular Data\auto_experiment_runner" && pytest`.

## License
No LICENSE file is present at the repository root.
