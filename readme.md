# Machine Learning Projects

[![Python >= 3.11](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)

A compact portfolio of independent ML projects across market data, security analytics, NLP, reinforcement learning, API serving, AutoML, backtesting, and learned-index sorting.

Each project is self-contained. Dependencies, data assumptions, run commands, and generated artifacts live inside the project folder. The root-level verifier gives a consistent way to check that each folder is still present, readable, and reproducible.

## Project Index

| ID | Project | Focus | Default proof path |
| --- | --- | --- | --- |
| `a` | SPY Market Regime Clustering | KMeans clustering on SPY returns and volatility | network-backed script |
| `b` | Short-Term Price Move Classifier | next-day direction classification | network-backed package run |
| `c` | Volatility Forecasting | next-day realized volatility regression | network-backed script |
| `d` | Synthetic SOC Anomaly Detector | IsolationForest and OneClassSVM on synthetic alerts | offline synthetic-data run |
| `e` | News Headline Sentiment | TF-IDF sentiment classification | data/API-dependent pipeline |
| `f` | Market Regime Sequence Model | LSTM, 1D-CNN, and RF regime models | network-backed training run |
| `g` | Alert Triage Scorer | XGBoost/LightGBM plus SHAP explanations | offline synthetic-data run |
| `h` | Mini-RL Trading Sandbox | Gymnasium trading environment and DQN agent | synthetic data plus training run |
| `i` | Risk Score API | FastAPI model-serving demo | pytest/TestClient |
| `j` | Mini AutoML Runner | tabular experiment runner | pytest and demo configs |
| `k` | Backtest & Research Lab | quant backtesting and validation harness | pytest and config runs |
| `l` | Learned Bucket Sort Predictor | learned CDF bucket assignment for bucket sort | pytest and offline synthetic benchmarks |

## Repository Layout

- `projects.json` - reproducibility contract for every project.
- `scripts/verify_projects.py` - root verifier.
- `docs/reproducibility.md` - verification workflow and flags.
- `a - .../` through `l - .../` - independent project folders.

Generated verification reports are written to `reports/reproducibility/` and ignored by Git. Disposable verifier workspaces and virtual environments are written to `.verify/` and ignored by Git.

## Quick Verification

Run this from the repository root:

```bash
python scripts/verify_projects.py --level quick
```

This safe default checks required files, dependency specs, Python syntax, notebook JSON, current artifacts, and JSON artifact parseability. It does not install dependencies or run project commands.

To run declared commands in disposable copies:

```bash
python scripts/verify_projects.py --level quick --run-commands
```

To install per-project dependencies and run heavier checks:

```bash
python scripts/verify_projects.py --level full --install --allow-network --run-commands
```

## Artifact Policy

Keep durable source, configs, tests, notebooks, and concise docs in Git. Generated reports, model outputs, plots, and raw data should only be tracked when they are intentionally part of the portfolio evidence.

Recent cleanup removed generated Backtest Lab `report.html` files from version control and ignores future `k - Backtest & Research Lab/artifacts/**/report.html` outputs.

## Notes

- Some finance projects need live market data through `yfinance` or fallback data sources.
- Some projects ship pre-generated artifacts for review, but a passing artifact inventory is not the same as a full rerun.
- Use `projects.json` as the source of truth for current verification commands.
