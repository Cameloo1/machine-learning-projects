# Tiny RL Trading Sandbox

A learning-focused reinforcement learning sandbox for algorithmic trading, built to explore RL + ML engineering principles using SPY intraday data.

## Project Overview

This project implements an end-to-end RL trading pipeline using a custom Gymnasium environment and Deep Q-Network (DQN) agent. The goal is **not** to achieve production-grade trading performance, but rather to demonstrate clean ML engineering practices including:

- Proper train/validation/test time-series splits to avoid data leakage
- Custom Gymnasium environment with realistic trading mechanics (transaction costs, position constraints)
- Structured experiment logging and reproducibility
- Comprehensive evaluation with baseline comparisons

**Environment Design:**
- **Actions**: `0=hold current position`, `1=go/hold long`, `2=go/hold short`, `3=explicit close`
- **Reward**: Per-bar PnL minus trade/exposure/flip penalties with a small exit bonus tied to realized trade return
- **Observation**: Rolling feature window plus columns for current position, normalized trade age, trade-level return, and trade drawdown
- **Trading Costs**: Configurable basis points on position changes

**Data Splits:**
- **Training**: 2019-01-01 to 2022-12-31 (4 years)
- **Validation**: 2023-01-01 to 2024-06-30 (1.5 years)  
- **Test**: 2024-07-01 onwards (held out for final evaluation)

This time-based split ensures the model is evaluated on the most recent market regime.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  DataConfig       │ CSV path, date ranges for splits            │
│  SPYDataLoader    │ Load CSV, parse timestamps, create splits   │
│  features.py      │ Technical indicators + z-score normalization│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Environment Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  TradingEnv       │ Gymnasium env with windowed features        │
│                   │ Position as part of state, discrete actions │
│                   │ Reward = position × return - trade_cost     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Agent & Training Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  DQN Agent        │ stable-baselines3 with config-driven params │
│  train.py         │ Training loop with periodic validation      │
│  TrainingConfig   │ Hyperparameters (lr, gamma, buffer, etc.)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  metrics.py       │ Equity, drawdown, Sharpe, trade statistics  │
│  test_eval.py     │ Test set evaluation with B&H comparison     │
│  plotting.py      │ Equity curves, action distributions, PnL    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Experiment Outputs                         │
├─────────────────────────────────────────────────────────────────┤
│  experiments/{exp_name}/                                        │
│    ├── config.json           # Full configuration snapshot      │
│    ├── training_log.csv      # Per-epoch validation metrics     │
│    ├── best_model.zip        # Best model by validation metric  │
│    ├── test_metrics.json     # Final test performance           │
│    ├── trades_test.csv       # Per-step trading log             │
│    ├── equity_curve_test.png # RL vs Buy-and-Hold               │
│    └── action_distribution_test.png                             │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rl_trading_sandbox/
├── data/
│   └── spy_30m_2019_2025.csv    # SPY intraday data (download or provide)
├── scripts/
│   └── download_data.py         # Data download utility
├── experiments/                  # Training outputs (auto-generated)
│   └── exp_001_baseline_dqn/
│       ├── config.json
│       ├── training_log.csv
│       ├── best_model.zip
│       ├── test_metrics.json
│       ├── trades_test.csv
│       └── *.png (plots)
├── rl_trading/
│   ├── __init__.py
│   ├── config.py                # Configuration dataclasses
│   ├── data_loader.py           # SPYDataLoader class
│   ├── features.py              # Feature engineering pipeline
│   ├── envs/
│   │   ├── __init__.py
│   │   └── trading_env.py       # TradingEnv Gymnasium environment
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # DQN training script
│   │   ├── eval.py              # Model evaluation utilities
│   │   └── test_eval.py         # Test set evaluation script
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py           # Performance metrics
│       ├── logging_utils.py     # Experiment logging
│       ├── plotting.py          # Visualization utilities
│       └── random_runner.py     # Random baseline
├── notebooks/
│   └── 01_eda_and_features.ipynb
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download or Prepare Data

**Option A: Use the download script (recommended)**

```bash
# Download from yfinance/stooq with synthetic intraday generation
python scripts/download_data.py

# Or specify options
python scripts/download_data.py --interval 1h --start 2020-01-01
python scripts/download_data.py --synthetic  # Force synthetic generation
```

> **Note**: yfinance has limited intraday history (60 days for 30m bars). The script 
> automatically falls back to Stooq for daily data and generates synthetic intraday bars.
> This is sufficient for learning/testing the pipeline.

**Option B: Provide your own data**

Place your SPY intraday CSV at `data/spy_30m_2019_2025.csv` with columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Datetime for each bar |
| `open` | Opening price |
| `high` | High price |
| `low` | Low price |
| `close` | Closing price |
| `volume` | Trading volume |

### 3. Run Training

```bash
# With real SPY data
python -m rl_trading.training.train

# With synthetic data (for testing the pipeline)
python -m rl_trading.training.train --synthetic

# Hyperparameter Search
python -m rl_trading.training.hp_search

```
The hyperparameter helper samples a handful of `TrainingConfig` variants (default 10 trials)
and runs a full training loop for each, saving outputs under `experiments/exp_hp_XX/`.
To sweep fewer or more trials, edit the `run_random_search(n_trials=...)` call at the
bottom of `rl_trading/training/hp_search.py` or import the function from a notebook.

### 4. Run Test Evaluation

```bash
# Evaluate on held-out test set
python -m rl_trading.training.test_eval

# With synthetic data
python -m rl_trading.training.test_eval --synthetic
```

### 5. View Results

Artifacts are saved to `experiments/{exp_name}/`:

| File | Description |
|------|-------------|
| `config.json` | Full configuration snapshot |
| `training_log.csv` | Per-epoch validation metrics |
| `best_model.zip` | Best model checkpoint |
| `test_metrics.json` | Final test performance |
| `trades_test.csv` | Per-step trading log |
| `equity_curve_test.png` | RL vs Buy-and-Hold |
| `action_distribution_test.png` | Action frequency |
| `pnl_histogram_test.png` | PnL distribution |

## Quick Start (Code)

```python
from rl_trading import (
    get_default_configs,
    SPYDataLoader,
    add_basic_features,
    get_feature_columns,
    FeatureScaler,
    TradingEnv,
    train_dqn,
    run_test_evaluation,
)

# Train a model
exp_path = train_dqn(exp_name="my_experiment")

# Evaluate on test set
metrics = run_test_evaluation(exp_name="my_experiment")
print(f"Test Return: {metrics['total_return']:.2%}")
print(f"B&H Return:  {metrics['bh_total_return']:.2%}")
```

## Configuration

### Data Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_path` | `data/spy_30m_2019_2025.csv` | Path to data |
| `train_start` | `2019-01-01` | Training start |
| `train_end` | `2022-12-31` | Training end |
| `val_start` | `2023-01-01` | Validation start |
| `val_end` | `2024-06-30` | Validation end |
| `test_start` | `2024-07-01` | Test start |

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 150,000 | Total training steps (split into 10 epochs) |
| `num_epochs` | 10 | Validation checkpoints per run |
| `learning_rate` | 1e-4 | Base Adam learning rate (decays 50% every 5 epochs) |
| `buffer_size` | 300,000 | Replay buffer size |
| `batch_size` | 32 | Minibatch size |
| `gamma` | 0.995 | Discount factor |
| `train_freq` | 4 | Gradient updates every N env steps |
| `target_update_interval` | 10,000 | Target network sync frequency |
| `exploration_fraction` | 0.2 | Portion of training used for epsilon decay |
| `exploration_final_eps` | 0.02 | Final epsilon |
| `policy_hidden_sizes` | (256, 256) | MLP architecture |
| `patience` | 5 | Epochs without improvement before early stopping |

### Feature Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 30 | Observation window (bars) |
| `use_log_returns` | True | Use log vs simple returns |

## Features

The feature engineering pipeline computes:

| Feature | Description |
|---------|-------------|
| `ret_log_1` | Log return |
| `vol_20` | 20-bar rolling volatility |
| `ma_fast` | 10-bar moving average |
| `ma_slow` | 50-bar moving average |
| `rsi_14` | 14-bar RSI |
| `vol_rel` | Relative volume |

All features are z-score normalized using statistics from training data only.

## Analysis & Methodology

See [`ANALYSIS.md`](ANALYSIS.md) for the latest experiment methodology,
validation/test findings, and recommendations for future work.

## Limitations & Notes

**This is a learning sandbox, not production trading software.**

Key limitations:
- **Simplified execution model**: No slippage, partial fills, or market impact
- **Single asset**: Only SPY, no portfolio management
- **Basic features**: Hand-crafted indicators, no learned representations
- **DQN only**: No advanced algorithms (PPO, SAC, etc.)
- **No live trading**: Backtesting only

**The goal is pipeline quality, structure, and explainability—not trading performance.**

## Development Status

| Slice | Component | Status |
|-------|-----------|--------|
| 1 | Data & Configuration | ✅ Complete |
| 2 | Environment & Metrics | ✅ Complete |
| 3 | DQN Training Pipeline | ✅ Complete |
| 4 | Test Evaluation & Plotting | ✅ Complete |

## License

MIT

---

*Built as a learning project to explore reinforcement learning in quantitative finance.*
