# Mini-RL Trading Sandbox

Learning-focused reinforcement-learning trading sandbox built around a custom Gymnasium environment and a DQN agent.

## What It Includes

- SPY data loader and synthetic-data fallback
- feature engineering and scaling
- custom discrete-action trading environment
- DQN training with validation checkpoints
- evaluation against buy-and-hold
- experiment logs, metrics, model checkpoints, and plots

## Data

Default input path:

```text
data/spy_30m_2019_2025.csv
```

Generate synthetic data for a local smoke path:

```bash
python scripts/download_data.py --synthetic
```

## Run

```bash
pip install -r requirements.txt
python -m rl_trading.training.train
python -m rl_trading.training.test_eval
```

## Outputs

Experiment outputs are written under `experiments/<experiment-name>/`:

- `config.json`
- `training_log.csv`
- `best_model.zip`
- `final_model.zip`
- `test_metrics.json`
- `trades_test.csv`
- evaluation plots

## Reproducibility Notes

RL results are sensitive to seeds, data, and library versions. Treat this as a sandbox for pipeline mechanics and evaluation discipline, not as a profitable trading strategy.
