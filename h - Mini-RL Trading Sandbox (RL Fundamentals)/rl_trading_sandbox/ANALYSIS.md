# Analysis & Methodology

This document captures the latest experimental methodology plus a consolidated
view of the real-data baseline, the synthetic-data dry run, and the first
hyperparameter-search sweep conducted with `hp_search.py`.

## Methodology

- **Data pipeline** – `SPYDataLoader` enforces chronological splits  
  (`2019‑01‑01`→`2022‑12‑31` train, `2023‑01‑01`→`2024‑06‑30` validation,
  `2024‑07‑01` onward test). Features are engineered via `features.py` and
  z-scored with `FeatureScaler`.
- **Environment** – `TradingEnv` tracks trade-level context (`trade_age`,
  `entry_price`, trade return, drawdown) and exposes a 4-action space
  (`hold`, `long`, `short`, `close`). Reward shaping includes trade/exposure/flip
  penalties plus a small exit bonus tied to realized trade returns.
- **Training loop** – `train_dqn_with_configs` (used by both the CLI and the
  random-search helper) trains a Stable-Baselines3 DQN for 150k steps over 10
  epochs, halves the learning rate every 5 epochs, validates every epoch on 10
  episodes, logs metrics, and applies early stopping (patience=5).
- **Evaluation** – `test_eval.py` loads the best checkpoint, runs a single
  deterministic episode across the held-out 2025 test window, computes metrics
  vs. buy-and-hold, and generates the equity/action/PnL plots.
- **Hyperparameter search** – `rl_trading/training/hp_search.py` samples light
  perturbations of `TrainingConfig`, calls `train_dqn_with_configs`, and saves
  each trial under `experiments/exp_hp_XX/` with full configs and logs.

## Key Findings

1. **Validation optimism vs. test drawdown** – Because validation episodes reuse
   the same deterministic start, the agent can look strong on that slice while
   failing on the 2025 test regime (RL return −30.99% vs. B&H +11.67%).
2. **Trade-level diagnostics now possible** – The enriched observation/info
   payloads surface trade age, return, and drawdown, making it straightforward
   to explain failures (e.g., 79 trades with 0.49 profit factor on test).
3. **Search tooling works but needs richer hypotheses** – All 10 random-search
   trials finished with validation equity ≤1.0, so simple hyperparameter tweaks
   are insufficient; future work needs regime-aware features or new algorithms.

## Experiment Snapshots

| Experiment | Data | Notes | Val Final Equity | Test Final Equity | Test Return |
|------------|------|-------|------------------|-------------------|-------------|
| `exp_001_baseline_dqn` | Real SPY splits | Validation solid, test collapses in mid‑2025 regime shift | **1.092** | **0.690** | **−30.99%** |
| `exp_synthetic_test` | Synthetic intraday bars | Dry run to validate the pipeline with tiny training budget | **0.964** | **0.984** | **−1.60%** |
| `exp_hp_01` (best of `exp_hp_00–09`) | Real SPY splits + sampled hyperparams | Best random-search trial; still < 1.0 on validation, so not evaluated on test | **0.970** | _n/a_ | _n/a_ |

### `exp_001_baseline_dqn` (real data)
- **Validation:** +9.19% return, Sharpe 0.38, profit factor 0.99 across 10
  episodes.
- **Test vs. B&H:** RL final equity **0.69** vs. buy-and-hold **1.12** (+11.67%);
  Sharpe plunged to **−5.76** with −30.99% total return and −41.6% max drawdown.
- **Trade stats:** 79 trades, 41.8% win rate, profit factor 0.49, average
  duration 2.8 bars, showing the agent kept pressing directional exposure during
  the downturn.

### `exp_synthetic_test`
- **Validation:** Already negative after epoch 1 (−3.63% return, final equity
  0.964) and degraded further in epoch 2, highlighting overfitting.
- **Test vs. B&H:** RL −1.60% vs. B&H +0.60%; similar drawdowns (~−2.1%) but poor
  recovery.
- **Behaviour:** 95 trades with 27% win rate, profit factor 0.58, average
  duration 2.5 bars, and a heavy long bias (56% long, 10% short). Useful for
  validating turnover/transaction-cost effects even though performance was poor.

### Random-search trials (`exp_hp_00` → `exp_hp_09`)
- **Sampling:** `hp_search.py` varied learning rate, gamma, buffer size,
  exploration fraction, and batch size for 10 trials.
- **Best trial (`exp_hp_01`):** validation equity 0.970 (−2.98% return) with
  Sharpe 0.07—below the baseline’s 1.09—so we skipped the expensive test eval.
- **Overall:** No sampled configuration crossed validation >1.0, reinforcing
  that we need structural changes (regime features, alternate algorithms, risk
  filters) rather than small hyperparameter nudges.

## Recommendations

1. **Vary validation start indices** – Run validation episodes with randomized
   starts (or multiple deterministic offsets) so metrics aggregate across
   distinct sub-periods.
2. **Add regime-aware features/filters** – Feed slowdown filters or regime
   indicators into the observation to help the policy exit prolonged drawdowns.
3. **Experiment with new learners** – Try PPO/SAC or an LSTM policy and evaluate
   them via `hp_search.py` once validation covers diverse windows.
4. **Extend evaluation coverage** – Run `test_eval` mid-training after promising
   validation epochs to detect overfitting earlier, and compare against rolling
   buy-and-hold windows.

## Reproduction Checklist

```bash
# Install deps
pip install -r requirements.txt

# Train baseline (SPY data must exist under data/)
python -m rl_trading.training.train --exp-name exp_001_baseline_dqn

# Run held-out test evaluation
python -m rl_trading.training.test_eval --exp-name exp_001_baseline_dqn

# Optional: launch the random-search helper (default 10 trials)
python -m rl_trading.training.hp_search
```

Artifacts for each experiment live under `experiments/<name>/` with configs,
logs, metrics, and plots so findings can be audited or extended.

