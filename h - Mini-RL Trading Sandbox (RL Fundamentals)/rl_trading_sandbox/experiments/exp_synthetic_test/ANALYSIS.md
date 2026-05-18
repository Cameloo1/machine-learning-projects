# Synthetic Experiment Analysis

## Result

The saved synthetic experiment proves the RL pipeline can train, save models, and produce evaluation artifacts.

Current `test_metrics.json` shows:

- final equity: `0.9713`
- total return: `-0.0287`
- max drawdown: `-0.0303`
- trades: `77`
- buy-and-hold total return: `0.0060`

## Interpretation

This run is an engineering smoke artifact, not a successful trading result. The agent underperformed buy-and-hold on this saved test.

## Useful Files

- `config.json`
- `training_log.csv`
- `best_model.zip`
- `final_model.zip`
- `test_metrics.json`
- `trades_test.csv`
- evaluation plots
