# Backtest Lab Research Report

## Purpose

This report summarizes the current Backtest Lab methodology and evidence surface. It is documentation for the research harness, not an investment recommendation.

## Methodology

- Long-format OHLCV input data.
- Centralized signal/return alignment to avoid lookahead.
- Explicit `decision_ts` and realized return `ts`.
- Config-driven strategy runs.
- Transaction-cost and slippage modeling.
- Validation artifacts written per run.

## Strategies

- SMA trend
- RSI mean reversion
- equal-weight baseline
- volatility-target overlay
- ML-gated signals
- walk-forward variants

## Evidence Artifacts

Each run writes to `artifacts/<run_id>/`:

- resolved config and config integrity JSON
- run metadata
- diagnostics and data-integrity reports
- returns, weights, trades, and metrics
- plots
- generated `report.html`

Generated `report.html` files are not intended for source control.

## Current Confidence

The strongest evidence in this project is the test suite and the explicit timing/alignment contract. Strategy performance should be evaluated from fresh runs and should not be treated as stable unless the exact data, config, and artifact hash are preserved.

## Known Limits

- Market-data downloads are external and can change.
- Generated artifacts are snapshots, not proof of future performance.
- ML-gated runs depend on prediction CSV quality.
- Portfolio risk assumptions are simplified.

## Recommended Verification

```bash
pytest -q -ra -W error::FutureWarning -W error::UserWarning -W error::DeprecationWarning
python scripts/run_required_experiments.py
```
