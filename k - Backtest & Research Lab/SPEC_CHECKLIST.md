# Backtest Lab Spec Checklist

## Implemented

- long-format price data contract
- config-driven runs
- no-lookahead timing contract with `decision_ts`
- centralized alignment logic
- SMA, RSI, equal-weight, volatility-target, and ML-gated strategies
- turnover, transaction cost, and slippage handling
- data validation and integrity diagnostics
- per-run resolved config and metadata
- metrics, returns, weights, trades, and plots
- walk-forward tests and sensitivity harness
- pytest coverage for alignment, config, costs, validation, strategies, and pipeline output

## Verification Commands

```bash
pytest -q -ra -W error::FutureWarning -W error::UserWarning -W error::DeprecationWarning
python scripts/run_required_experiments.py
```

## Artifact Policy

Keep source, configs, tests, and stable reports in Git. Generated run artifacts should stay local unless deliberately promoted. Root `.gitignore` excludes generated `report.html` files.

## Open Hardening Items

- add a tiny offline sample-data smoke command for CI
- add artifact hash comparison for promoted reports
- add explicit storage budget guidance for large runs
- separate stable public reports from local audit output
