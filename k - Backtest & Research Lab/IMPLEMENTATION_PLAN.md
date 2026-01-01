## Implementation Plan

### What exists
- `src/backtest_lab/config.py` with schema validation, path resolution, config dump + run metadata.
- Data loaders in `src/backtest_lab/data/loader.py` (CSV + yfinance cache).
- Data validation in `src/backtest_lab/data/validate.py` and universe selection in `src/backtest_lab/data/universe.py`.
- Basic backtest pipeline in `src/backtest_lab/run.py` (single pass, SMA only).
- SMA trend strategy in `src/backtest_lab/strategies/sma_trend.py`.
- Minimal metrics + report scaffolding (`metrics/performance.py`, `metrics/drawdown.py`, `report/build.py`).
- Tests for loader, validation, SMA warmup policy, config resolution, and pipeline smoke.

### What is missing
- Alignment contract module and explicit returns computation.
- Execution layer: constraints, costs/slippage, accounting consistent with t+1 return timing.
- RSI technicals + multi-strategy support (RSI mean reversion, vol target overlay, ML-gated).
- Walk-forward windows + engine with universe locking and leakage checks.
- Expanded metrics, rolling stats, and richer HTML report (config snapshot, diagnostics, plots).
- Multi-asset config + walk-forward config, additional tests (alignment, costs, drawdown, WF leakage).
- README explaining timing model, data contract, artifacts, and assumptions.

### Files to create/modify
- Create: `src/backtest_lab/signals/align.py`, `src/backtest_lab/signals/technical.py`,
  `src/backtest_lab/signals/ml_ingest.py`, `src/backtest_lab/metrics/returns.py`,
  `src/backtest_lab/execution/constraints.py`, `src/backtest_lab/execution/costs.py`,
  `src/backtest_lab/metrics/tables.py`, `src/backtest_lab/walkforward/windows.py`,
  `src/backtest_lab/walkforward/engine.py`, new configs and tests, update report template.
- Modify: `src/backtest_lab/run.py`, `src/backtest_lab/signals/features.py`,
  `src/backtest_lab/strategies/*`, `src/backtest_lab/execution/accounting.py`,
  `src/backtest_lab/metrics/performance.py`, `src/backtest_lab/report/build.py`,
  `README.md`, and tests to match new artifacts.

### Key decisions (documented in code + README)
- Alignment contract: weights decided at end of day t apply to returns at t+1; `returns.csv`
  is timestamped by the realized return date (t+1).
- Missing return handling: if an asset return at t+1 is missing, its weight at t is set to 0
  (logged explicitly) rather than imputing a return.
- Universe lock per walk-forward window: computed once on the combined train+test slice and
  reused for all window operations; `asset_hash` logged per window.
