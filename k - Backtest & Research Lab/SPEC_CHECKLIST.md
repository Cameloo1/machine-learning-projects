## Professional Backtest & Research Lab (Quant Rigor) Spec Checklist

### Timing + alignment contract
- [x] Single alignment module with t->t+1 contract and diagnostics: `src/backtest_lab/signals/align.py`
- [x] Off-by-one alignment test present: `tests/test_alignment_contract.py`
- [x] Weights decided at t, returns realized at t+1 in accounting: `src/backtest_lab/execution/accounting.py`

### Config integrity checks
- [x] Config integrity guardrails executed on run: `src/backtest_lab/config.py`, `src/backtest_lab/run.py`

### No leakage
- [x] Features use only historical data (rolling, no centered windows): `src/backtest_lab/signals/technical.py`
- [x] Walk-forward leakage checks enforce train_end < test_start: `src/backtest_lab/walkforward/engine.py`
- [x] Walk-forward training excludes test data for model fitting; fit called once per window: `src/backtest_lab/walkforward/engine.py`
- [x] Validation selection (if enabled) evaluates only val window timestamps: `src/backtest_lab/walkforward/engine.py`

### Universe stability
- [x] Universe locked per window and reused for train/test: `src/backtest_lab/walkforward/engine.py`, `src/backtest_lab/data/universe.py`
- [x] final_assets + asset_hash logged per window: `src/backtest_lab/data/universe.py`, `src/backtest_lab/walkforward/engine.py`

### Missing data handling
- [x] Missing data policies explicit and logged: `src/backtest_lab/data/universe.py`
- [x] Missing returns zero-weighted and logged: `src/backtest_lab/signals/align.py`
- [x] Duplicate detection + data integrity report: `src/backtest_lab/data/validate.py`, `src/backtest_lab/run.py`

### Reproducibility
- [x] Data cached locally (CSV/yfinance cache): `src/backtest_lab/data/loader.py`
- [x] Resolved config + run metadata persisted: `src/backtest_lab/config.py`, `src/backtest_lab/run.py`
- [x] Artifacts include required CSVs + report for required runs (verified runtime)
- [x] Data integrity report artifact per run (verified runtime)

### Realism (costs/slippage)
- [x] Transaction costs + slippage models implemented: `src/backtest_lab/execution/costs.py`
- [x] Net vs gross returns and turnover computed: `src/backtest_lab/execution/accounting.py`

### Strategies
- [x] Equal-weight baseline with monthly/daily rebalance: `src/backtest_lab/strategies/equal_weight.py`
- [x] Vol-target overlay diagnostics include scale stats: `src/backtest_lab/strategies/overlays.py`
- [x] ML-gated strategy with CSV ingestion: `src/backtest_lab/strategies/ml_gated.py`

### Tests
- [x] Unit tests: alignment, costs/turnover, drawdown, constraints: `tests/test_alignment_contract.py`, `tests/test_costs_turnover.py`, `tests/test_drawdown.py`, `tests/test_constraints.py`
- [x] Integration: e2e smoke test: `tests/test_pipeline_e2e.py`
- [x] Integration: walk-forward no-leakage + universe lock: `tests/test_walkforward_integration.py`
- [x] pytest passing on clean run (verified runtime)

### Required run commands + outputs
- [x] `python -m backtest_lab.run --config configs/sma_spy.yaml` produces full artifacts (verified runtime)
- [x] `python -m backtest_lab.run --config configs/rsi_spy.yaml` produces full artifacts (verified runtime)
- [x] `python -m backtest_lab.run --config configs/vol_target_multi.yaml` produces full artifacts (verified runtime)
- [x] Walk-forward config produces >=6 windows (verified runtime)
