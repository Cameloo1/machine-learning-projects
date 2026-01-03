## Verification Report

### Summary
- Alignment/leakage contract enforced and tested; walk-forward windows validated.
- Required configs run successfully and produce full artifacts + reports.
- Duplicate detection and data integrity reporting implemented and verified.
- Costs/slippage affect net returns; turnover is non-zero for active strategies.
- Walk-forward fit/validation selection added with diagnostics and tests.
- Equal-weight baseline, sensitivity harness, and reporting upgrades implemented.

### Commands Run
```
pytest -q
python -m backtest_lab.run --config configs/sma_spy.yaml
python -m backtest_lab.run --config configs/walkforward_rsi_spy.yaml
```

### Config Integrity Checks
- `artifacts/*/config_integrity.json` written for every run.
- Example (sma_spy): `artifacts/sma_spy/config_integrity.json`

### Data Integrity / Duplicate Audit
- Duplicate detection before cleaning is now explicit and logged.
- Conflicting duplicate OHLCV values raise an error to prevent silent mutation.
- Data integrity report generated per run: `artifacts/<run_id>/data_integrity.json`
  - `duplicate_count_before`: 0 across all required runs
  - `changed_value_rows`: all zeros across OHLCV columns

### Alignment + Leakage Audits
- Alignment contract: tests `tests/test_alignment_contract.py` and `tests/test_alignment_toy.py`
  verify weights at t apply to returns at t+1, with `decision_ts` carried.
- No lookahead in features: `tests/test_features_no_lookahead.py` checks SMA uses
  only historical data.
- Walk-forward: diagnostics show `train_end < test_start` for every window.
  - Verified in `artifacts/walkforward_rsi_spy/diagnostics.json` (26 windows).
- Validation selection uses only val timestamps (tested in `tests/test_walkforward_val_selection_no_leakage.py`).

### Realism Checks
- Net vs gross returns differ when costs > 0:
  - `artifacts/sma_spy/returns.csv` and `artifacts/rsi_spy/returns.csv` show
    `net` < `gross` on average.
- Turnover > 0 for active strategies:
  - `returns.csv` `turnover` mean is positive for SMA/RSI/vol-target runs.
- Cost sensitivity harness produces 3 runs + summary CSV:
  - `scripts/run_sensitivity.py` and `tests/test_sensitivity_harness_outputs.py`

### Issues Found and Fixes Applied
1) **Non-price CSVs in yfinance cache caused loader failure**
   - Impact: `vol_target_multi` crashed when encountering `SPY_actions.csv`.
   - Fix: skip non-price CSVs in `src/backtest_lab/data/loader.py`.
   - Snippet:
     ```
     if not _csv_has_required_cols(csv_file):
         logger.warning("Skipping non-price CSV file: %s", csv_file)
         continue
     ```

2) **No explicit duplicate conflict policy / integrity report**
   - Impact: duplicate OHLCV conflicts could silently corrupt data.
   - Fix: conflict detection + integrity report in `src/backtest_lab/data/validate.py`,
     persisted by `src/backtest_lab/run.py` to `data_integrity.json`.

3) **Config integrity not enforced**
   - Impact: invalid min_history vs feature windows could slip through.
   - Fix: `check_config_integrity()` in `src/backtest_lab/config.py` and persisted
     `config_integrity.json` in `src/backtest_lab/run.py`.

### Data Wiring Updates
- Replaced prior ad-hoc data with the dedicated downloader output.
- Configs now point to `data/processed/prices_long.csv` as the canonical dataset,
  per the downloader’s own integration note.

Config diffs (wiring only):
```
configs/sma_spy.yaml
  data.prices_path: ../data/raw/spy.csv -> ../data/processed/prices_long.csv
configs/rsi_spy.yaml
  data.prices_path: ../data/raw/spy.csv -> ../data/processed/prices_long.csv
configs/vol_target_multi.yaml
  data.mode: yfinance -> csv
  data.prices_path: (none) -> ../data/processed/prices_long.csv
configs/walkforward_sma_multi.yaml
  data.mode: yfinance -> csv
  data.prices_path: (none) -> ../data/processed/prices_long.csv
```

### Limitations / Notes
- Yahoo Finance API responses were unstable (JSON decode errors); the downloader
  successfully used the built-in direct chart fallback and still produced a
  validated `data/processed/prices_long.csv`.
