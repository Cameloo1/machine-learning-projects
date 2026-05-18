# Backtest & Research Lab

Quant research project with explicit data contracts, no-lookahead signal alignment, execution costs, strategy configs, and validation tests.

## Core Contract

- Features/signals at decision time `t` use information available through close `t`.
- Weights are decided at `t`.
- Portfolio return is realized on `t+1`.
- `returns.csv` includes both realized `ts` and `decision_ts`.

## Data Format

Input prices use long format:

```text
ts,symbol,open,high,low,close,volume
```

Rows must be parseable by timestamp and sorted by `symbol, ts`.

## Install

```bash
pip install -e .
```

## Run Tests

```bash
pytest -q -ra -W error::FutureWarning -W error::UserWarning -W error::DeprecationWarning
```

## Run Backtests

```bash
python scripts/download_market_data.py --tickers SPY,QQQ,IWM,EFA,EEM,TLT,GLD,USO,XLF,XLK --start 2016-01-01 --end 2024-01-01 --out-format csv --min-rows 252
python -m backtest_lab.run --config configs/sma_spy.yaml
python -m backtest_lab.run --config configs/rsi_spy.yaml
python -m backtest_lab.run --config configs/ew_baseline_multi.yaml
python -m backtest_lab.run --config configs/vol_target_multi.yaml
python -m backtest_lab.run --config configs/ml_gated_spy.yaml
```

Optional:

```bash
python -m backtest_lab.run --config configs/walkforward_sma_multi.yaml
python -m backtest_lab.run --config configs/walkforward_rsi_spy.yaml
python -m backtest_lab.run --config configs/walkforward_ew_multi.yaml
python scripts/run_sensitivity.py --config configs/vol_target_multi.yaml
python scripts/run_required_experiments.py
```

## Outputs

Each run writes to `artifacts/<run_id>/`:

- resolved config and run metadata
- diagnostics and data-integrity JSON
- returns, weights, trades, and metrics CSV/JSON
- plots
- `report.html`

Generated `report.html` files are intentionally ignored at the repository root.

## Reports

- `reports/research_report.md` summarizes methodology and current findings.
- `SPEC_CHECKLIST.md` tracks implementation coverage.
- `VERIFICATION_REPORT.md` records the last stable verification snapshot.
