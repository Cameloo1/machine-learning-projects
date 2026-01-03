# Professional Backtest & Research Lab (Quant Rigor)

## Timing model (no leakage)
- Signals/features at decision time t use data available up to close of day t.
- Weights are decided end of day t.
- Realized portfolio return is applied on t+1:
  - `rp[t+1] = sum_i w_i[t] * r_i[t+1] - costs[t+1]`
- Alignment is centralized in `src/backtest_lab/signals/align.py` (single contract).
- `returns.csv` timestamps use the realized return date (t+1) and include `decision_ts` to map back to the decision date.

## Data format (long)
Required columns (data contract):
```
ts,symbol,open,high,low,close,volume
```
- `ts` must be parseable to datetime.
- Rows are sorted by `symbol, ts`.
- Long format is used everywhere (prices, features, weights, returns, trades).

## How to run
From repo root:
```
python scripts/download_market_data.py --tickers SPY,QQQ,IWM,EFA,EEM,TLT,GLD,USO,XLF,XLK --start 2016-01-01 --end 2024-01-01 --out-format csv --min-rows 252
python -m backtest_lab.run --config configs/sma_spy.yaml
python -m backtest_lab.run --config configs/rsi_spy.yaml
python -m backtest_lab.run --config configs/ew_baseline_multi.yaml
python -m backtest_lab.run --config configs/vol_target_multi.yaml
python -m backtest_lab.run --config configs/ml_gated_spy.yaml
```
Note: raw data is cached under `data/raw/yahoo`, with canonical output in
`data/processed/prices_long.csv`.
Optional walk-forward:
```
python -m backtest_lab.run --config configs/walkforward_sma_multi.yaml
python -m backtest_lab.run --config configs/walkforward_rsi_spy.yaml
python -m backtest_lab.run --config configs/walkforward_ew_multi.yaml
python scripts/run_sensitivity.py --config configs/vol_target_multi.yaml
python scripts/run_required_experiments.py
```

Strategies included: SMA trend, RSI mean reversion (stateful enter/exit), equal-weight baseline,
vol-target overlay (SMA base), and ML-gated signals (CSV predictions + probability threshold).
Sample predictions are in `data/processed/preds_spy_sample.csv`. Regenerate with:
```
python scripts/generate_sample_preds.py --prices data/processed/prices_long.csv --symbols SPY --out data/processed/preds_spy_sample.csv --start 2016-01-01 --end 2016-12-31
```

## Artifacts
Each run writes to `artifacts/<run_id>/`:
- `config.json` (resolved config with absolute paths)
- `run_metadata.json` (environment + config hash)
- `diagnostics.json` (validation, universe, alignment, warmup, benchmark, walk-forward)
- `data_integrity.json` (row counts, duplicate counts, summary stats before/after)
- `returns.csv` (ts,decision_ts,gross,net,exposure,turnover,txn_cost,slippage_cost,costs,...)
- `weights.csv` (ts,symbol,weight[,window_id])
- `trades.csv` (ts,symbol,weight,dw,abs_dw,txn_cost,slippage_cost,cost[,window_id])
- `metrics.csv`
- `report.html` (+ `plots/` PNGs)
Walk-forward runs include `walkforward.window_universe_diagnostics` entries with `asset_hash` per window.

## Tests
```
pytest -q -ra -W error::FutureWarning -W error::UserWarning -W error::DeprecationWarning
```

## Research report
See `reports/research_report.md` for the methodology, experiments, and limitations.

## Key assumptions & decisions
- Prices use `close` (no adjusted close unless provided in the input data).
- Missing data policy is explicit and logged via `data.universe.missing_data_policy`.
- Missing next-day returns: weights at t are set to 0 when `r_i[t+1]` is missing (logged as `alignment_forced_zero_count`).
- RSI signal: stateful long/flat. Enter long when `rsi < rsi_low`, exit to flat when `rsi > rsi_high`,
  otherwise hold prior state.
- Vol targeting uses rolling vol and a zero-correlation approximation for portfolio risk.
- ML-gated strategy expects a long-format predictions CSV with `ts,symbol,pred` and unique `(ts, symbol)` rows.
- Turnover definition: `turnover[t] = sum_i |w_i[t] - w_i[t-1]|`.
- Transaction costs: `cost_bps / 10000 * turnover[t]`.
- Slippage:
  - `bps`: `slip_bps / 10000 * |dw|`
  - `vol_prop`: `slip_mult * |dw| * rolling_vol`
- Execution caps are enforced in `execution/constraints.py` (clip → scale). `execution.renorm_policy` controls
  whether to scale down (`scale_down_if_exceeded`, default) or error (`error_if_exceeded`).
- `strategy_internal_risk_controls` (default false) keeps strategy outputs raw; set true to re-enable strategy
  caps (SMA trend only).
- Walk-forward universe selection defaults to `universe_selection_mode: train_only` to avoid test leakage;
  `window_full` retains the prior behavior.

## Metric definitions
- Annualized return: `mean(net) * 252`
- Annualized vol: `std(net) * sqrt(252)` (ddof=0)
- Sharpe: `mean(net) / std(net) * sqrt(252)` (rf=0)
- Sortino: `mean(net) / std(net<0) * sqrt(252)` (rf=0)
- Max drawdown: min of `equity / equity.cummax - 1`
- Win rate: fraction of `net` periods > 0
