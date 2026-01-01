# Professional Backtest & Research Lab (Quant Rigor)

## Timing model (no leakage)
- Signals/features at decision time t use data available up to close of day t.
- Weights are decided end of day t.
- Realized portfolio return is applied on t+1:
  - `rp[t+1] = sum_i w_i[t] * r_i[t+1] - costs[t+1]`
- Alignment is centralized in `src/backtest_lab/signals/align.py` (single contract).

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
python -m backtest_lab.run --config configs/sma_spy.yaml
python -m backtest_lab.run --config configs/rsi_spy.yaml
python -m backtest_lab.run --config configs/vol_target_multi.yaml
```
Note: yfinance data is cached locally under `data/raw/yahoo` for reproducibility.
Optional walk-forward:
```
python -m backtest_lab.run --config configs/walkforward_sma_multi.yaml
```

## Artifacts
Each run writes to `artifacts/<run_id>/`:
- `config.json` (resolved config with absolute paths)
- `run_metadata.json` (environment + config hash)
- `diagnostics.json` (validation, universe, alignment, walk-forward)
- `returns.csv` (ts,gross,net,turnover,costs,txn_cost,slippage_cost,gross_exposure,...)
- `weights.csv` (ts,symbol,weight)
- `trades.csv` (ts,symbol,dw,abs_dw,txn_cost,slippage_cost,cost)
- `metrics.csv`
- `report.html` (+ `plots/` PNGs)

## Tests
```
pytest -q -ra -W error::FutureWarning -W error::UserWarning -W error::DeprecationWarning
```

## Key assumptions & decisions
- Prices use `close` (no adjusted close unless provided in the input data).
- Missing data policy is explicit and logged via `data.universe.missing_data_policy`.
- Missing next-day returns: weights at t are set to 0 when `r_i[t+1]` is missing (logged).
- Transaction costs: `cost_bps / 10000 * turnover[t]`.
- Slippage:
  - `bps`: `slippage_bps / 10000 * |dw|`
  - `vol_prop`: `slip_mult * |dw| * rolling_vol`
- Cost sensitivity: rerun the same config with different `execution.cost_bps` values
  (e.g., 0, 2, 5) for a three-point sweep.
