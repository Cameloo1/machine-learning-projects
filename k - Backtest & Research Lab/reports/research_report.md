## Professional Backtest & Research Lab - Research Report

### 1) Problem statement and hypotheses
Goal: build a reproducible research lab that evaluates simple, interpretable
strategies under strict no-leakage rules, realistic trading frictions, and
walk-forward validation. The project should produce comparable artifacts for
single-asset and multi-asset portfolios, and provide diagnostics for data
integrity, alignment, and universe stability.

Hypotheses:
- Trend-following (SMA crossover) should outperform flat or mean-reversion
  regimes during sustained upward markets, with higher turnover in choppy
  regimes.
- RSI mean reversion (stateful enter/exit) should improve risk-adjusted returns
  in oscillatory regimes but may underperform in strong trends.
- Vol targeting should reduce drawdowns and smooth exposure by scaling risk to
  recent realized volatility.
- ML gating should improve risk-adjusted performance only when the model has
  predictive skill; otherwise, it reduces exposure and may reduce returns.

### 2) Data description and assumptions
Data contract:
- Long-format OHLCV with columns: `ts,symbol,open,high,low,close,volume`
- `ts` parsed to datetime and sorted by `symbol,ts`

Assumptions:
- Close-to-close returns drive signals and realized PnL.
- Missing next-day returns are treated as untradable; weights for that decision
  date are zeroed to avoid leakage.
- Missing data policy is explicit per config:
  - `drop_symbol`: remove assets with any missing rows in window
  - `drop_rows`: remove rows with missing data
  - `keep_gaps`: keep gaps but enforce minimum history

ML predictions:
- Predictions are long format `ts,symbol,pred` with unique `(ts, symbol)` rows.
- The sample predictions file is synthetic and deterministic, generated from
  lagged returns (see `scripts/generate_sample_preds.py`).

### 3) Strategy definitions (math/pseudocode)
SMA trend (long/flat):
```
if sma_fast[t] > sma_slow[t]: signal[t] = 1
else: signal[t] = 0
weight[t] = signal[t] / active_count[t]
```

RSI mean reversion (stateful enter/exit, long/flat):
```
if rsi[t] is NaN: position[t] = 0
else if rsi[t] < rsi_low: position[t] = 1
else if rsi[t] > rsi_high: position[t] = 0
else: position[t] = position[t-1]
weight[t] = position[t] / active_count[t]
```

Equal-weight baseline:
```
weight[t] = 1 / N_t   (daily or monthly rebalance, then forward-fill)
```

Vol target overlay:
```
vol[t] = rolling_std(ret[t - window + 1 : t]) * sqrt(252)
scale[t] = clip(target_vol / max(vol[t], min_vol), max_scale)
weight[t] = base_weight[t] * scale[t]
```

ML gating:
```
if pred[t] >= threshold: weight[t] = base_weight[t]
else: weight[t] = 0
```

### 4) Backtest design and leakage controls
Timing contract:
- Signals and features use data available through close of day t.
- Weights are decided end-of-day t.
- Realized return is applied on t+1:
  `rp[t+1] = sum_i w_i[t] * r_i[t+1] - costs[t+1]`

Alignment is centralized in `src/backtest_lab/signals/align.py`. The pipeline
adds `decision_ts` to returns, keeping both decision and realized timestamps.

Leakage controls:
- Indicators use rolling windows (no centered lookahead).
- Walk-forward enforces `train_end < test_start` and optional `val_end < test_start`.
- Validation selection uses only validation slices (no test data).

Dataflow (simplified):
```
prices -> validate -> universe -> features -> strategy -> weights
   -> align(t -> t+1) -> costs/constraints -> returns -> metrics/report
```

### 5) Results (IS vs OOS, WF summary)
In-sample (IS) results are captured for each run in:
- `artifacts/<run_id>/metrics.csv`
- `artifacts/<run_id>/report.html`

Walk-forward (OOS) runs aggregate per-window test slices into a single returns
series with `window_id` and include a per-window summary table in the HTML
report. Use:
- `configs/walkforward_sma_multi.yaml`
- `configs/walkforward_rsi_spy.yaml`
- `configs/walkforward_ew_multi.yaml`

Interpretation guidance:
- Compare net vs gross equity curves to quantify friction drag.
- Inspect drawdowns and rolling Sharpe to assess regime dependence.
- Confirm universe stability via per-window `asset_hash` diagnostics.

### 6) Sensitivity analysis (costs/slippage, thresholds)
Cost and slippage sensitivity is run via:
```
python scripts/run_sensitivity.py --config configs/vol_target_multi.yaml
```
This produces `artifacts/<run_id>_sensitivity_summary.csv`, which includes
total return, CAGR, Sharpe, max drawdown, turnover, and average costs.

Optional ML gating threshold sweeps can be performed by adjusting:
`strategy.params.threshold` in `configs/ml_gated_spy.yaml` and re-running.

### 7) Failure modes + limitations
- Indicator strategies can underperform in sideways or fast-reversing regimes.
- Vol targeting uses a simplified zero-correlation approximation and does not
  model portfolio covariance.
- ML gating is ingest-only; predictive skill is assumed, not trained here.
- Slippage models are simple and do not model market impact depth or liquidity.

### 8) Next steps
- Add portfolio-level risk models and sector caps.
- Expand signal set beyond SMA/RSI (e.g., cross-sectional factors).
- Integrate walk-forward ML training and calibration with strict leakage checks.
- Add robustness checks on different markets and frequencies.
