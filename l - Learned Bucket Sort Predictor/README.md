# Learned Bucket Sort Predictor

A learned-index take on bucket sort. Instead of assigning elements to buckets with a fixed
formula like `floor(N * (x - min) / (max - min))`, this project trains a regression model to
approximate the empirical CDF of the data and uses that model as the bucketing function. On
non-uniform data the learned bucketer spreads elements far more evenly than the analytic
formula, which is what bucket sort needs to hit its near-linear best case.

**Thesis:** the classical bucketing step hardcodes an assumption (data is uniform). Replace that
assumption with a model that learns the actual distribution, and the downstream sort gets cheaper.

## Focus

| Item              | Detail                                                              |
| ----------------- | ------------------------------------------------------------------- |
| Core idea         | Regression model of the CDF used as a bucket-assignment function    |
| Baseline          | Analytic min-max bucketing (`floor(N * normalized_value)`)          |
| Learned variant   | Linear Regression and a small MLP fit to the empirical CDF          |
| Primary metric    | Bucket-occupancy balance (variance / max-min spread across buckets) |
| Secondary metrics | Total sort time, intra-bucket comparison count                      |
| Default proof path | offline synthetic-data run                                          |

## What it does

1. Generate a dataset under a chosen distribution (uniform, gaussian, lognormal, bimodal, clustered).
2. Fit the empirical CDF: sort a sample, pair each value with its normalized rank, and train a
   regressor to map `value -> rank in [0, 1]`.
3. Bucket assignment: `bucket(x) = clamp(floor(N * model.predict(x)), 0, N-1)`. A perfect CDF model
   maps the data to a uniform distribution over `[0, 1]`, so buckets fill evenly by construction.
4. Sort within each bucket (insertion sort by default) and concatenate.
5. Benchmark the learned bucketer against the analytic baseline on identical inputs and report
   occupancy balance and sort time across every distribution.

## Layout

```
l - Learned Bucket Sort Predictor/
├── learned_bucket_sort/
│   ├── data.py          # distribution generators
│   ├── baseline.py      # analytic min-max bucket sort
│   ├── cdf_model.py     # Linear Regression + MLP CDF estimators
│   ├── learned_sort.py  # learned bucketing + intra-bucket sort
│   └── benchmark.py     # baseline vs. learned, metrics + plots
├── tests/
│   └── test_learned_sort.py
├── artifacts/           # generated plots + metrics json (see artifact policy)
├── requirements.txt
└── README.md
```

## Run

From this project folder:

```bash
pip install -r requirements.txt

# Reproduce the full benchmark across all distributions
python -m learned_bucket_sort.benchmark --n 100000 --buckets 1000 --out artifacts/

# Single distribution, quick check
python -m learned_bucket_sort.benchmark --dist lognormal --n 20000 --buckets 200
```

Tests:

```bash
python -m pytest tests/ -q
```

## What to expect

On uniform data the learned bucketer roughly ties the analytic baseline — there is nothing to
learn. On skewed distributions (lognormal, bimodal, clustered) the analytic formula piles most
elements into a few buckets while leaving others empty; the learned bucketer flattens occupancy
toward uniform, cutting intra-bucket comparison counts and overall sort time. The benchmark
emits an occupancy histogram per method and a metrics summary so the win (and where there is no
win) is visible rather than asserted.

## Correctness

The learned variant is verified to produce output identical to a reference sort on every
distribution — bucketing only changes *how* elements are partitioned, never the final order.
Tests assert full sorted-order equality plus occupancy-balance improvement on skewed inputs.

## Artifact Policy

Source, tests, configs, and this README are tracked. Generated plots and the metrics JSON in
`artifacts/` are portfolio evidence and tracked intentionally; large regenerated runs are not.
Use `projects.json` at the repo root as the source of truth for the current verification command.

## Tech

Python, scikit-learn (LinearRegression, MLPRegressor), NumPy, Matplotlib. No network or live data.