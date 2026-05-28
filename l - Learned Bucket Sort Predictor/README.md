# Learned Bucket Sort Predictor

Learned bucket-sort benchmark for studying how bucket assignment affects sort
work.

The project compares a classical min-max bucket index with learned CDF bucket
indexes across controlled distributions and realistic synthetic scenarios.

## Why This Exists

Bucket sort is fast when values are spread evenly across buckets. Min-max
bucketing assumes the data is roughly uniform across the value range. Skewed,
clustered, or long-tailed data can overload a few buckets and leave others
empty.

This project tests a different index:

```text
classic bucket = floor(bucket_count * minmax_normalize(x))
learned bucket = floor(bucket_count * learned_cdf(x))
```

A CDF maps values to approximate ranks. Rank space is closer to uniform, so a
good CDF model can spread bucket work more evenly.

## Current Result

- `linear_cdf` is the current practical runtime win.
- `mlp_cdf` usually gives the strongest bucket-quality result.
- `mlp_cdf` is not a proven speed win in this implementation because prediction
  and training cost still matter.
- Every benchmark row must remain correct against a reference sort.

For the guided explanation and evidence plots, read
[ML project walkthrough](docs/ml-project-walkthrough.md).

For a browser-based visual demo, open [mockup/index.html](mockup/index.html).
The browser demo is educational only; it does not run the Python models.

## Methods

- `analytic_baseline`: fixed min-max bucket assignment.
- `linear_cdf`: CPU linear CDF estimator.
- `mlp_cdf`: optional Torch MLP CDF estimator on CPU or CUDA.

Torch is optional. Baseline and linear benchmarks run without it. The MLP path
requires a working local PyTorch install.

## Quick Start

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Run a controlled benchmark without Torch:

```powershell
.\.venv\Scripts\python.exe -m learned_bucket_sort.benchmark --dist lognormal --n 5000 --buckets 50 --seed 11 --method linear
```

Run all benchmark methods. This includes `mlp_cdf` and requires PyTorch:

```powershell
.\.venv\Scripts\python.exe -m learned_bucket_sort.benchmark --dist lognormal --n 5000 --buckets 50 --seed 11 --method all
```

Disable console color:

```powershell
.\.venv\Scripts\python.exe -m learned_bucket_sort.benchmark --dist lognormal --n 5000 --buckets 50 --seed 11 --method all --no-color
```

## Scenario Data

Realistic scenario datasets are synthetic and local. Generator scripts are
tracked; generated `.npy` files and manifests are ignored.

Generate one scenario:

```powershell
.\.venv\Scripts\python.exe scripts\generate_scenario_datasets.py --scenario response_times --n 5000 --seed 11
```

Benchmark one scenario:

```powershell
.\.venv\Scripts\python.exe -m learned_bucket_sort.benchmark --scenario response_times --n 5000 --buckets 50 --seed 11
```

Generate and benchmark all scenarios:

```powershell
.\.venv\Scripts\python.exe scripts\generate_scenario_datasets.py --scenario all --n 5000 --seed 11
$manifest = Get-ChildItem datasets\generated\manifest_scenarios_n5000_seed11_*.json | Sort-Object Name | Select-Object -Last 1
.\.venv\Scripts\python.exe -m learned_bucket_sort.benchmark --scenario all --n 5000 --buckets 50 --seed 11 --manifest $manifest.FullName --method all
```

## Evidence Commands

Run the scale-closure benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\run_scale_closure.py
```

Run the train-once amortized benchmark:

```powershell
.\.venv\Scripts\python.exe -m learned_bucket_sort.amortized_benchmark --dist lognormal --n 50000 --buckets 100 --train-seed 11 --eval-seed 12 --method all
```

The script wrapper is also available:

```powershell
.\.venv\Scripts\python.exe scripts\run_amortized_benchmark.py --dist lognormal --n 50000 --buckets 100 --train-seed 11 --eval-seed 12 --method all
```

Regenerate the promoted evidence plots:

```powershell
.\.venv\Scripts\python.exe scripts\run_part5_evidence.py
```

This writes a local ignored evidence bundle and promotes selected PNGs into
`assets/`.

## Benchmark Output

Regular benchmark columns:

- `distribution`: controlled distribution or scenario name
- `method`: `analytic_baseline`, `linear_cdf`, or `mlp_cdf`
- `device`: `cpu`, `cuda`, or `-` when no model device applies
- `n`: item count
- `buckets`: bucket count
- `fit_ms`: model fit time
- `bucket_ms`: bucket assignment time
- `sort_ms`: sorting time inside buckets
- `total_ms`: reported method time
- `variance`: bucket-size variance
- `max_bucket`: largest bucket size
- `empty`: empty bucket count
- `ok`: sorted output matched the reference sort

Scenario benchmark JSON also records `dataset_file`, the exact NPY file used.
JSON artifacts record `device`; baseline rows use `null`.

Interactive console output highlights best and worst problem metrics. Redirected
output, `NO_COLOR`, and `--no-color` produce plain text. JSON artifacts never
include color metadata.

Amortized benchmark columns separate one-time training from reused sorting:

- `train_ms`: one-time fit cost
- `predict_ms`: reused-model inference cost
- `index_ms`: vectorized bucket-index calculation cost
- `group_ms`: bucket grouping cost
- `sort_path_total_ms`: prediction plus bucketing plus bucket sorting
- `end_to_end_total_ms`: training plus reused sort path

## Practice Problems

Useful LeetCode problems for the ideas behind this project:

| Problem | Why it helps |
| --- | --- |
| 164. Maximum Gap | bucket sort and value-range partitioning |
| 347. Top K Frequent Elements | grouping items into buckets by a derived score |
| 220. Contains Duplicate III | bucketization with fixed-width value ranges |

## Project Layout

```text
learned_bucket_sort/
  benchmark.py
  amortized_benchmark.py
  baseline.py
  cdf_model.py
  data.py
  learned_sort.py
  metrics.py
  part5_evidence.py
  scale_closure.py
  scenarios.py
  torch_mlp_cdf.py
scripts/
  generate_scenario_datasets.py
  run_amortized_benchmark.py
  run_part5_evidence.py
  run_scale_closure.py
tests/
  test_*.py
datasets/
  generated/
    .gitkeep
artifacts/
  .gitkeep
assets/
  promoted evidence images
docs/
  ml-project-walkthrough.md
mockup/
  index.html
```

## Artifact Policy

Tracked source is code, tests, config, stable docs, and intentionally promoted
assets. Generated benchmark JSON files, evidence bundles, scenario datasets,
test-run output, caches, and virtual environments stay local unless explicitly
promoted.
