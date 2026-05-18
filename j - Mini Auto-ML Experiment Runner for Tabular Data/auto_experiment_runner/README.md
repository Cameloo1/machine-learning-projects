# Mini AutoML Experiment Runner

Small AutoML-style runner for tabular CSV experiments. It trains a configured set of scikit-learn models, saves metrics, and writes a concise report.

## Install

```bash
pip install -e .
```

## Run Demo Experiments

```bash
autotab --config configs/demo_classification.yaml
autotab --config configs/demo_regression.yaml
```

Equivalent module form:

```bash
python -m autotab.cli --config configs/demo_classification.yaml
```

## Tests

```bash
pytest
```

## Inputs

- configs under `configs/`
- example CSVs under `examples/data/`

## Outputs

Each run writes to `outputs/<problem-name>_<timestamp>/`:

- `config.yaml`
- `metadata.json`
- `leaderboard.csv`
- `leaderboard.json`
- `report.md`
- per-model metrics, plots, and `model.joblib`

## Reproducibility Notes

This project has the cleanest local proof path in the repo: install it, run tests, then run one demo config.
