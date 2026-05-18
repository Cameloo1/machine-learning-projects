# Synthetic SOC Alert Anomaly Detector

Offline security-analytics project that generates synthetic SOC events and evaluates unsupervised anomaly detectors.

## Models

- IsolationForest
- OneClassSVM

## Run

```bash
pip install -r requirements.txt
python scripts/generate_data.py --output-path data/soc_synthetic.csv --n-users 200 --events-per-user 200 --anomaly-fraction 0.01 --random-state 42
python scripts/run_anomaly_detection.py --data-path data/soc_synthetic.csv
```

For a small smoke run:

```bash
python scripts/generate_data.py --output-path data/verify_soc_synthetic.csv --n-users 20 --events-per-user 20 --anomaly-fraction 0.02 --random-state 42
python scripts/run_anomaly_detection.py --data-path data/verify_soc_synthetic.csv
```

## Outputs

- generated CSV under `data/`
- plots under `plots/`
- console metrics and local anomaly explanations

## Reproducibility Notes

Synthetic data generation is seed-controlled and does not require external data.
