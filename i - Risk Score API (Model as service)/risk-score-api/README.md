# Risk Score API

FastAPI service that scores SOC alerts as low, medium, or high priority using a trained tabular model.

## What It Provides

- JSON prediction endpoint
- simple HTML/HTMX demo UI
- model metadata endpoint
- training metrics endpoint
- tests with FastAPI `TestClient`

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/` for the demo UI
- `http://127.0.0.1:8000/docs` for Swagger UI

## API

```http
POST /predict
GET /model-info
GET /metrics
GET /
POST /predict-form
```

Example prediction payload:

```json
{
  "alert_type": "suspicious_login",
  "source_ip_risk": 0.75,
  "user_risk_score": 0.65,
  "failed_login_count_24h": 12,
  "geo_impossible_travel": 1,
  "asset_criticality": "high",
  "historical_false_positive_rate": 0.15
}
```

## Tests

```bash
pytest tests/
```

## Artifacts

- `models/model.joblib`
- `models/model_meta.json`
- `metrics/metrics.json`
- `data/alerts_sample.csv`

## Reproducibility Notes

The service ships with trained model artifacts. Retraining support exists in `train_model.py` and the notebook, but API verification should start with the test suite.
