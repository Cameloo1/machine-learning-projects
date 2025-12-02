# Alert Risk Score API

A production-minded machine learning service that scores SOC/security alerts by priority (low/medium/high) using a trained tabular ML model. The service exposes a FastAPI API with a simple HTML+HTMX demo UI for interactive testing.

## Project Overview

The Alert Risk Score API is an end-to-end machine learning service designed to help Security Operations Centers (SOC) prioritize alerts by automatically assigning risk scores. The system uses a gradient-boosted classifier trained on synthetic SOC alert data to predict whether an alert should be classified as **low**, **medium**, or **high** priority.

This project demonstrates:
- **ML Engineering**: Reproducible training pipeline, model versioning, and metadata tracking
- **API Development**: RESTful API with Pydantic validation, JSON logging, and interactive documentation
- **Production Practices**: Type hints, comprehensive testing, structured logging, and clear documentation

## ML Model

### Features

The model uses 7 features extracted from SOC alerts:

**Numeric Features:**
- `source_ip_risk` (0-1): Risk score associated with the source IP address
- `user_risk_score` (0-1): Risk score associated with the user account
- `failed_login_count_24h` (≥0): Number of failed login attempts in the last 24 hours
- `historical_false_positive_rate` (0-1): Historical rate of false positives for similar alerts

**Categorical Features:**
- `alert_type`: Type of alert (`brute_force`, `malware`, `suspicious_login`, `data_exfil`, `other`)
- `geo_impossible_travel` (0/1): Binary flag indicating impossible geographic travel detected
- `asset_criticality`: Criticality level of the affected asset (`low`, `medium`, `high`)

### Target

- **Priority**: Three-class classification problem
  - `low`: Low-priority alerts that can be reviewed later
  - `medium`: Medium-priority alerts requiring attention
  - `high`: High-priority alerts requiring immediate investigation

### Algorithm

- **Model**: `HistGradientBoostingClassifier` from scikit-learn
- **Preprocessing**:
  - Numeric features: StandardScaler
  - Categorical features: OneHotEncoder
- **Pipeline**: End-to-end scikit-learn Pipeline for reproducibility

### Key Metrics

Training metrics are saved in `metrics/metrics.json` and include:
- **Classification Report**: Precision, recall, and F1-score for each class
- **Confusion Matrix**: Per-class prediction accuracy
- **Weighted Average Metrics**: Overall model performance

The model metadata (`models/model_meta.json`) contains feature lists, class labels, and training/validation sample counts, enabling full explainability and reproducibility.

## API Endpoints

### `POST /predict`
Predict alert priority from JSON payload.

**Request:**
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

**Response:**
```json
{
  "label": "high",
  "confidence": 0.89,
  "probabilities": {
    "low": 0.05,
    "medium": 0.06,
    "high": 0.89
  },
  "explanation": "Predicted high priority (confidence=0.89) based on: high asset criticality; elevated user risk score; multiple failed login attempts."
}
```

### `GET /model-info`
Returns model metadata including features, target, classes, and sample counts.

**Response:**
```json
{
  "features": {
    "numeric": ["source_ip_risk", "user_risk_score", "failed_login_count_24h", "historical_false_positive_rate"],
    "categorical": ["alert_type", "geo_impossible_travel", "asset_criticality"]
  },
  "target": "priority",
  "classes": ["high", "low", "medium"],
  "training_samples": 5000,
  "validation_samples": 1250
}
```

### `GET /metrics`
Returns training metrics including classification report and confusion matrix.

### `GET /`
Serves the interactive HTML demo UI with HTMX form for manual alert scoring.

### `POST /predict-form`
Form-based endpoint (used by the HTML UI) that accepts form-encoded data and returns an HTML fragment.

## How to Run

### Step 1: Setup Environment

1. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

**Option A: Using Jupyter Notebook (Recommended)**
1. Open `notebooks/01_train_alert_model.ipynb`
2. Execute all cells sequentially
3. This will generate:
   - `data/alerts_sample.csv` - Synthetic training dataset
   - `models/model.joblib` - Trained model
   - `models/model_meta.json` - Model metadata
   - `metrics/metrics.json` - Training metrics

**Option B: Using Python Script (Alternative)**
If you prefer a script-based approach, you can convert the notebook cells into a Python script and run:
```bash
python train_model.py  # (if created)
```

### Step 3: Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

The `--reload` flag enables auto-reload during development.

### Step 4: Access the Application

- **Interactive API Documentation (Swagger UI)**: http://127.0.0.1:8000/docs
- **Alternative API Documentation (ReDoc)**: http://127.0.0.1:8000/redoc
- **Demo UI**: http://127.0.0.1:8000/
- **API Base**: http://127.0.0.1:8000/

### Step 5: Test the API

1. **Via Swagger UI**: Visit `/docs` to interactively test all endpoints
2. **Via Demo UI**: Visit `/` to use the HTML form with HTMX
3. **Via cURL or Postman**: Send POST requests to `/predict` with JSON payloads

### Running Tests

```bash
pytest tests/
```

## High-Level Architecture

The project follows a two-phase architecture:

### 1. Offline Training Phase

- **Data**: Synthetic or real-like SOC alert dataset with realistic features
- **Model**: Gradient-boosted classifier (HistGradientBoostingClassifier) trained to predict alert priority
- **Outputs**:
  - `models/model.joblib` - Serialized trained model
  - `models/model_meta.json` - Model metadata (features, classes, sample counts, etc.)
  - `metrics/metrics.json` - Training metrics (classification report, confusion matrix)

### 2. Online Inference Phase

- **FastAPI Application** exposing:
  - `GET /` - HTML page with form using HTMX to call `/predict-form`
  - `POST /predict` - Accepts JSON alert features, returns priority label, confidence, probabilities, and explanation
  - `POST /predict-form` - Form-based endpoint for HTML UI
  - `GET /model-info` - Returns model metadata as JSON
  - `GET /metrics` - Returns training metrics as JSON

- **Core Features**:
  - Lazy-loading and caching of model, metadata, and metrics at startup
  - Pydantic validation for request/response schemas
  - JSON logging of all requests and responses
  - Human-readable explanation strings for predictions

## Project Structure

```
risk-score-api/
├─ data/
│  └─ alerts_sample.csv               # Synthetic or real-like training data
├─ models/
│  ├─ model.joblib                    # Trained model (generated by training)
│  └─ model_meta.json                 # Model metadata (generated by training)
├─ metrics/
│  └─ metrics.json                    # Training metrics (generated by training)
├─ notebooks/
│  └─ 01_train_alert_model.ipynb      # Training & evaluation notebook
├─ app/
│  ├─ __init__.py
│  ├─ main.py                         # FastAPI entrypoint
│  ├─ config.py                       # Settings (paths, env)
│  ├─ schemas.py                      # Pydantic models for request/response
│  ├─ core/
│  │  ├─ model_loader.py              # Loading model/meta/metrics with caching
│  │  ├─ predict.py                   # Inference + explanation logic
│  │  └─ logging_utils.py             # JSON logging setup
│  └─ templates/
│     └─ index.html                   # HTMX demo UI
├─ tests/
│  └─ test_api.py                     # Basic API tests using TestClient
├─ requirements.txt
└─ README.md
```

## Model Explainability

The API provides explainability through:

1. **Human-Readable Explanations**: Each prediction includes a rule-based explanation highlighting key factors (e.g., "high asset criticality", "elevated user risk score")

2. **Probability Distributions**: Full probability scores for all classes are returned, showing model confidence

3. **Model Metadata**: `model_meta.json` contains complete feature lists and class information

4. **Training Metrics**: `metrics.json` provides detailed performance metrics including per-class precision, recall, and F1-scores

These artifacts can be used to:
- Demonstrate model quality to stakeholders
- Debug prediction behavior
- Validate model performance
- Support regulatory compliance requirements

## Environment Configuration

The application uses Pydantic BaseSettings for configuration. You can override defaults via environment variables or a `.env` file:

- `MODEL_PATH` - Path to model.joblib (default: `models/model.joblib`)
- `MODEL_META_PATH` - Path to model_meta.json (default: `models/model_meta.json`)
- `METRICS_PATH` - Path to metrics.json (default: `metrics/metrics.json`)
- `ENV` - Environment name (default: `local`)

## Development

### Code Organization

- **Training**: All model training logic is in the Jupyter notebook for reproducibility
- **Inference**: Core prediction logic is in `app/core/predict.py`
- **API**: FastAPI routes and schemas are in `app/main.py` and `app/schemas.py`
- **Configuration**: Centralized in `app/config.py` with environment variable support

### Code Quality

- Type hints throughout the codebase
- Comprehensive docstrings for core functions
- Pydantic models for request/response validation
- Structured JSON logging for observability
- Unit tests for API endpoints

## License

This project is intended for portfolio/demonstration purposes.
