"""FastAPI application entrypoint."""
import json
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from app.config import settings
from app.core.logging_utils import get_logger, log_event
from app.core.model_loader import get_model_meta, get_metrics
from app.core.predict import predict_alert_priority
from app.schemas import AlertInput, MetricsResponse, ModelInfoResponse, PredictionResponse

# Initialize FastAPI app
app = FastAPI(
    title="Alert Risk Score API",
    version="1.0.0",
    description="API for predicting SOC alert priority using machine learning"
)

# Initialize Jinja2 templates
templates = Jinja2Templates(directory=str(settings.BASE_DIR / "app" / "templates"))

# Initialize logger
logger = get_logger("risk_score_api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the HTML demo UI.
    
    Returns:
        HTML template response with the demo form.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    alert_type: str = Form(...),
    source_ip_risk: float = Form(...),
    user_risk_score: float = Form(...),
    failed_login_count_24h: int = Form(...),
    geo_impossible_travel: int = Form(...),
    asset_criticality: str = Form(...),
    historical_false_positive_rate: float = Form(...)
):
    """
    Form-based prediction endpoint that returns HTML fragment.
    
    This endpoint accepts form-encoded data and returns an HTML snippet
    for HTMX to display. The /predict endpoint remains pure JSON.
    
    Args:
        Form fields matching AlertInput schema.
    
    Returns:
        HTML fragment with prediction results.
    """
    # Build AlertInput from form data
    alert_dict = {
        "alert_type": alert_type,
        "source_ip_risk": source_ip_risk,
        "user_risk_score": user_risk_score,
        "failed_login_count_24h": failed_login_count_24h,
        "geo_impossible_travel": geo_impossible_travel,
        "asset_criticality": asset_criticality,
        "historical_false_positive_rate": historical_false_positive_rate
    }
    
    # Validate and create AlertInput
    alert = AlertInput(**alert_dict)
    
    # Log the request
    log_event(
        logger,
        "predict_request",
        {
            "alert_type": alert.alert_type,
            "source_ip_risk": alert.source_ip_risk,
            "user_risk_score": alert.user_risk_score,
            "failed_login_count_24h": alert.failed_login_count_24h,
            "geo_impossible_travel": alert.geo_impossible_travel,
            "asset_criticality": alert.asset_criticality,
            "historical_false_positive_rate": alert.historical_false_positive_rate
        }
    )
    
    # Make prediction
    result = predict_alert_priority(alert.dict())
    
    # Log the response
    log_event(
        logger,
        "predict_response",
        {
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }
    )
    
    # Build HTML fragment
    label_upper = result["label"].upper()
    confidence_pct = result["confidence"] * 100
    
    # Color based on priority
    if result["label"] == "high":
        label_color = "#ff4444"  # Red
    elif result["label"] == "medium":
        label_color = "#ffaa00"  # Orange
    else:
        label_color = "#44ff44"  # Green
    
    html = f"""
    <div style="margin-top: 20px; padding: 20px; border: 1px solid #45b3d6; background: #1a1b26;">
        <h3 style="margin-top: 0; color: {label_color}; text-transform: uppercase; letter-spacing: 2px;">
            Priority: {label_upper}
        </h3>
        <p style="color: #c5c6c7; margin: 10px 0;">
            <strong>Confidence:</strong> {confidence_pct:.2f}%
        </p>
        <p style="color: #c5c6c7; margin: 10px 0;">
            <strong>Explanation:</strong> {result["explanation"]}
        </p>
        <details style="margin-top: 15px;">
            <summary style="color: #45b3d6; cursor: pointer; padding: 5px 0;">
                Raw Probabilities (JSON)
            </summary>
            <pre style="background: #0b0c10; padding: 10px; border: 1px solid #45b3d6; margin-top: 10px; color: #c5c6c7; overflow-x: auto;">{json.dumps(result["probabilities"], indent=2)}</pre>
        </details>
    </div>
    """
    
    return HTMLResponse(content=html)


@app.post("/predict", response_model=PredictionResponse)
async def predict(alert: AlertInput):
    """
    Predict alert priority from alert features.
    
    Args:
        alert: AlertInput containing alert features.
    
    Returns:
        PredictionResponse with predicted label, confidence, probabilities, and explanation.
    """
    # Log the request
    log_event(
        logger,
        "predict_request",
        {
            "alert_type": alert.alert_type,
            "source_ip_risk": alert.source_ip_risk,
            "user_risk_score": alert.user_risk_score,
            "failed_login_count_24h": alert.failed_login_count_24h,
            "geo_impossible_travel": alert.geo_impossible_travel,
            "asset_criticality": alert.asset_criticality,
            "historical_false_positive_rate": alert.historical_false_positive_rate
        }
    )
    
    # Make prediction
    result = predict_alert_priority(alert.dict())
    
    # Log the response
    log_event(
        logger,
        "predict_response",
        {
            "label": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }
    )
    
    # Return as Pydantic model
    return PredictionResponse(**result)


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get model metadata information.
    
    Returns:
        ModelInfoResponse containing model features, target, classes, and sample counts.
    """
    meta = get_model_meta()
    return ModelInfoResponse(**meta)


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """
    Get training metrics.
    
    Returns:
        MetricsResponse containing classification report and confusion matrix.
    """
    metrics_data = get_metrics()
    return MetricsResponse(**metrics_data)
