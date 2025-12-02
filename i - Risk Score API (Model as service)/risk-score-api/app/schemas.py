"""Pydantic models for request/response validation."""
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, confloat, conint


class AlertInput(BaseModel):
    """Input schema for alert features."""
    
    alert_type: Literal["brute_force", "malware", "suspicious_login", "data_exfil", "other"]
    source_ip_risk: confloat(ge=0, le=1)
    user_risk_score: confloat(ge=0, le=1)
    failed_login_count_24h: conint(ge=0)
    geo_impossible_travel: Literal[0, 1]
    asset_criticality: Literal["low", "medium", "high"]
    historical_false_positive_rate: confloat(ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_type": "suspicious_login",
                "source_ip_risk": 0.75,
                "user_risk_score": 0.65,
                "failed_login_count_24h": 12,
                "geo_impossible_travel": 1,
                "asset_criticality": "high",
                "historical_false_positive_rate": 0.15
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""
    
    label: Literal["low", "medium", "high"]
    confidence: float
    probabilities: Dict[str, float]
    explanation: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "high",
                "confidence": 0.89,
                "probabilities": {
                    "low": 0.05,
                    "medium": 0.06,
                    "high": 0.89
                },
                "explanation": "Predicted high priority (confidence=0.89) based on: high asset criticality; elevated user risk score; multiple failed login attempts."
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata."""
    
    features: Dict[str, List[str]]
    target: str
    classes: List[str]
    training_samples: int
    validation_samples: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "numeric": ["source_ip_risk", "user_risk_score", "failed_login_count_24h", "historical_false_positive_rate"],
                    "categorical": ["alert_type", "geo_impossible_travel", "asset_criticality"]
                },
                "target": "priority",
                "classes": ["high", "low", "medium"],
                "training_samples": 5000,
                "validation_samples": 1250
            }
        }


class MetricsResponse(BaseModel):
    """Response schema for training metrics."""
    
    classification_report: Dict[str, Any]  # Can contain both dicts and floats (e.g., accuracy)
    confusion_matrix: List[List[int]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "classification_report": {
                    "high": {
                        "precision": 0.85,
                        "recall": 0.82,
                        "f1-score": 0.83,
                        "support": 400
                    }
                },
                "confusion_matrix": [
                    [350, 30, 20],
                    [25, 380, 15],
                    [10, 15, 375]
                ]
            }
        }
