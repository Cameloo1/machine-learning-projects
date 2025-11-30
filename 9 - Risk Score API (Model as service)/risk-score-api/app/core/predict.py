"""Prediction and explanation logic."""
import pandas as pd
from typing import Dict, Any

from app.core.model_loader import get_model, get_model_meta


def predict_alert_priority(alert_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict alert priority from alert features.
    
    Args:
        alert_features: Dictionary containing alert features:
            - alert_type (str)
            - source_ip_risk (float)
            - user_risk_score (float)
            - failed_login_count_24h (int)
            - geo_impossible_travel (int: 0 or 1)
            - asset_criticality (str)
            - historical_false_positive_rate (float)
    
    Returns:
        Dictionary with:
            - label: Predicted priority class (str)
            - confidence: Confidence score (float)
            - probabilities: Dict mapping class labels to probabilities
            - explanation: Human-readable explanation string
    """
    # Load model and metadata
    model = get_model()
    meta = get_model_meta()
    
    # Get class labels in model order
    classes = meta["classes"]
    
    # Create single-row DataFrame from features
    # Use the exact feature order from training (matches training notebook)
    feature_order = [
        "alert_type",
        "source_ip_risk",
        "user_risk_score",
        "failed_login_count_24h",
        "geo_impossible_travel",
        "asset_criticality",
        "historical_false_positive_rate"
    ]
    
    # Create DataFrame with proper column order
    df = pd.DataFrame([alert_features], columns=feature_order)
    
    # Get prediction probabilities
    probabilities_array = model.predict_proba(df)
    
    # Convert to dictionary mapping class labels to probabilities
    probabilities = {
        class_label: float(prob)
        for class_label, prob in zip(classes, probabilities_array[0])
    }
    
    # Find class with highest probability
    label = classes[probabilities_array[0].argmax()]
    confidence = float(probabilities_array[0].max())
    
    # Build explanation
    explanation = build_simple_explanation(alert_features, label, confidence)
    
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "explanation": explanation
    }


def build_simple_explanation(
    features: Dict[str, Any],
    label: str,
    confidence: float
) -> str:
    """
    Build a human-readable explanation for the prediction.
    
    Args:
        features: Dictionary of alert features.
        label: Predicted priority label.
        confidence: Confidence score.
        
    Returns:
        Human-readable explanation string.
    """
    reasons = []
    
    if label == "high":
        # High priority reasons
        if features.get("asset_criticality") == "high":
            reasons.append("high asset criticality")
        
        if features.get("user_risk_score", 0) > 0.7:
            reasons.append("elevated user risk score")
        
        if features.get("failed_login_count_24h", 0) > 10:
            reasons.append("multiple failed login attempts")
        
        if features.get("geo_impossible_travel", 0) == 1:
            reasons.append("impossible geographic travel detected")
        
        if features.get("source_ip_risk", 0) > 0.7:
            reasons.append("high source IP risk")
        
        if features.get("historical_false_positive_rate", 1.0) < 0.2:
            reasons.append("low historical false positive rate")
    
    elif label == "medium":
        # Medium priority reasons
        if features.get("asset_criticality") == "medium":
            reasons.append("moderate asset criticality")
        
        if 0.4 < features.get("user_risk_score", 0) <= 0.7:
            reasons.append("moderate user risk score")
        
        if 5 < features.get("failed_login_count_24h", 0) <= 10:
            reasons.append("some failed login attempts")
        
        if 0.3 < features.get("source_ip_risk", 0) <= 0.7:
            reasons.append("moderate source IP risk")
        
        if 0.2 <= features.get("historical_false_positive_rate", 1.0) < 0.5:
            reasons.append("moderate historical false positive rate")
    
    elif label == "low":
        # Low priority reasons
        if features.get("asset_criticality") == "low":
            reasons.append("low asset criticality")
        
        if features.get("historical_false_positive_rate", 0) > 0.5:
            reasons.append("high historical false positive rate")
        
        if features.get("user_risk_score", 1.0) < 0.4:
            reasons.append("low user risk score")
        
        if features.get("source_ip_risk", 1.0) < 0.3:
            reasons.append("low source IP risk")
        
        if features.get("failed_login_count_24h", 0) <= 5:
            reasons.append("few failed login attempts")
    
    # Build explanation string
    if reasons:
        reasons_str = "; ".join(reasons)
        explanation = (
            f"Predicted {label} priority (confidence={confidence:.2f}) "
            f"based on: {reasons_str}."
        )
    else:
        # Fallback generic explanation
        explanation = (
            f"Predicted {label} priority (confidence={confidence:.2f}) "
            "based on learned patterns from the training data."
        )
    
    return explanation
