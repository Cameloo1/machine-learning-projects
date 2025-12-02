"""Training script converted from notebook."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

print("Setup complete!")
print(f"Base directory: {BASE_DIR}")

# Check if dataset already exists
csv_path = DATA_DIR / "alerts_sample.csv"

if csv_path.exists():
    print(f"Dataset already exists at {csv_path}")
    print("Skipping generation. Delete the file to regenerate.")
else:
    print("Generating synthetic alert dataset...")
    
    # Generate ~6000-7000 rows (random between 5000-8000)
    n_samples = np.random.randint(5000, 8001)
    print(f"Generating {n_samples} samples...")
    
    # Feature definitions
    alert_types = ["brute_force", "malware", "suspicious_login", "data_exfil", "other"]
    asset_criticalities = ["low", "medium", "high"]
    
    # Initialize lists for features
    data = {
        "alert_type": [],
        "source_ip_risk": [],
        "user_risk_score": [],
        "failed_login_count_24h": [],
        "geo_impossible_travel": [],
        "asset_criticality": [],
        "historical_false_positive_rate": [],
        "priority": []
    }
    
    for i in range(n_samples):
        # Generate base features with some correlation to priority
        asset_crit = np.random.choice(asset_criticalities, p=[0.4, 0.4, 0.2])
        alert_type = np.random.choice(alert_types)
        
        # Generate risk scores (correlated with priority)
        base_risk = np.random.beta(2, 5)  # Skewed toward lower values
        source_ip_risk = base_risk + np.random.normal(0, 0.15)
        source_ip_risk = np.clip(source_ip_risk, 0, 1)
        
        user_risk_score = base_risk + np.random.normal(0, 0.15)
        user_risk_score = np.clip(user_risk_score, 0, 1)
        
        # Failed logins (correlated with priority)
        failed_logins = np.random.poisson(2) if np.random.random() > 0.3 else np.random.poisson(10)
        failed_logins = max(0, failed_logins)
        
        # Geo impossible travel (binary, correlated with priority)
        geo_impossible = 1 if np.random.random() < 0.15 else 0
        
        # Historical false positive rate (inverse correlation with priority)
        hist_fp_rate = np.random.beta(3, 2)  # Skewed toward higher values
        
        # Determine priority based on features (realistic rules)
        priority_score = 0.0
        
        # Asset criticality contribution
        if asset_crit == "high":
            priority_score += 0.4
        elif asset_crit == "medium":
            priority_score += 0.2
        
        # Risk scores contribution
        priority_score += (source_ip_risk * 0.2) + (user_risk_score * 0.2)
        
        # Failed logins contribution
        priority_score += min(0.15, failed_logins / 20.0)
        
        # Geo impossible travel contribution
        if geo_impossible == 1:
            priority_score += 0.15
        
        # Historical false positive rate (inverse)
        priority_score += (1 - hist_fp_rate) * 0.1
        
        # Add some noise
        priority_score += np.random.normal(0, 0.1)
        priority_score = np.clip(priority_score, 0, 1)
        
        # Map to priority class
        if priority_score < 0.35:
            priority = "low"
        elif priority_score < 0.65:
            priority = "medium"
        else:
            priority = "high"
        
        # Store features
        data["alert_type"].append(alert_type)
        data["source_ip_risk"].append(round(source_ip_risk, 4))
        data["user_risk_score"].append(round(user_risk_score, 4))
        data["failed_login_count_24h"].append(int(failed_logins))
        data["geo_impossible_travel"].append(int(geo_impossible))
        data["asset_criticality"].append(asset_crit)
        data["historical_false_positive_rate"].append(round(hist_fp_rate, 4))
        data["priority"].append(priority)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nPriority distribution:")
    print(df["priority"].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())

# Load the dataset
df = pd.read_csv(csv_path)
print(f"\nLoaded dataset: {df.shape[0]} samples, {df.shape[1]} features")

# Separate features and target
feature_columns = [
    "alert_type",
    "source_ip_risk",
    "user_risk_score",
    "failed_login_count_24h",
    "geo_impossible_travel",
    "asset_criticality",
    "historical_false_positive_rate"
]

X = df[feature_columns]
y = df["priority"]

print(f"\nFeatures: {list(X.columns)}")
print(f"Target: priority")
print(f"\nTarget distribution:")
print(y.value_counts().sort_index())

# Split into train and validation sets (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"\nTrain target distribution:")
print(y_train.value_counts().sort_index())
print(f"\nValidation target distribution:")
print(y_val.value_counts().sort_index())

# Define feature groups
numeric_features = [
    "source_ip_risk",
    "user_risk_score",
    "failed_login_count_24h",
    "historical_false_positive_rate"
]

categorical_features = [
    "alert_type",
    "geo_impossible_travel",
    "asset_criticality"
]

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Build preprocessing pipeline
# Use sparse=False for compatibility (works in all sklearn versions)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)

# Build full pipeline: preprocessor + model
model = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=100,
    learning_rate=0.1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

print("\nPipeline created:")
print(pipeline)

# Train the model
print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Training complete!")

# Make predictions on validation set
y_val_pred = pipeline.predict(X_val)
y_val_proba = pipeline.predict_proba(X_val)

# Compute metrics
classification_rep = classification_report(
    y_val, y_val_pred,
    output_dict=True,
    target_names=sorted(y.unique())
)

conf_matrix = confusion_matrix(
    y_val, y_val_pred,
    labels=sorted(y.unique())
)

# Print metrics
print("\n" + "="*60)
print("VALIDATION SET METRICS")
print("="*60)
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=sorted(y.unique())))

print("\nConfusion Matrix:")
print(f"Labels: {sorted(y.unique())}")
print(conf_matrix)
print("\n" + "="*60)

# Get the trained classifier to access classes_
classifier = pipeline.named_steps["classifier"]
classes = classifier.classes_.tolist()

# Create model metadata
model_meta = {
    "features": {
        "numeric": numeric_features,
        "categorical": categorical_features
    },
    "target": "priority",
    "classes": classes,
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "description": "HistGradientBoostingClassifier for predicting alert priority (low/medium/high) based on SOC alert features"
}

# Save model
model_path = MODELS_DIR / "model.joblib"
joblib.dump(pipeline, model_path)
print(f"\nModel saved to {model_path}")

# Save model metadata
meta_path = MODELS_DIR / "model_meta.json"
with open(meta_path, "w") as f:
    json.dump(model_meta, f, indent=2)
print(f"Model metadata saved to {meta_path}")

# Create metrics dictionary (convert numpy types to native Python)
metrics = {
    "classification_report": {
        k: {
            k2: float(v2) if isinstance(v2, (np.integer, np.floating)) else v2
            for k2, v2 in v.items()
        } if isinstance(v, dict) else float(v) if isinstance(v, (np.integer, np.floating)) else v
        for k, v in classification_rep.items()
    },
    "confusion_matrix": conf_matrix.tolist()
}

# Save metrics
metrics_path = METRICS_DIR / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

print("\nAll artifacts saved successfully!")

# Display summary
print("\n" + "="*60)
print("MODEL METADATA")
print("="*60)
print(json.dumps(model_meta, indent=2))

print("\n" + "="*60)
print("MAIN METRICS")
print("="*60)

# Display per-class metrics
print("\nPer-class Performance:")
for class_name in sorted(y.unique()):
    if class_name in metrics["classification_report"]:
        metrics_dict = metrics["classification_report"][class_name]
        print(f"\n{class_name.upper()}:")
        print(f"  Precision: {metrics_dict.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics_dict.get('recall', 0):.4f}")
        print(f"  F1-Score:  {metrics_dict.get('f1-score', 0):.4f}")
        print(f"  Support:   {int(metrics_dict.get('support', 0))}")

# Display overall metrics
overall = metrics["classification_report"].get("weighted avg", {})
print(f"\nOverall (Weighted Average):")
print(f"  Precision: {overall.get('precision', 0):.4f}")
print(f"  Recall:    {overall.get('recall', 0):.4f}")
print(f"  F1-Score:  {overall.get('f1-score', 0):.4f}")

print("\n" + "="*60)
print("Training complete! Model ready for deployment.")
print("="*60)

