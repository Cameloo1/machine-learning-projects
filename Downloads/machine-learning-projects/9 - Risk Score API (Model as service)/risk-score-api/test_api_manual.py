"""Manual API test script."""
import requests
import json
import time

# Wait for server to start
time.sleep(3)

base_url = "http://127.0.0.1:8000"

print("Testing Alert Risk Score API...")
print("=" * 60)

# Test 1: GET /
print("\n1. Testing GET / (index page)")
try:
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
    if response.status_code == 200:
        print("   [OK] Index page loaded successfully")
    else:
        print(f"   [FAIL] Failed with status {response.status_code}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 2: POST /predict
print("\n2. Testing POST /predict")
test_payload = {
    "alert_type": "suspicious_login",
    "source_ip_risk": 0.75,
    "user_risk_score": 0.65,
    "failed_login_count_24h": 12,
    "geo_impossible_travel": 1,
    "asset_criticality": "high",
    "historical_false_positive_rate": 0.15
}
try:
    response = requests.post(f"{base_url}/predict", json=test_payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Prediction successful!")
        print(f"   Label: {data.get('label')}")
        print(f"   Confidence: {data.get('confidence'):.4f}")
        print(f"   Explanation: {data.get('explanation')}")
        print(f"   Probabilities: {json.dumps(data.get('probabilities'), indent=6)}")
    else:
        print(f"   [FAIL] Failed with status {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
        print(f"   [ERROR] Error: {e}")

# Test 3: GET /model-info
print("\n3. Testing GET /model-info")
try:
    response = requests.get(f"{base_url}/model-info")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Model info retrieved!")
        print(f"   Target: {data.get('target')}")
        print(f"   Classes: {data.get('classes')}")
        print(f"   Training samples: {data.get('training_samples')}")
        print(f"   Validation samples: {data.get('validation_samples')}")
    else:
        print(f"   [FAIL] Failed with status {response.status_code}")
except Exception as e:
        print(f"   [ERROR] Error: {e}")

# Test 4: GET /metrics
print("\n4. Testing GET /metrics")
try:
    response = requests.get(f"{base_url}/metrics")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Metrics retrieved!")
        if "classification_report" in data:
            print(f"   Classification report keys: {list(data['classification_report'].keys())[:5]}...")
        if "confusion_matrix" in data:
            print(f"   Confusion matrix shape: {len(data['confusion_matrix'])}x{len(data['confusion_matrix'][0])}")
    else:
        print(f"   [FAIL] Failed with status {response.status_code}")
except Exception as e:
        print(f"   [ERROR] Error: {e}")

# Test 5: POST /predict-form
print("\n5. Testing POST /predict-form (form endpoint)")
form_data = {
    "alert_type": "malware",
    "source_ip_risk": 0.5,
    "user_risk_score": 0.4,
    "failed_login_count_24h": 5,
    "geo_impossible_travel": 0,
    "asset_criticality": "medium",
    "historical_false_positive_rate": 0.3
}
try:
    response = requests.post(f"{base_url}/predict-form", data=form_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   [OK] Form prediction successful!")
        print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
        if "Priority" in response.text or "priority" in response.text.lower():
            print(f"   Response contains prediction result")
    else:
        print(f"   [FAIL] Failed with status {response.status_code}")
except Exception as e:
        print(f"   [ERROR] Error: {e}")

print("\n" + "=" * 60)
print("Testing complete!")
print(f"\nAccess the API at:")
print(f"  - Swagger UI: {base_url}/docs")
print(f"  - Demo UI: {base_url}/")
print(f"  - API Base: {base_url}")

