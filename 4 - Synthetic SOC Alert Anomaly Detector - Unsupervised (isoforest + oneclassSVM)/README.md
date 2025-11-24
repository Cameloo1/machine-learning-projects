## Synthetic SOC Alert Anomaly Detector (Unsupervised ML)

Build, train, and explain an unsupervised anomaly detector for Security Operations Center (SOC) alerts without relying on real data. The project fabricates realistic user profiles, produces thousands of benign login/network events, injects multiple attack behaviors, and evaluates IsolationForest and OneClassSVM on the resulting dataset. Everything runs on CPU-only environments and is fully reproducible thanks to explicit random seeds.

---

### Why this project exists
- SOC teams often cannot export production telemetry for prototyping due to privacy or regulatory constraints.
- High-fidelity synthetic alerts let analysts vet tradecraft (exfiltration bursts, impossible travel, brute-force storms) before touching real systems.
- Unsupervised algorithms such as IsolationForest and OneClassSVM learn the shape of normal traffic; explainability layers help analysts understand unusual spikes.

---

### Key capabilities
- **Synthetic user catalog**  
  - Risk scores (10% high-risk), working hours, baseline inbound/outbound bytes, and geolocation.
- **Normal event generator**  
  - Hour sampled around the user’s typical login time, near-zero geo distance, and mostly successful logins.
- **Anomaly injector**  
  - Data exfiltration at late hours with massive bytes out and untrusted devices.  
  - Impossible travel (thousands of kilometers).  
  - Brute-force storms (20–50 failures, zero successes).
- **Modeling & evaluation**  
  - IsolationForest (200 estimators, contamination=1%).  
  - OneClassSVM (RBF kernel, nu=1%).  
  - ROC AUC, precision@k, percentile-based confusion matrices.
- **Explainability**  
  - Global: Pearson correlation between features and anomaly scores.  
  - Local: z-score narratives highlighting which fields deviate from the learned baseline.
- **Visualization outputs** (`plots/`)  
  - Feature correlation bars, score histograms, confusion matrices, model comparison, and deviant feature frequency for top anomalies.

---

### Repository structure
```
synth-soc/
├── soc_anomaly/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generation.py
│   ├── anomaly_detection.py
│   ├── visualization.py
│   └── utils.py
├── scripts/
│   ├── generate_data.py
│   └── run_anomaly_detection.py
├── data/                # created after running generate_data.py
├── plots/               # generated visualizations
├── requirements.txt
└── README.md
```

---

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate    # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

---

### 1. Generate a synthetic SOC dataset
```bash
python scripts/generate_data.py \
  --output-path data/soc_synthetic.csv \
  --n-users 200 \
  --events-per-user 200 \
  --anomaly-fraction 0.01 \
  --random-state 42
```
Flags are optional; defaults already produce ~40k events with ~1% anomalies.

---

### 2. Train, evaluate, and visualize anomaly detectors
```bash
python scripts/run_anomaly_detection.py \
  --data-path data/soc_synthetic.csv \
  --k 50 \
  --threshold-percentile 99 \
  --plots-dir plots
```
What this run does:
1. Splits and scales the dataset using **only normal rows for training**.
2. Fits IsolationForest and OneClassSVM, outputs ROC AUC, precision@k, and confusion matrices.
3. Computes feature correlations plus the top-10 local explanations.
4. Saves seven publication-ready PNGs inside `plots/`.

---

### Example local explanation
```
idx=6535 | score=0.155 | true=ANOMALY
  - geo_distance is +20.59 std from normal
  - bytes_out is +11.92 std from normal
  - device_trust_score is -7.21 std from normal
```
Analysts immediately see the story: impossible travel, massive egress, and an untrusted device at night.

---

### Detailed results (default configuration)
Dataset: 40,000 total events (200 users × 200 events) with 1% injected anomalies.  
Train/test split: 80% of **normal** events for training, remaining normals + all anomalies for testing.

| Metric / Count | IsolationForest | OneClassSVM |
| -------------- | --------------- | ----------- |
| ROC AUC        | 0.994           | **1.000**   |
| Precision@50   | 0.980           | **1.000**   |
| True Positives | 82              | **371**     |
| False Negatives| 318             | **29**      |
| False Positives| 2               | **0**       |
| True Negatives | 7,918           | **7,920**   |

**Interpretation**
- OneClassSVM clearly outperforms IsolationForest on this synthetic SOC scenario. It captures 92.75% of anomalies (371/400) while producing zero false alarms at the 99th percentile threshold.  
- IsolationForest is conservative: it only surfaces 20.5% of anomalies (82/400) despite a solid ROC AUC. Lowering its internal contamination or threshold would be required to approach the SVM’s recall.  
- Global correlation plots highlight that low device trust, failed login bursts, and massive bytes_out swings are the most discriminative features. Local z-score explanations reinforce those signals on a per-alert basis, so SOC analysts can describe **why** each anomaly was raised.  
- The generated PNGs in `plots/` mirror this breakdown, providing ready-to-share visuals for stakeholders.
