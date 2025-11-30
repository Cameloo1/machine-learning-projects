# Model Card – Alert Triage Scorer

## Overview
- **Use case:** Assist SOC analysts in ranking alerts (Low/Medium/High) so they can prioritize investigations.
- **System type:** Gradient boosting classifiers (XGBoost + LightGBM) trained on synthetic SOC telemetry with explainability support.
- **Intended audience:** Security engineers, detection engineers, ML/SOC leads evaluating interpretable triage tooling.

## Data
- **Source:** High-quality synthetic dataset generated via `src.data_generation.generate_synthetic_alerts`.
- **Schema (13 features):**
  1. `alert_type` – categorical descriptor of the triggering analytic.
  2. `src_asset_criticality`, `dst_asset_criticality` – integers 1–5, skewed toward business-as-usual tiers.
  3. `user_risk_score` – beta-distributed risk telemetry scaled to 0–100.
  4. `event_count_24h` – Poisson activity bursts with occasional spikes.
  5. `failed_login_ratio` – proportion of failed logins in the last day.
  6. `geo_distance_km` – mix of local vs. long-distance activity.
  7. `rule_severity` – analyst-defined priority (1–5).
  8. `rule_historical_fpr` – historical false positive rate.
  9. `detection_confidence` – model confidence 0–1.
  10. `is_known_fp_source` – binary flag for noisy sources.
  11. `hour_of_day` – integer 0–23.
  12. `kill_chain_stage` – categorical kill-chain placement.
- **Label:** `priority` ∈ {0: Low, 1: Medium, 2: High}.
- **Hidden scoring logic:** Combines asset criticality, user risk, rule severity/quality, and contextual boosts (geo/time, alert types, kill-chain stage). ≤4% label noise simulates analyst disagreement.
- **Limitations:** Synthetic, rule-derived ground truth; not representative of every SOC’s telemetry. Retraining on real alerts is mandatory before production.

## Modeling
- **Preprocessing:** Shared ColumnTransformer (StandardScaler for numeric, OneHotEncoder for categorical) reused across both models.
- **Models:** 
  - XGBoost (`multi:softprob`, 400 estimators, depth≈4, tuned subsampling/learning rate).
  - LightGBM (`objective=multiclass`, 400 estimators, tuned leaves/depth).
- **Training:** Stratified train/val/test split (70/15/15). RandomizedSearchCV (macro-F1 scoring, 4-fold CV) on train; refit on train+val.

## Performance (test set)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| XGBoost | _see `artifacts/metrics/xgb_metrics.json`_ | _see file_ |
| LightGBM | _see `artifacts/metrics/lgbm_metrics.json`_ | _see file_ |

Confusion matrices saved to `artifacts/plots/confusion_matrix_{model}.png`. Metrics JSON includes per-class precision/recall/F1 and probability summaries.

## Explainability
- **Technique:** SHAP TreeExplainer on the best macro-F1 model (auto-selected).
- **Artifacts:** 
  - Global summary plot: `artifacts/shap/global_summary.png`
  - Local example bar plot: `artifacts/shap/local_example_bar.png`
  - Sample explanations CSV (`artifacts/explanations_sample.csv`) containing predictions, probabilities, and natural-language rationales (e.g., “Priority High because user risk score is elevated, involves a highly critical asset, detection confidence is high.”).
- **Usage:** Explanations highlight top SHAP contributors and translate them into SOC-friendly phrasing for reports or downstream analyst tooling.

## Limitations & Ethical Considerations
- Synthetic data may not reflect emergent attacker behavior or SOC-specific noise patterns.
- Rule-derived labels risk encoding existing biases; analysts must recalibrate on human-reviewed alerts.
- The model is a **decision-support** aid, not an autonomous responder. Humans should remain in the loop.
- SHAP explanations rely on model fidelity; if the model is retrained or drift occurs, explanations must be regenerated.

## Deployment Guidance
- Retrain on proprietary telemetry, re-run the training/evaluation/explain scripts, and refresh the model card metrics.
- Integrate `src.inference.score_csv` or `score_single` within triage services; ensure monitoring for drift and periodic SHAP review.

