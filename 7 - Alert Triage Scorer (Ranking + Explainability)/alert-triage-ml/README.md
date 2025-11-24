# Alert Triage Scorer – Gradient Boosting + SHAP

Production-quality mini-project demonstrating how gradient boosting and SHAP explainability can accelerate SOC alert triage workflows.

## Why this project matters
- **SOC decision support** – triages synthetic security alerts with 13 curated features and outputs Low/Medium/High priority labels.
- **Interpretable ML** – compares XGBoost and LightGBM pipelines, retaining SHAP explainability for both global and local insight.
- **Reproducible engineering** – clean architecture with reusable modules, CLI entry points, persisted artifacts, and inference helpers.

## Project structure
```
alert-triage-ml/
├── data/
│   ├── raw/alerts_synthetic.csv
│   └── processed/{train,val,test}.csv
├── src/                # reusable Python package
├── models/             # serialized pipelines (.pkl)
├── artifacts/
│   ├── metrics/        # JSON metrics per model
│   ├── plots/          # class dist + confusion matrices
│   └── shap/           # SHAP summary + local plots
├── notebooks/          # 01_eda.ipynb & 02_modeling_explainability.ipynb
└── reports/
    ├── model_card.md
    ├── production_readiness_analysis.md
    ├── shap_plot_explanation.md
    └── shap_quick_reference.md
```

## Tech stack
Python 3.10+, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, matplotlib, seaborn, joblib.

## End-to-end workflow
```bash
pip install -r requirements.txt

# 1. Generate synthetic data that respects the 13-feature schema.
python -m src.data_generation --n_samples 6000

# 2. Train gradient boosting pipelines with stratified CV.
python -m src.train

# 3. Evaluate on the held-out test set and persist metrics/plots.
python -m src.evaluate

# 4. Produce SHAP global + local explanations & text rationales.
python -m src.explain

# 5. Run inference utilities (batch CSV scoring or programmatic single).
python -m src.inference --mode csv --input data/raw/alerts_synthetic.csv --output artifacts/scored_alerts.csv --model_path models/xgb_pipeline.pkl
```

### Continuous Integration
- `.github/workflows/ci.yml` runs on pushes/PRs: data generation (`--n_samples 1500`), fast CV training (`python -m src.train --search-iterations 5 --cv-folds 3`), evaluation, SHAP artifacts, and an inference smoke test.
- Use the same flags locally for quick sanity checks before full training runs.

## Key features
- **Synthetic data procurement** – `src.data_generation` builds realistic alert distributions plus hidden-label logic and optional noise.
- **Validation & preprocessing** – `src.data_loading` enforces schema, while `src.preprocessing` ensures consistent scaling/encoding.
- **Model comparison** – `src.train` reuses the same ColumnTransformer for XGBoost and LightGBM pipelines, hyper-tuned with macro-F1.
- **Metrics & plots** – `src.evaluate` stores JSON metrics and confusion matrices; class distribution and histogram plots live in `artifacts/plots`.
- **Explainability** – `src.explain` selects the best model via metrics, generates SHAP summary/local plots, and exports natural-language rationales.
- **Inference APIs** – `src.inference` supports csv batch scoring and single-alert scoring with optional explanations for downstream tooling.

## Results snapshot
- **XGBoost** tops the leaderboard with accuracy **0.90** and macro-F1 **0.88**, delivering balanced per-class F1 scores (Low 0.82, Medium 0.92, High 0.91). Its confusion matrix (see `artifacts/plots/confusion_matrix_xgb.png`) shows only ~6% of High alerts downgraded to Medium.
- **LightGBM** remains competitive at accuracy **0.87**, macro-F1 **0.85**. It predicts Medium alerts well but confuses a few more Low/High cases with Medium (`artifacts/plots/confusion_matrix_lgbm.png`), matching the slightly lower recall reported in `artifacts/metrics/lgbm_metrics.json`.
- **SHAP insights** (`artifacts/shap/global_summary.png`) confirm that trustworthy rules drive decisions: low `rule_historical_fpr` and the absence of `is_known_fp_source` push alerts toward High priority, while asset criticality, rule severity, kill-chain phase, detection confidence, and alert type one-hots provide the next tier of influence. The local bar plot highlights that individual alerts still draw on a mix of these features, even if the top two stand out for many cases.
  - 📖 **Quick guide:** [`reports/shap_quick_reference.md`](reports/shap_quick_reference.md) — 3 simple rules for reading SHAP plots
  - 📚 **Detailed breakdown:** [`reports/shap_plot_explanation.md`](reports/shap_plot_explanation.md) — Feature-by-feature interpretation and model logic analysis

## Reproducibility + configs
- Global seeds live in `src/config.py`.
- Paths rely on `pathlib.Path`, no hard-coded strings.
- All scripts expose CLI entry points guarded by `if __name__ == "__main__":`.

## Production readiness

**Current Status:** XGBoost achieves **90% accuracy** and **0.88 macro-F1**, with **88% High priority recall**. For SOC workflows, missing 12% of critical alerts may require tuning before production deployment.

**Key Gap:** High priority recall (88%) — 35 High alerts misclassified (33 as Medium, 2 as Low) on test set.

**Quick Wins to Improve:**
1. **Add class weights** to penalize High priority misclassifications (+3-5% High recall expected)
2. **Lower High priority threshold** (predict High if `prob_high > 0.25` instead of argmax) (+2-3% High recall)
3. **Increase hyperparameter search iterations** from 15 to 50+ (+1-2% overall)

**Expected Outcome:** With Phase 1 optimizations, High priority recall → **92-93%** (from 88%).

📋 **Full analysis:** [`reports/production_readiness_analysis.md`](reports/production_readiness_analysis.md) — Detailed tuning roadmap, implementation code examples, and production deployment checklist.

## Documentation

### Core Documentation
- 📄 **[Model Card](reports/model_card.md)** — Human-readable summary covering problem context, data schema, modeling choices, performance, explainability approach, and limitations.

### Performance & Production
- 📊 **[Production Readiness Analysis](reports/production_readiness_analysis.md)** — Comprehensive assessment of model production readiness, tuning recommendations (3-phase roadmap), implementation examples, and deployment checklist.

### Explainability & SHAP
- 🎯 **[SHAP Quick Reference](reports/shap_quick_reference.md)** — 3 simple rules for reading SHAP plots, common patterns, and visual guide.
- 📚 **[SHAP Plot Explanation](reports/shap_plot_explanation.md)** — Detailed feature-by-feature breakdown, interpretation guide, and model logic analysis.

