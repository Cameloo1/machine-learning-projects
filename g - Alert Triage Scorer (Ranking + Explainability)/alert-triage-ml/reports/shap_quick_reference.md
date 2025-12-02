# SHAP Global Summary Plot: Quick Reference

## What You're Looking At

```
┌─────────────────────────────────────────────────────────┐
│  Feature Name (Y-axis)                                   │
│  ─────────────────────────────────────────────────────    │
│  num_rule_historical_fpr  ●●●●●●●●●●●●●●●●●●●●●●●●●●●    │ ← Most important
│  num_is_known_fp_source   ●●●●●●●●●●●●●●●●●●●●●●●●●●●    │
│  num_dst_asset_criticality ●●●●●●●●●●●●●●●●●●●●●●●●●    │
│  ...                                                      │
│  num_hour_of_day          ●●●●●●●●●●●●●●●●●●●●●●●●●●●    │ ← Less important
│                                                           │
│  ← Negative SHAP (Lowers Priority)  0  Positive SHAP →   │
│     (X-axis: SHAP Value)                                 │
└─────────────────────────────────────────────────────────┘
```

## Reading the Plot: 3 Simple Rules

### 1. **Top to Bottom = Most to Least Important**
- Features at the top have the biggest impact on predictions
- Features at the bottom still matter, just less

### 2. **Left vs Right = Decreases vs Increases Priority**
- **Left side (negative SHAP)** = Feature pushes alert toward **Low priority**
- **Right side (positive SHAP)** = Feature pushes alert toward **High priority**
- **Center (near 0)** = Feature has little impact for that alert

### 3. **Blue vs Red = Low vs High Feature Value**
- **Blue dots** = Low feature value (e.g., low FPR, low criticality)
- **Red dots** = High feature value (e.g., high FPR, high criticality)
- **Color + Position together** = How the feature value affects priority

## Top 2 Features Explained

### `num_rule_historical_fpr` (Rule Historical False Positive Rate)
- **Blue dots (low FPR) → Right** = Low false positive rate → **Higher priority** ✓
- **Red dots (high FPR) → Left** = High false positive rate → **Lower priority** ✓
- **Why it matters:** Trustworthy rules = more reliable alerts

### `num_is_known_fp_source` (Is Known False Positive Source)
- **Blue dots (0 = not FP) → Right** = Not a known FP source → **Higher priority** ✓
- **Red dots (1 = is FP) → Left** = Known FP source → **Lower priority** ✓
- **Why it matters:** Source reputation is critical for triage

## Common Patterns You'll See

### Pattern 1: "High Value = High Priority"
```
Feature: num_dst_asset_criticality
Red dots (high criticality) → Right (positive SHAP)
Interpretation: Critical assets get higher priority alerts
```

### Pattern 2: "Low Value = High Priority"
```
Feature: num_rule_historical_fpr
Blue dots (low FPR) → Right (positive SHAP)
Interpretation: Reliable rules get higher priority alerts
```

### Pattern 3: "Wide Spread = Context-Dependent"
```
Feature: num_geo_distance_km
Dots scattered left and right
Interpretation: Geographic distance matters differently depending on other features
```

## Key Takeaway

**The model uses ALL features, not just the top 2.**

- Top 2 features (`rule_historical_fpr`, `is_known_fp_source`) are the **strongest signals**
- But every feature below them still contributes to predictions
- The spread of dots shows that even "less important" features can be decisive for specific alerts

## When to Worry

### Red Flags:
- ❌ All dots clustered at center (0) = Feature has no impact (remove it)
- ❌ Dots randomly scattered = Feature is noisy (consider feature engineering)
- ❌ Counter-intuitive patterns (e.g., high criticality → lower priority) = Check data quality

### Good Signs:
- ✅ Clear left/right patterns = Feature has consistent impact
- ✅ Blue/red separation = Feature values map to priority changes
- ✅ Intuitive patterns = Model logic aligns with security best practices

## Next Steps

For a detailed feature-by-feature breakdown, see `reports/shap_plot_explanation.md`.

