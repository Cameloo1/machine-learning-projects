# Understanding the SHAP Global Summary Plot

## What is This Plot?

The SHAP (SHapley Additive exPlanations) global summary plot visualizes **how each feature influences the model's predictions across all test samples**. It's like a "feature importance map" that shows not just *which* features matter, but *how* they matter (positively or negatively) and *for which types of alerts*.

---

## Plot Components Explained

### 1. **Y-Axis (Vertical): Feature Names**
- Lists all features from **most important** (top) to **least important** (bottom)
- Features are prefixed with:
  - `num_` = numeric features (e.g., `num_rule_historical_fpr`)
  - `cat_` = categorical features (e.g., `cat_alert_type_malware_detected`)

### 2. **X-Axis (Horizontal): SHAP Value**
- **Center line at 0** = neutral impact (feature doesn't change the prediction)
- **Right side (positive SHAP)** = feature **increases** alert priority (pushes toward High)
- **Left side (negative SHAP)** = feature **decreases** alert priority (pushes toward Low)

### 3. **Color Scale (Right Side)**
- **Blue dots** = **low** feature value for that alert
- **Red dots** = **high** feature value for that alert
- Gradient shows the actual feature value for each individual alert

### 4. **Dots (Each Point)**
- Each dot = **one alert** from the test set
- **Position (left/right)** = how much that feature pushed the prediction
- **Color (blue/red)** = the actual feature value for that alert
- **Spread** = how variable the feature's impact is across different alerts

---

## How to Read the Plot: Feature-by-Feature Breakdown

### Top Features (Most Impactful)

#### 1. `num_rule_historical_fpr` (Historical False Positive Rate)
**What it means:** How often this detection rule has been wrong in the past.

**Pattern:**
- **Blue dots (low FPR) → Right side (positive SHAP)** = Low false positive rate **increases** priority
- **Red dots (high FPR) → Left side (negative SHAP)** = High false positive rate **decreases** priority

**Interpretation:** 
- If a rule has historically been accurate (low FPR), the model trusts it more → Higher priority
- If a rule is noisy (high FPR), the model discounts it → Lower priority
- **This is the #1 most important feature** — the model heavily relies on rule reliability

#### 2. `num_is_known_fp_source` (Is Known False Positive Source)
**What it means:** Binary flag (0/1) indicating if the source is known to generate false positives.

**Pattern:**
- **Blue dots (0 = not FP source) → Right side** = Not a known FP source **increases** priority
- **Red dots (1 = is FP source) → Left side** = Known FP source **decreases** priority

**Interpretation:**
- Trusted sources (blue) → Higher priority alerts
- Noisy sources (red) → Lower priority alerts
- **Second most important feature** — source reputation matters a lot

---

### Secondary Features (Still Important)

#### 3. `num_dst_asset_criticality` (Destination Asset Criticality)
**Pattern:** Red dots (high criticality) tend right → Higher criticality increases priority
**Interpretation:** Alerts targeting critical assets (servers, databases) get higher priority

#### 4. `num_rule_severity` (Rule Severity)
**Pattern:** Red dots (high severity) tend right → Higher severity increases priority
**Interpretation:** More severe rules (e.g., "data exfiltration detected") → Higher priority

#### 5. `num_src_asset_criticality` (Source Asset Criticality)
**Pattern:** Red dots (high criticality) tend right → Higher criticality increases priority
**Interpretation:** Alerts from critical assets (e.g., admin workstations) → Higher priority

#### 6. `cat_kill_chain_stage_execution` (Kill Chain: Execution Stage)
**Pattern:** Red dots (stage = execution) tend right → Execution stage increases priority
**Interpretation:** Alerts in the "execution" phase (malware running, commands executed) are more urgent than earlier stages

#### 7. `cat_kill_chain_stage_exfiltration` (Kill Chain: Exfiltration Stage)
**Pattern:** Red dots (stage = exfiltration) tend right → Exfiltration increases priority
**Interpretation:** Data being stolen is a critical late-stage attack → Highest priority

#### 8. `num_detection_confidence` (Detection Confidence)
**Pattern:** Red dots (high confidence) tend right → Higher confidence increases priority
**Interpretation:** When the detection system is confident, the model trusts it more

#### 9. Alert Type Features
- `cat_alert_type_malware_detected` → Red dots right (malware = high priority)
- `cat_alert_type_privilege_escalation` → Red dots right (privilege escalation = high priority)
- `cat_alert_type_data_exfil` → Red dots right (data exfiltration = high priority)
- `cat_alert_type_failed_login_burst` → Mixed pattern (some right, some left)

#### 10. `num_user_risk_score` (User Risk Score)
**Pattern:** Red dots (high risk) tend right → High-risk users increase priority
**Interpretation:** Alerts involving users with elevated risk scores get higher priority

#### 11. `num_geo_distance_km` (Geographic Distance)
**Pattern:** Red dots (large distance) tend right → Unusual location increases priority
**Interpretation:** Logins from far away (potential travel or compromise) → Higher priority

#### 12. `num_failed_login_ratio` (Failed Login Ratio)
**Pattern:** Red dots (high ratio) tend right → Many failed logins increase priority
**Interpretation:** Brute-force attempts or account compromise → Higher priority

---

### Counter-Intuitive Patterns (Why They Make Sense)

#### `cat_alert_type_policy_violation` → Red dots LEFT (decreases priority)
**Why:** Policy violations are often less urgent than active attacks. They might be:
- Compliance issues (not immediate threats)
- Frequently false positives
- Lower severity than malware/exfiltration

#### `cat_alert_type_suspicious_process` → Red dots LEFT (decreases priority)
**Why:** "Suspicious process" is a broad category that often includes:
- Legitimate but unusual software
- False positives from heuristic detection
- Less specific than "malware_detected"

#### `cat_kill_chain_stage_recon` → Red dots LEFT (decreases priority)
**Why:** Reconnaissance is the **earliest** attack stage:
- Less immediately actionable than execution/exfiltration
- Often just information gathering (not data loss yet)
- Model correctly prioritizes later-stage attacks as more urgent

---

## Key Insights from This Plot

### 1. **Two Dominant Features, But Many Contributors**
- `rule_historical_fpr` and `is_known_fp_source` are the **strongest signals**
- But **all other features still contribute** — the model uses the full feature set
- The spread of dots shows that even "less important" features can be decisive for specific alerts

### 2. **Feature Interactions Are Visible**
- Notice how some features have **wide spreads** (dots scattered left and right)
- This means the feature's impact **depends on other features** (interactions)
- Example: `geo_distance_km` might increase priority for some alerts but decrease it for others (depending on user context)

### 3. **Model Logic Aligns with Security Best Practices**
- Low false positive rates → Higher priority ✓
- Known noisy sources → Lower priority ✓
- Critical assets → Higher priority ✓
- Late-stage attacks (exfiltration) → Higher priority ✓
- Early-stage attacks (recon) → Lower priority ✓

### 4. **Class Imbalance Effects**
- Medium priority alerts are most common (471/900 test samples)
- Some features show clustering around Medium predictions (dots near center)
- This explains why the model sometimes "defaults" to Medium when uncertain

---

## How to Use This Plot for Model Improvement

### If High Priority Recall is Too Low:
1. **Check `rule_historical_fpr`** — Are too many high-priority alerts coming from rules with high FPR?
2. **Check `is_known_fp_source`** — Are legitimate high-priority sources being flagged as FP sources?
3. **Check `detection_confidence`** — Are high-priority alerts having their confidence discounted?

### If Low Priority Precision is Too Low:
1. **Check `rule_historical_fpr`** — Are low-FPR rules being over-trusted?
2. **Check alert type features** — Are benign alert types (policy_violation) being misclassified?

### For Feature Engineering:
1. **Wide spreads** suggest feature interactions — consider creating interaction features
2. **Tight clusters** suggest the feature is redundant — consider removing it
3. **Asymmetric patterns** (more dots on one side) suggest class imbalance — consider class weights

---

## Summary

This SHAP plot reveals that the model is **well-calibrated** to security logic:
- It heavily weights rule reliability and source reputation (top 2 features)
- It correctly prioritizes critical assets, severe rules, and late-stage attacks
- It appropriately discounts early-stage attacks and known noisy sources
- **All features contribute**, even if the top 2 dominate

The model is using the full feature set effectively, not just relying on 2 features. The dominance of `rule_historical_fpr` and `is_known_fp_source` reflects that these are the **strongest signals** in the data, which aligns with how SOC analysts actually triage alerts.

