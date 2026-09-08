# Model Card: Phishing URL Guard V2

## Model Details
- **Model Type:** LightGBM Classifier + Platt Scaling (CalibratedClassifierCV)
- **Version:** v2.0.0
- **Artifact Name:** `url_only_lgb_v2.0.0.joblib`
- **Release Date:** September 2026

## Intended Use
- **Primary Use Case:** Real-time enterprise static URL threat validation.
- **Out-of-Scope:** Deep HTML DOM analysis (designed purely for raw URL string metrics to maintain sub-2ms latency).

## Training Data & Metrics
- **Dataset:** PhiUSIIL Phishing URL Dataset (Aggressively pruned to 14 lightweight features).
- **Calibration Method:** Platt Scaling (Sigmoid) evaluated against a 95% Recall target for the Phishing class to minimize False Negatives.
- **Final Threshold Selected:** `0.9514`
- **Locked Test Set Performance:**
  - **Overall Accuracy:** 0.98 (98%)
  - **Phishing Precision:** 1.00 (100% exact match on flagged threats)
  - **Phishing Recall:** 0.96 (96% of unseen phishing URLs caught)

## Critical Observations & Production Constraints (Dataset Bias)
During production dry-runs and dynamic endpoint testing, a notable structural bias was identified within the underlying training dataset (PhiUSIIL):

1. **Protocol & Prefix Shortcut Learning:** The training data links `https://www.` prefix tightly with legitimate sites, causing the LightGBM model to over-rely on structural features (such as token counts and protocol presence) rather than purely analyzing semantic threat tokens like `bank`, `secure`, or `update`.
2. **Behavioral Impact:** 
   - Stripping `www.` from a verified safe URL (e.g., inputting `https://google.com` instead of `https://www.google.com`) can artificially trigger a false positive (`is_phishing: true`) due to shifting feature weights.
   - Conversely, a malicious URL prefixed with `https://www.` can artificially suppress the model's inner probability output below the safer thresholds temporarily.
3. **Recommendation for Production:** While the model satisfies structural performance metrics flawlessly on the test set, any external production orchestration layer should enforce a strict **URL Sanitization/Normalization pre-processing filter** (e.g., stripping prefixes down to raw apex domains) prior to running feature extraction to completely mitigate this dataset bias.
