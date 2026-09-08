# Model Card: Phishing URL Guard V2

## Model Details
- **Developed by:** Yavuz Ahmet
- **Model Type:** LightGBM Classifier + Platt Scaling (CalibratedClassifierCV)
- **Version:** v2.0.0
- **Artifact Name:** `url_only_lgb_v2.0.0.joblib`

## Intended Use
- **Primary Use Case:** Real-time enterprise static URL threat validation.
- **Out-of-Scope:** Deep HTML DOM analysis (designed purely for raw URL string metrics to maintain sub-2ms latency).

## Training Data & Metrics
- **Dataset:** PhiUSIIL Phishing URL Dataset (Aggressively pruned to 14 lightweight features).
- **Calibration Method:** Platt Scaling (Sigmoid) evaluated against a 95% Recall target for the Phishing class to minimize False Negatives.
