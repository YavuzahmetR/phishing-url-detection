# API contract and verification

Updated: **2026-10-09**. This document describes the existing service, required assets and observed behavior. The checks are not a latency benchmark or a new model evaluation.

## Setup

Install `requirements.txt` in a virtual environment from the repository root. Supply a trusted trained `models/url_only_lgb_v2.0.0.joblib` artifact; the checkout does not include it. For a different location:

```powershell
$env:MODEL_PATH = 'C:\models\url_only_lgb_v2.0.0.joblib'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/docs`. Example request:

```json
{"url": "http://secure-bank-update-verify-checkpoint.com"}
```

Observed response with the artifact used in verification:

```json
{
  "is_phishing": true,
  "probability": 0.9999381527163119,
  "threshold": 0.9514251634442327,
  "model_version": "v2.0.0"
}
```

Artifact SHA-256: `6d5366d05b35bad92809f8fd3008f0b25dd16e18a430c94c1039df519611874c`.
These values are specific to that artifact. The comparison does not establish which historical experiment produced the README metrics.

## Request and response contract

| Request or condition | Behavior |
|---|---|
| `GET /health` | HTTP 200, `{"status":"ok"}` |
| `GET /docs` | Interactive Swagger UI |
| `POST /predict`, string `url` | `is_phishing`, `probability`, `threshold`, `model_version` |
| Missing or numeric `url` | HTTP 422 |
| URL longer than 2,048 characters | HTTP 422 |
| Model object absent | `/predict` HTTP 503, `detail: Model not loaded` |
| Prediction calculation exception | Existing HTTP 500 path |

Probability is `predict_proba(...)[0, 1]`; the decision is `probability >= threshold`. The model extracts features from the URL text and does not visit the page.

## Recorded checks

Twelve bounded HTTP calls had identical status and JSON before/after the package cleanup: health, Swagger UI, five real-artifact predictions, three invalid payloads and two calls with the loaded model object removed. The latter exercises the existing 503 path; it is not a fresh installation check. See [CLEANUP_RESULTS.json](CLEANUP_RESULTS.json).

## Known limitations

| Topic | Observation |
|---|---|
| Artifact access | No trained artifact or published model-download link is included. A clone alone is insufficient for inference. |
| Readiness | Health remains 200/ok without a model; prediction returns 503. Health is not model readiness. |
| Empty or non-URL text | Empty string returned HTTP 200 and a phishing prediction. Validation checks type and length, not URL format or nonempty content. |
| False-positive example | `https://example.com` returned `is_phishing=true`, probability `0.9965848256754277`. One example is not a general false-positive-rate measurement. |
| Model identity | `model_version` is the hardcoded `v2.0.0`, not the artifact's version field. |
| Feature contract | The API does not validate the artifact's `feature_columns`; it uses the preserved extractor ordering. Substituting a differently ordered model is not verified. |
| Environment | Minimum dependencies do not define a historical training lock. The artifact has sklearn 1.9.0 metadata; verification used 1.9.1 and emitted a version warning. |

These behaviors existed before the cleanup. The response schema, model, calibration and prediction calculations were unchanged. Existing calibration and feature-definition limitations are described in the README.
