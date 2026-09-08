from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import os
from src.phishing_guard.features.url_lexical import extract_url_only_features

app = FastAPI(title="Phishing URL Guard V2")

MODEL_PATH = os.getenv("MODEL_PATH", "models/url_only_lgb_v2.0.0.joblib")

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    threshold = artifact['threshold']
    print(f"✅ Model loaded from {MODEL_PATH}, threshold={threshold:.4f}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    threshold = 0.5

class URLInput(BaseModel):
    url: str

    @validator('url')
    def validate_url(cls, v):
        if len(v) > 2048:
            raise ValueError("URL too long")
        return v

class PredictionResponse(BaseModel):
    is_phishing: bool
    probability: float
    threshold: float
    model_version: str = "v2.0.0"

@app.post("/predict", response_model=PredictionResponse)
def predict(input: URLInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame({"URL": [input.url]})
        features = extract_url_only_features(df)
        proba = model.predict_proba(features)[0, 1]
        is_phishing = proba >= threshold
        return PredictionResponse(
            is_phishing=bool(is_phishing),
            probability=float(proba),
            threshold=float(threshold)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
