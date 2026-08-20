import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import URLFeaturesInput, PredictionOutput
from src.config import MODEL_PATH, URL_FEATURES

model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print("Production ML model successfully loaded into memory.")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield

app = FastAPI(
    title="Phishing URL Threat Detection API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_pipeline is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict_phishing(input_data: URLFeaturesInput):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")
    
    data_dict = input_data.model_dump()
    synchronized_data = {feature: [data_dict[feature]] for feature in URL_FEATURES}
    input_df = pd.DataFrame(synchronized_data, columns=URL_FEATURES).astype(float)
    
    prediction = int(model_pipeline.predict(input_df)[0])
    probabilities = model_pipeline.predict_proba(input_df)[0]
    
    # 0 -> Legitimate, 1 -> Phishing
    is_phishing = bool(prediction == 1)
    
    confidence = float(probabilities[1] if is_phishing else probabilities[0])
    
    return PredictionOutput(
        is_phishing=is_phishing,
        threat_label="Phishing" if is_phishing else "Legitimate",
        confidence_score=round(confidence, 4)
    )