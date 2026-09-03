import pytest
import pandas as pd
import numpy as np
from src.phishing_guard.modeling.train import URLOnlyLightGBMModel

def test_url_only_lightgbm_pipeline_fit_and_predict():
    """
    KURAL: URLOnlyLightGBMModel sınıfımız:
    1. Hata vermeden fit (eğitim) olabilmelidir.
    2. predict ve predict_proba adımlarında doğru boyutlarda numpy dizileri üretmelidir.
    """
    mock_train_df = pd.DataFrame({
        "URL": [
            "https://paypal-secure-login.com",
            "https://google.com",
            "http://verify-bank-account.net",
            "https://github.com"
        ]
    })
    mock_labels = np.array([1, 0, 1, 0])  # V2 anayasası: 1=Phishing, 0=Legitimate
    
    model = URLOnlyLightGBMModel(random_seed=42)
    model.fit(mock_train_df, mock_labels)
    
    mock_test_df = pd.DataFrame({
        "URL": ["https://suspicious-link-update.org"]
    })
    
    predictions = model.predict(mock_test_df)
    probabilities = model.predict_proba(mock_test_df)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
    
    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)