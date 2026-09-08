import pytest
import pandas as pd
import numpy as np
from src.phishing_guard.modeling.train import URLOnlyLightGBMModel

def test_url_only_lightgbm_pipeline_optimize_and_predict():
    """
    KURAL: URLOnlyLightGBMModel sınıfımız:
    1. Hata vermeden optimize_and_fit çalışabilmelidir (küçük veriyle).
    2. predict ve predict_proba adımlarında doğru boyutlarda numpy dizileri üretmelidir.
    """
    mock_train_df = pd.DataFrame({
        "URL": [
            "https://paypal-secure-login.com",
            "https://google.com",
            "http://verify-bank-account.net",
            "https://github.com"
        ],
        "registrable_domain": [
            "paypal-secure-login.com",
            "google.com",
            "verify-bank-account.net",
            "github.com"
        ]
    })
    mock_labels = np.array([1, 0, 1, 0])  # V2: 1=Phishing, 0=Legitimate

    model = URLOnlyLightGBMModel(random_seed=42)

    # optimize_and_fit çalıştır (n_trials=2 hızlı olsun)
    model.optimize_and_fit(mock_train_df, mock_labels, n_trials=2, use_groups=True)

    # Predict test
    preds = model.predict(mock_train_df)
    assert len(preds) == len(mock_labels)
    assert set(preds).issubset({0, 1})

    # Predict_proba test
    probas = model.predict_proba(mock_train_df)
    assert probas.shape == (len(mock_labels), 2)
    assert np.all((probas >= 0) & (probas <= 1))