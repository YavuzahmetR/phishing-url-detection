import pytest
import pandas as pd
import numpy as np
from src.phishing_guard.modeling.train import URLOnlyLightGBMModel

def test_url_only_lightgbm_pipeline_optimize_and_predict():
    """
    RULE: Our URLOnlyLightGBMModel class should:
    1. Run optimize_and_fit without errors (using a small dataset).
    2. Produce numpy arrays with the correct shapes in the predict and predict_proba steps.
    """
    # Create a mock training dataframe with at least 3 samples per class.
    # This allows splitting methods like StratifiedKFold to see both classes in each fold.
    mock_train_df = pd.DataFrame({
        "URL": [
            "https://paypal-secure-login.com",
            "https://google.com",
            "http://verify-bank-account.net",
            "https://github.com",
            "https://example.com",
            "https://test.com"
        ],
        "registrable_domain": [
            "paypal-secure-login.com",
            "google.com",
            "verify-bank-account.net",
            "github.com",
            "example.com",
            "test.com"
        ]
    })
    # Labels: balanced with 3 phishing (1) and 3 legitimate (0) samples.
    mock_labels = np.array([1, 0, 1, 0, 1, 0])

    model = URLOnlyLightGBMModel(random_seed=42)

    # Run with use_groups=False: in this case, StratifiedKFold is used
    # instead of group-based splitting.
    # n_trials=1 keeps the Optuna optimization minimal for the test.
    model.optimize_and_fit(mock_train_df, mock_labels, n_trials=1, use_groups=False)

    # Predict test
    preds = model.predict(mock_train_df)
    assert len(preds) == len(mock_labels)
    assert set(preds).issubset({0, 1})

    # Predict_proba test
    probas = model.predict_proba(mock_train_df)
    assert probas.shape == (len(mock_labels), 2)
    assert np.all((probas >= 0) & (probas <= 1))
