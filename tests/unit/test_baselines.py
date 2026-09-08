import pytest
import pandas as pd
import numpy as np
from src.phishing_guard.modeling.baselines import LexicalHeuristicBaseline
from src.phishing_guard.modeling.baselines import LogRegCharNgramBaseline


def test_lexical_heuristic_baseline_rules():
    """
    RULE: Our LexicalHeuristicBaseline model should:
    1. Classify URLs containing '@' or excessive dots as phishing (1).
    2. Classify clean, non-suspicious sites as legitimate (0).
    """
    model = LexicalHeuristicBaseline()

    mock_df = pd.DataFrame({
        "URL": [
            "http://paypal-update.com",
            "http://fake@bank.com",
            "http://malicious-site.com",
            "https://google.com"
        ]
    })

    predictions = model.predict(mock_df)

    assert predictions[0] == 1
    assert predictions[1] == 1
    assert predictions[2] == 1
    assert predictions[3] == 0


def test_logreg_char_ngram_baseline_fit_and_predict():
    """
    RULE: When our LogRegCharNgramBaseline model is fed with mock data:
    1. It should be able to fit (train) without errors.
    2. It should successfully complete predict and predict_proba steps.
    """
    train_df = pd.DataFrame({
        "URL": [
            "http://paypal-login.com",
            "https://google.com",
            "http://secure-bank-update.net"
        ]
    })
    train_labels = np.array([1, 0, 1])

    model = LogRegCharNgramBaseline(random_seed=42)
    model.fit(train_df, train_labels)

    test_df = pd.DataFrame({
        "URL": ["https://paypal-verify.com"]
    })

    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1

    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(), 1.0)