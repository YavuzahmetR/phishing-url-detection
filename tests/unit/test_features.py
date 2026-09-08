import pytest
import pandas as pd
from src.phishing_guard.features.url_lexical import extract_url_only_features


def test_extract_url_only_features_output_shape_and_types():
    """
    RULE: The extract_url_only_features function should:
    1. Return a numeric matrix with 14 columns and the same number of rows as the input.
    2. Not produce any null (NaN) values.
    """
    mock_df = pd.DataFrame({
        "URL": [
            "https://paypal-update.com",
            "http://192.168.1",
            "https://google.com"
        ]
    })

    features_df = extract_url_only_features(mock_df)

    assert features_df.shape == (3, 14)
    assert features_df.isnull().sum().sum() == 0
    assert features_df.loc[1, "is_ip_host"] == 1
    assert features_df.loc[2, "is_https"] == 1