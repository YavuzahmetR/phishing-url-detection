import pytest
import pandas as pd
from src.phishing_guard.features.url_lexical import extract_url_only_features 

import pytest
import pandas as pd
from src.phishing_guard.features.url_lexical import extract_url_only_features

def test_extract_url_only_features_output_shape_and_types():
    """
    KURAL: extract_url_only_features fonksiyonu:
    1. Girdi DataFrame boyutuyla uyumlu, 14 kolonlu sayısal bir matris dönmelidir.
    2. Boş (NaN) değer üretmemelidir.
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

