import pytest
import pandas as pd

from src.phishing_guard.data.contract import (
    transform_labels,
    SOURCE_LEGITIMATE,
    SOURCE_PHISHING,
    INTERNAL_LEGITIMATE,
    INTERNAL_PHISHING
)

def test_transform_labels_correctly_inverts_semantics():
    """
    RULE: When the transform_labels function receives the raw dataset,
    it must flawlessly invert the labels according to the V2 contract.
    """
    raw_data = pd.DataFrame({
        "URL": ["http://phish.com", "http://legit.com"],
        "Label": [SOURCE_PHISHING, SOURCE_LEGITIMATE]  # 0 and 1
    })
    
    # Run our actual function
    result_df = transform_labels(raw_data)
    
    # 1. Check: Is the old 'Label' column removed?
    assert "Label" not in result_df.columns
    
    # 2. Check: Is the new 'is_phishing' column added?
    assert "is_phishing" in result_df.columns
    
    # 3. Check: Did the original 0 (Phishing) become internal 1?
    assert result_df.loc[result_df["URL"] == "http://phish.com", "is_phishing"].values[0] == INTERNAL_PHISHING
    
    # 4. Check: Did the original 1 (Legitimate) become internal 0?
    assert result_df.loc[result_df["URL"] == "http://legit.com", "is_phishing"].values[0] == INTERNAL_LEGITIMATE