import pytest
import pandas as pd
from src.phishing_guard.data.make_dataset import prepare_frozen_splits
from src.phishing_guard.data.contract import transform_labels

def test_prepare_frozen_splits_mathematics_and_leakage():
    """
    RULE: When the prepare_frozen_splits function is run:
    1. It must split the data into 4 parts without any loss.
    2. None of these 4 parts (Train, Calibration, Validation, Test) may share
       a single common registrable_domain with another!
    """
    mock_data = pd.DataFrame({
        "URL": [
            "http://a.com", "http://a.com", "http://a.com",
            "http://b.com", "http://b.com",
            "http://c.com", "http://c.com",
            "http://d.com", "http://d.com",
            "http://e.com", "http://f.com",
            "http://g.com", "http://h.com"
        ],
        "label": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # Original UCI labels
    })

    # CORRECTION: label_column = "Label" (capital L) should be used
    train, calib, val, test = prepare_frozen_splits(mock_data, random_seed=42)

    # 1. Does the sum of the parts equal the original data?
    total = len(train) + len(calib) + len(val) + len(test)
    assert total == len(mock_data)

    # 2. Is there any domain intersection?
    def get_domains(df):
        return set(df["registrable_domain"].unique())

    domains_train = get_domains(train)
    domains_calib = get_domains(calib)
    domains_val = get_domains(val)
    domains_test = get_domains(test)

    assert domains_train.isdisjoint(domains_calib)
    assert domains_train.isdisjoint(domains_val)
    assert domains_train.isdisjoint(domains_test)
    assert domains_calib.isdisjoint(domains_val)
    assert domains_calib.isdisjoint(domains_test)
    assert domains_val.isdisjoint(domains_test)
