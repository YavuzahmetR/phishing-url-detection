import pytest
import pandas as pd
from src.phishing_guard.data.split import group_based_split


def test_group_based_split_prevents_domain_leakage():
    """
    RULE: Different URLs belonging to the same domain must NEVER appear
    in both the train and test sets at the same time! (Leakage must be Zero)
    """
    # Synthetic data containing multiple URLs from the same domain
    # (badsite.com and goodsite.com)

    mock_data = pd.DataFrame({
        "URL": [
            "http://badsite.com",
            "http://badsite.com",
            "http://badsite.com",
            "https://goodsite.com",
            "https://goodsite.com"
        ],
        "is_phishing": [1, 1, 1, 0, 0]
    })

    # Run our function with a large enough test size so it actually splits the groups
    train_df, test_df = group_based_split(mock_data, test_size=0.4, random_seed=42)

    # Take the unique domains in the train and test sets as sets
    train_domains = set(train_df["registrable_domain"])
    test_domains = set(test_df["registrable_domain"])

    # STRICT CHECK: The intersection of the two sets must be the empty set!
    intersection = train_domains.intersection(test_domains)

    assert len(intersection) == 0, f"Leakage detected! Shared domains found: {intersection}"
