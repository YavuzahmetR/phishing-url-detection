import pandas as pd

from src.phishing_guard.data.contract import transform_labels
from src.phishing_guard.data.split import group_based_split


def prepare_frozen_splits(
    raw_df: pd.DataFrame, random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, calibration, validation and test with separate domains.

    The nominal proportions are 70/10/10/10 by groups. Row proportions can
    differ when domains contain different numbers of URLs.
    """
    # Step 1: Correct labels according to the V2 contract (0->1, 1->0)
    processed_df = transform_labels(raw_df, label_column="label")

    # Step 2: First, set aside 20% of the data for future Validation and Locked Test
    train_calib_df, val_test_df = group_based_split(
        processed_df, test_size=0.20, random_seed=random_seed
    )

    # Reserve 12.5% of the remaining groups for calibration (80% * 12.5% = 10%).
    train_df, calib_df = group_based_split(
        train_calib_df, test_size=0.125, random_seed=random_seed
    )

    # Step 4: Split the 20% val_test_df in half (10% Validation, 10% Test)
    val_df, test_df = group_based_split(
        val_test_df, test_size=0.50, random_seed=random_seed
    )

    return train_df, calib_df, val_df, test_df
