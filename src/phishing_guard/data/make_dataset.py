import pandas as pd
from src.phishing_guard.data.contract import transform_labels
from src.phishing_guard.data.split import group_based_split


def prepare_frozen_splits(raw_df: pd.DataFrame, random_seed : int=42):
    """
    Applies the end-to-end data contract and splits the data without leakage into Train (70%), Calibration (10%), Validation (10%), and Locked Test (10%).
    """
    # Step 1: Correct labels according to the V2 contract (0->1, 1->0)
    processed_df = transform_labels(raw_df, label_column="label")

    # Step 2: First, set aside 20% of the data for future Validation and Locked Test
    train_calib_df, val_test_df = group_based_split(
        processed_df,
        test_size=0.20,
        random_seed=random_seed
    )

    # Step 3: The remaining 80% will temporarily be Train + Calibration. Separate Calibration (10%) from Train + Calibration (80%). 
    # 12.5% of 80% is exactly 10% of the total data (0.80 * 0.125 = 0.10)
    train_df, calib_df = group_based_split(
        train_calib_df,
        test_size=0.125,
        random_seed=random_seed
    )

    # Step 4: Split the 20% val_test_df in half (10% Validation, 10% Test)
    val_df, test_df = group_based_split(
        val_test_df,
        test_size=0.50,
        random_seed=random_seed
    )

    return train_df, calib_df, val_df, test_df



