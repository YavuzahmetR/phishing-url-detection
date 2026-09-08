import pandas as pd

# UCI Original Labels
SOURCE_PHISHING = 0
SOURCE_LEGITIMATE = 1


# V2 Internal System Target Labels (Positive Class = Phishing)
INTERNAL_PHISHING = 1
INTERNAL_LEGITIMATE = 0


def transform_labels(df: pd.DataFrame, label_column: str = "Label"):
    """
    Transforms the original UCI dataset labels to our V2 standards. Source 0 (Phishing) -> Internal 1 (Phishing) Source 1 (Legitimate) -> Internal 0 (Legitimate) This ensures that when our model predicts '1', it truly means 'Phishing'.
    """
    # Create a copy to avoid modifying the original dataset
    processed_df = df.copy()

    if label_column not in processed_df.columns:
        raise KeyError(f"Column '{label_column}' not found in the dataset!")

    # Transformation logic: if source is 0, set internal to 1, otherwise 0
    processed_df["is_phishing"] = processed_df[label_column].apply( lambda x: INTERNAL_PHISHING if x == SOURCE_PHISHING else INTERNAL_LEGITIMATE)

    processed_df = processed_df.drop(columns=[label_column])

    return processed_df