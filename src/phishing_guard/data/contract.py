import pandas as pd

# UCI Original Labels
SOURCE_PHISHING = 0
SOURCE_LEGITIMATE = 1


# V2 Internal System Target Labels (Positive Class = Phishing)
INTERNAL_PHISHING = 1
INTERNAL_LEGITIMATE = 0


def transform_labels(df: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
    """Copy the data and replace the source label with ``is_phishing``.

    Source 0 becomes 1 (phishing); every other value becomes 0. This existing
    behavior is preserved, including its handling of unexpected labels.
    """
    # Create a copy to avoid modifying the original dataset
    processed_df = df.copy()

    if label_column not in processed_df.columns:
        raise KeyError(f"Column '{label_column}' not found in the dataset!")

    # Transformation logic: if source is 0, set internal to 1, otherwise 0
    processed_df["is_phishing"] = processed_df[label_column].apply(
        lambda label: (
            INTERNAL_PHISHING if label == SOURCE_PHISHING else INTERNAL_LEGITIMATE
        )
    )

    processed_df = processed_df.drop(columns=[label_column])

    return processed_df
