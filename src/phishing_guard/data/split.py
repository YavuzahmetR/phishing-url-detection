import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit


def extract_registrable_domain(url: str) -> str:
    """
    Return the registrable domain used to group related URLs during splitting.
    """
    if not isinstance(url, str) or url.strip() == "":
        return "unknown_domain"

    ext = tldextract.extract(url)
    # Combine domain and suffix (e.g., example + .com)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return "unknown_domain"


def group_based_split(
    df: pd.DataFrame,
    url_column: str = "URL",
    target_column: str = "is_phishing",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split domain groups, then reset row indexes in both returned frames.

    ``test_size`` is a fraction of groups, not an exact fraction of rows.
    """

    processed_df = df.copy()
    processed_df["registrable_domain"] = processed_df[url_column].apply(
        extract_registrable_domain
    )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )

    train_idx, test_idx = next(
        splitter.split(
            X=processed_df,
            y=processed_df[target_column],
            groups=processed_df["registrable_domain"],
        )
    )

    train_df = processed_df.iloc[train_idx].reset_index(drop=True)
    test_df = processed_df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df
