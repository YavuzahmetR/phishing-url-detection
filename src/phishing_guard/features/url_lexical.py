import re
from urllib.parse import urlparse

import pandas as pd


def _is_ip_host(url: str) -> int:
    """Preserve the original digits-and-dots host heuristic (not IP validation)."""
    try:
        if "://" not in url:
            url = "//" + url
        host = urlparse(url).hostname
        if host is None:
            return 0
        return 1 if re.match(r"^[\d.]+$", host) and "." in host else 0
    except Exception:
        return 0


def extract_url_only_features(
    df: pd.DataFrame, url_column: str = "URL"
) -> pd.DataFrame:
    """Return the original 14 URL-only columns in their trained order.

    Several historical column names do not describe their actual values.
    They are retained to avoid breaking existing artifacts; see the README
    feature table before interpreting these columns.
    """
    features_df = pd.DataFrame(index=df.index)
    urls = df[url_column].astype(str)

    # 1. URL Length
    features_df["URLLength"] = urls.apply(len)

    # 2. Number of Dots (.)
    features_df["NoOfLettersInURL"] = urls.apply(lambda x: x.count("."))

    # 3. Number of Hyphens (-)
    features_df["NoOfEqualsInURL"] = urls.apply(lambda x: x.count("-"))

    # 4. Number of Question Marks (?)
    features_df["NoOfQMarkInURL"] = urls.apply(lambda x: x.count("?"))

    # 5. Number of Equals Signs (=)
    features_df["NoOfOtherSpecialCharsInURL"] = urls.apply(lambda x: x.count("="))

    # 6. Number of Underscores (_)
    features_df["NoOfAmpersandInURL"] = urls.apply(lambda x: x.count("_"))

    # 7. Number of Slashes (/)
    features_df["NoOfHashInURL"] = urls.apply(lambda x: x.count("/"))

    # 8. Number of Digits
    features_df["NoOfDigitsInURL"] = urls.apply(lambda x: sum(c.isdigit() for c in x))

    # 9. Number of Letters
    features_df["NoOfEqualsInURL_letter_count"] = urls.apply(
        lambda x: sum(c.isalpha() for c in x)
    )

    # 10. HTTPS Check
    features_df["is_https"] = urls.apply(
        lambda x: 1 if x.lower().startswith("https") else 0
    )

    # 11. Digits-and-dots host heuristic; invalid numeric hosts may also match.
    features_df["is_ip_host"] = urls.apply(_is_ip_host)

    # 12. Check if URL contains '@' symbol
    features_df["has_at_symbol"] = urls.apply(lambda x: 1 if "@" in x else 0)

    # 13. Rough subdomain depth calculation (based on number of dots in URL)
    features_df["subdomain_depth"] = urls.apply(lambda url: max(0, url.count(".") - 1))

    # Digit Character Ratio (Digits / Total Length)
    features_df["digit_ratio"] = features_df["NoOfDigitsInURL"] / (
        features_df["URLLength"] + 1e-5
    )

    return features_df
