import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

def extract_registrable_domain(url: str) -> str:
    """
    Bir URL'nin içinden sızıntıyı önleyecek ana kök domaini (registrable domain) çıkarır.
    Örnek: '://example.com' -> 'example.com'
    """
    if not isinstance(url , str) or url.strip() == "":
        return "unknown_domain"

    ext = tldextract.extract(url)
    # domain ve suffix'i birleştiriyoruz (örn: example + .com)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return "unknown_domain"


def group_based_split(df: pd.DataFrame, url_column: str = "URL", target_column: str = "is_phishing", test_size: float= 0.2, random_seed: int = 42):
     """
    Aynı domaine sahip URL'lerin train ve test setlerine sızmasını engelleyen
    Grup Tabanlı Bölümleme (Grouped Train-Test Split) yapar.
    """

     processed_df = df.copy()
      # 1. Her satır için registrable_domain sütununu türetiyoruz
     processed_df["registrable_domain"] = processed_df[url_column].apply(extract_registrable_domain)

      # 2. Scikit-learn'ün grupları birbirinden tamamen ayıran split nesnesini kuruyoruz
     gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)

     # 3. Bölme işlemini domain gruplarına göre tetikliyoruz

     train_idx, test_idx = next(gss.split(X= processed_df, y=processed_df[target_column],  groups=processed_df["registrable_domain"]))

     train_df = processed_df.iloc[train_idx].reset_index(drop=True)
     test_df = processed_df.iloc[test_idx].reset_index(drop=True)

     return train_df, test_df

