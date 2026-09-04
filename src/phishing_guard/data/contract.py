import pandas as pd

# UCI Orijinal Etiketleri
SOURCE_PHISHING = 0
SOURCE_LEGITIMATE = 1


# V2 İç Sistem Hedef Etiketleri (Pozitif Sınıf = Phishing)
INTERNAL_PHISHING = 1
INTERNAL_LEGITIMATE = 0


def transform_labels(df: pd.DataFrame, label_column: str = "Label"):
    """
    UCI veri setinden gelen orijinal etiketleri V2 standartlarımıza dönüştürür.
    Kaynak 0 (Phishing) -> İç 1 (Phishing)
    Kaynak 1 (Legitimate) -> İç 0 (Legitimate)
    
    Bu işlem sayesinde modelimiz '1' tahmini ürettiğinde bu gerçekten
    'Phishing' anlamına gelecektir.
    """
    # Veri setini bozmamak için bir kopyasını oluşturuyoruz
    processed_df = df.copy()

    if label_column not in processed_df.columns:
        raise KeyError(f"Veri setinde '{label_column}' kolonu bulunamadı!")

    # Dönüşüm mantığı: Kaynak 0 ise içeride 1, değilse 0 yap
    processed_df["is_phishing"] = processed_df[label_column].apply( lambda x: INTERNAL_PHISHING if x == SOURCE_PHISHING else INTERNAL_LEGITIMATE)

    processed_df = processed_df.drop(columns=[label_column])

    return processed_df