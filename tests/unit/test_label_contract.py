import pytest
import pandas as pd

from src.phishing_guard.data.contract import (
    transform_labels,
    SOURCE_LEGITIMATE,
    SOURCE_PHISHING,
    INTERNAL_LEGITIMATE,
    INTERNAL_PHISHING
)

def test_transform_labels_correctly_inverts_semantics():
    """
    KURAL: transform_labels fonksiyonu ham veri setini aldığında,
    etiketleri V2 anayasasına uygun olarak kusursuzca tersine çevirmelidir.
    """

    raw_data = pd.DataFrame({
        "URL" : ["http://phish.com", "http://legit.com"],
        "Label" : [SOURCE_PHISHING, SOURCE_LEGITIMATE] # 0 ve 1
    })
    # Gerçek fonksiyonumuzu çalıştırıyoruz
    result_df = transform_labels(raw_data)
    # 1. Kontrol: Eski 'Label' kolonu silinmiş mi?
    assert "Label" not in result_df.columns
    
    # 2. Kontrol: Yeni 'is_phishing' kolonu eklenmiş mi?
    assert "is_phishing" in result_df.columns
    
    # 3. Kontrol: Orijinal 0 olan Phishing satırı içeride 1 olmuş mu?
    assert result_df.loc[result_df["URL"] == "http://phish.com", "is_phishing"].values[0] == INTERNAL_PHISHING
    
    # 4. Kontrol: Orijinal 1 olan Legitimate satırı içeride 0 olmuş mu?
    assert result_df.loc[result_df["URL"] == "http://legit.com", "is_phishing"].values[0] == INTERNAL_LEGITIMATE