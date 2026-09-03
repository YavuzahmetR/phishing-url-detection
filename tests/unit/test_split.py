import pytest
import pandas as pd
from src.phishing_guard.data.split import group_based_split


def test_group_based_split_prevents_domain_leakage():
    """
    KURAL: Aynı domaine ait farklı URL'ler asla ve asla hem train 
    hem de test setinde aynı anda bulunamaz! (Sızıntı Sıfır Olmalı)
    """
     # Aynı domaine (badsite.com ve goodsite.com) ait birden fazla URL içeren yapay veri

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

    # Fonksiyonumuzu test boyutunu büyük tutarak çalıştırıyoruz ki grupları bölsün
    train_df, test_df = group_based_split(mock_data, test_size=0.4, random_seed=42)

    # Train ve test setlerindeki benzersiz domainleri küme (set) olarak alıyoruz
    train_domains = set(train_df["registrable_domain"])
    test_domains = set(test_df["registrable_domain"])

     # KESİN KONTROL: İki kümenin kesişimi (ortak elemanı) boş küme olmalıdır!
    intersection = train_domains.intersection(test_domains)

    assert len(intersection) == 0, f"Sızıntı yakalandı! Ortak domainler var: {intersection}"