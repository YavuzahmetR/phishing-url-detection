import pytest
import pandas as pd
from src.phishing_guard.data.make_dataset import prepare_frozen_splits

def test_prepare_frozen_splits_mathematics_and_leakage():
    """
    KURAL: prepare_frozen_splits fonksiyonu çalıştırıldığında:
    1. Veriyi 4 parçaya eksiksiz bölmelidir.
    2. Bu 4 parçanın (Train, Calibration, Validation, Test) hiçbirinde 
       birbiriyle ortak tek bir registrable_domain bulunmamalıdır!
    """

 # Test için sızıntı yaratabilecek bol tekrarlı domain içeren yapay veri seti
    mock_data = pd.DataFrame({
        "URL": [
            "http://a.com", "http://a.com", "http://a.com",
            "http://b.com", "http://b.com",
            "http://c.com", "http://c.com",
            "http://d.com", "http://d.com",
            "http://e.com", "http://f.com",
            "http://g.com", "http://h.com"
        ],
        "Label": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # Ham UCI etiketleri (0 ve 1)
    })

    # Gerçek akışımızı tetikliyoruz
    train, calib, val, test = prepare_frozen_splits(mock_data, random_seed=42)

    #1. KONTROL: Eski 'Label' kolonu gitmiş, yeni 'is_phishing' anayasası gelmiş mi?
    for df in [train, calib, val, test]:
        assert "Label" not in df.columns
        assert "is_phishing" in df.columns
        assert "registrable_domain" in df.columns

    # 2. KONTROL: Parçaların domain kümelerini çıkarıyoruz
    train_doms = set(train["registrable_domain"])
    calib_doms = set(calib["registrable_domain"])
    val_doms = set(val["registrable_domain"])
    test_doms = set(test["registrable_domain"])

     # KESİN KONTROL: Herhangi iki parça arasında ortak tek bir domain dahi KALMAMALI
    assert train_doms.intersection(calib_doms) == set(), "Train ve Calib arasında sızıntı var!"
    assert train_doms.intersection(val_doms) == set(), "Train ve Val arasında sızıntı var!"
    assert train_doms.intersection(test_doms) == set(), "Train ve Test arasında sızıntı var!"
    assert calib_doms.intersection(val_doms) == set(), "Calib ve Val arasında sızıntı var!"
    assert calib_doms.intersection(test_doms) == set(), "Calib ve Test arasında sızıntı var!"
    assert val_doms.intersection(test_doms) == set(), "Val ve Test arasında sızıntı var!"
