import pytest

# UCI standartlarına göre modelin ürettiği ham tahmin değerleri
MODEL_OUTPUT_LEGITIMATE = 1  # Veri setinde 1 = Legitimate (Meşru)
MODEL_OUTPUT_PHISHING = 0    # Veri setinde 0 = Phishing (Zararlı)

def legacy_api_logic(prediction: int) -> bool:
    """
    Şu an mevcut app/main.py içindeki mantığın aynısı.
    Mevcut kodda prediction == 1 olduğunda 'is_phishing' True dönüyor.
    """

    if prediction == 1:
        return True # Mevcut kodda 1 gelirse Phishing deniyor!
    return False

def test_v1_api_returns_wrong_flag_for_legitimate_site():

    """
    KURAL: Model arkadan 1 (Legitimate) ürettiğinde, API kullanıcıya 
    is_phishing = False (Yani phishing DEĞİL) dönmelidir.
    Mevcut V1 API kodunu test ettiğimiz için bu test PATLAMALIDIR (Fail).
    """

    # Model 1 (Legitimate) üretti. 
    # API'nin False dönmesi gerekirken mevcut kod True dönecek ve test çökecek!

    assert legacy_api_logic(MODEL_OUTPUT_LEGITIMATE) is False


def test_v1_api_returns_wrong_flag_for_phishing_site():
    """
    KURAL: Model arkadan 0 (Phishing) ürettiğinde, API kullanıcıya
    is_phishing = True (Yani phishing) dönmelidir.
    Mevcut V1 API kodunu test ettiğimiz için bu test de PATLAMALIDIR (Fail).
    """
    # Model 0 (Phishing) üretti.
    # API'nin True dönmesi gerekirken mevcut kod False dönecek.

    assert legacy_api_logic(MODEL_OUTPUT_PHISHING) is True