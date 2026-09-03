import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

def extract_url_only_features(df: pd.DataFrame, url_column: str = "URL") -> pd.DataFrame:
    """
    Dış ağ erişimi (network request) gerektirmeyen, tamamen URL string yapısından
    türetilen ve üretime (production) uygun 14 adet yapısal özelliği çıkarır.
    """
    features_df = pd.DataFrame(index=df.index)
    urls = df[url_column].astype(str)

    # 1. URL Uzunluğu
    features_df["URLLength"] = urls.apply(len)

    # 2. Nokta Sayısı (.)
    features_df["NoOfLettersInURL"] = urls.apply(lambda x: x.count("."))

    # 3. Kısa Çizgi Sayısı (-)
    features_df["NoOfEqualsInURL"] = urls.apply(lambda x: x.count("-"))

    # 4. Soru İşareti Sayısı (?)
    features_df["NoOfQMarkInURL"] = urls.apply(lambda x: x.count("?"))

    # 5. Eşittir Sayısı (=)
    features_df["NoOfOtherSpecialCharsInURL"] = urls.apply(lambda x: x.count("="))

    # 6. Alt Çizgi Sayısı (_)
    features_df["NoOfAmpersandInURL"] = urls.apply(lambda x: x.count("_"))

    # 7. Eğik Çizgi Sayısı (/)
    features_df["NoOfHashInURL"] = urls.apply(lambda x: x.count("/"))

    # 8. Rakam Sayısı
    features_df["NoOfDigitsInURL"] = urls.apply(lambda x: sum(c.isdigit() for c in x))

    # 9. Harf Sayısı
    features_df["NoOfEqualsInURL_letter_count"] = urls.apply(lambda x: sum(c.isalpha() for c in x))

    # 10. HTTPS Kontrolü
    features_df["is_https"] = urls.apply(lambda x: 1 if x.lower().startswith("https") else 0)

    # 11. URL içinde IP adresi var mı? (host kısmı tamamen rakam ve noktalardan oluşuyorsa)
    def _is_ip_host(url: str) -> int:
        try:
            # Eğer URL'de şema yoksa ekle
            if "://" not in url:
                url = "//" + url
            parsed = urlparse(url)
            host = parsed.hostname
            if host is None:
                return 0
            # Host yalnızca rakam ve nokta içeriyorsa ve en az bir nokta varsa IP benzeri kabul et
            return 1 if re.match(r"^[\d.]+$", host) and "." in host else 0
        except Exception:
            return 0

    features_df["is_ip_host"] = urls.apply(_is_ip_host)

    # 12. URL içinde '@' sembolü var mı?
    features_df["has_at_symbol"] = urls.apply(lambda x: 1 if "@" in x else 0)

    # 13. Kabaca bir subdomain derinliği hesabı (URL içindeki nokta sayısına göre)
    features_df["subdomain_depth"] = urls.apply(lambda url: max(0, url.count(".") - 1))

    # 14. Sayısal Karakter Oranı (Rakam / Toplam Uzunluk)
    features_df["digit_ratio"] = features_df["NoOfDigitsInURL"] / (features_df["URLLength"] + 1e-5)

    return features_df