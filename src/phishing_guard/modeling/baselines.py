import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class LexicalHeuristicBaseline:

    """
    Hiçbir yapay zekâ içermeyen, tamamen el yapımı kurallarla (heuristic)
    URL analizi yapan ve phishing riski dönen temel referans modelimiz.
    """
    def __init__(self):
        self.suspicious_keywords = ["login","verify","secure","banking","update","paypal", "malicious"]

    def fit(self, X, y=None):
        """
        Kural tabanlı bir model olduğu için herhangi bir eğitim (fit) yapmaz.
        Scikit-learn API standartlarına uyum sağlamak için boş bir fonksiyon olarak bırakılmıştır.
        """
        return self
    
    def predict_row(self, url: str) -> int:
        """
        Tek bir URL stringini inceleyip 1 (Phishing) veya 0 (Legitimate) döner.
        """
        if not isinstance(url, str):
            return 0

        url_lower = url.lower()

        if "@" in url_lower:
            return 1
        if url_lower.count(".") > 3:
            return 1
        if any(keyword in url_lower for keyword in self.suspicious_keywords):
            return 1

        return 0

    def predict(self, X: pd.DataFrame, url_column: str= "URL") -> np.ndarray:
        """
        Bir DataFrame dolusu URL'yi alır ve her biri için tahmin dizisi döner.
        """
        predictions = X[url_column].apply(self.predict_row)
        return predictions.to_numpy()


class LogRegCharNgramBaseline:
    """
    URL stringlerini karakter n-gramlarına (3'lü bloklar) ayıran TF-IDF ve
    ardından sınıflandırma yapan Logistic Regression tabanlı akıllı baseline modelimiz.
    """
    def __init__(self, random_seed: int=42):
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3,3),
            max_features=10000
        )
        self.classifier = LogisticRegression(
            max_iter= 1000,
            random_state=random_seed,
            class_weight="balanced"
        )

        self.pipeline = Pipeline(steps=[
            ("vectorizer", self.vectorizer),
            ("classifier", self.classifier)
        ])
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, url_column: str = "URL"):
        """
        Modeli train verisi üzerinde eğitir.
        X içinden sadece URL kolonunu alıp TF-IDF kalıplarını öğrenir.
        """
        url_series = X[url_column]
        self.pipeline.fit(url_series, y)
        return self

    def predict(self, X: pd.DataFrame, url_column: str = "URL") -> np.ndarray:
        """
        Yeni URL'ler için 1 (Phishing) veya 0 (Legitimate) tahmini üretir.
        """
        url_series = X[url_column]
        return self.pipeline.predict(url_series)

    def predict_proba(self, X: pd.DataFrame, url_column: str = "URL") -> np.ndarray:
        """
        Yeni URL'lerin phishing olma olasılığını (probability) döner.
        Çıktının 1. indeksi positive class (phishing) olasılığıdır.
        """
        url_series = X[url_column]
        return self.pipeline.predict_proba(url_series)


        



    
