import pandas as pd
import numpy as np
import lightgbm as lgb
from src.phishing_guard.features.url_lexical import extract_url_only_features

class URLOnlyLightGBMModel:
    """
    Sadece ham URL stringinden offline özellikler türeten ve
    LightGBM algoritması ile phishing tahmini yapan ana modelimiz.
    """
    def __init__(self, random_seed : int = 42):
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state= random_seed,
            class_weight= "balanced",
            verbose = -1 # Konsolda gereksiz log birikmesini engeller
        )
    def fit(self, X_train: pd.DataFrame, y_train : np.ndarray, url_column: str = "URL"):
        """
        Ham URL içeren DataFrame'i alır, bizim yazdığımız 14 özelliği türetir
        ve LightGBM modelini bu özellikler üzerinde eğitir.
        """
        X_features = extract_url_only_features(X_train, url_column=url_column)
        self.model.fit(X_features, y_train)
        return self
    def predict(self, X: pd.DataFrame, url_column: str = "URL") -> np.ndarray:
        """
        Ham URL'lerden 14 özelliği çıkarır ve 1 veya 0 tahmini döner.
        """
        X_features = extract_url_only_features(X, url_column=url_column)
        return self.model.predict(X_features)

    def predict_proba(self, X: pd.DataFrame, url_column: str = "URL") -> np.ndarray:
        """
        Ham URL'lerden 14 özellikleri çıkarır ve phishing riski (olasılığı) döner.
        """
        X_features = extract_url_only_features(X, url_column=url_column)
        return self.model.predict_proba(X_features)

