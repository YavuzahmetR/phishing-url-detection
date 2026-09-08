import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class LexicalHeuristicBaseline:

    """
    Our baseline reference model that uses no AI, relying entirely on hand-crafted heuristic rules to analyze URLs and return a phishing risk.
    """
    def __init__(self):
        self.suspicious_keywords = ["login","verify","secure","banking","update","paypal", "malicious"]

    def fit(self, X, y=None):
        """
        As a rule-based model, it does not perform any training (fit). Left as an empty function to comply with scikit-learn API standards..
        """
        return self
    
    def predict_row(self, url: str) -> int:
        """
        Inspects a single URL string and returns 1 (Phishing) or 0 (Legitimate).
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
        Takes a DataFrame of URLs and returns an array of predictions for each.
        """
        predictions = X[url_column].apply(self.predict_row)
        return predictions.to_numpy()


class LogRegCharNgramBaseline:
    """
    Our intelligent baseline model based on Logistic Regression, which splits URLs into character n-grams (3-grams) using TF-IDF and then classifies them.
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
        Trains the model on the training data. It takes only the URL column from X and learns the TF-IDF patterns.
        """
        url_series = X[url_column]
        self.pipeline.fit(url_series, y)
        return self

    def predict(self, X: pd.DataFrame, url_column: str = "URL") -> np.ndarray:
        """
        Generates predictions (1 for Phishing, 0 for Legitimate) for new URLs.
        """
        url_series = X[url_column]
        return self.pipeline.predict(url_series)

    def predict_proba(self, X: pd.DataFrame, url_column: str = "URL") -> np.ndarray:
        """
        Returns the probability of new URLs being phishing. The 1st index of the output is the probability of the positive class (phishing).
        """
        url_series = X[url_column]
        return self.pipeline.predict_proba(url_series)


        



    
