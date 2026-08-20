import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, URL_FEATURES, RANDOM_STATE, TEST_SIZE

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    
    # 0 = Legitimate, 1 = Phishing
    X = df[URL_FEATURES]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test