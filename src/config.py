import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "optimized_phishing_detector.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.2


URL_FEATURES = [
    'DomainLength', 'IsDomainIP', 'TLDLength', 'NoOfSubDomain', 'IsHTTPS',
    'CharContinuationRate', 'HasObfuscation', 'ObfuscationRatio', 
    'LetterRatioInURL', 'DegitRatioInURL', 'SpacialCharRatioInURL', 
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL'
]