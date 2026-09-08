import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

def calibrate_model(model, X_calib, y_calib, method='sigmoid'):
    """
    Calibrate a fitted model using calibration partition.
    Args:
        model: fitted scikit-learn compatible model
        X_calib: calibration features (DataFrame or array)
        y_calib: calibration labels (1=phishing)
        method: 'sigmoid' (Platt) or 'isotonic'
    Returns:
        calibrated_model: CalibratedClassifierCV instance (ensemble=False)
    """

    calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=None, ensemble=False)
    calibrated.fit(X_calib, y_calib)
    return calibrated


def select_threshold(calibrated_model, X_val, y_val, recall_target = 0.95):
    """
    Select threshold that meets a minimum phishing recall on validation set.
    Args:
        calibrated_model: calibrated model
        X_val: validation features
        y_val: validation labels (1=phishing)
        recall_target: minimum recall we want for phishing class
    Returns:
        best_threshold: float
        metrics: dict with precision, recall, f1 at that threshold
    """
    proba = calibrated_model.predict_proba(X_val)[:, 1]  # probability of phishing

    # Precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    best_threshold = 0.5
    best_precision = 0.0
    for t, p, r in zip(thresholds, precisions[:-1], recalls[:-1]):
        if r >= recall_target and p > best_precision:
            best_precision = p
            best_threshold = t
    
    # Compute metrics at threshold
    preds = (proba >= best_threshold).astype(int)
    metrics = {
        'precision': precision_score(y_val, preds),
        'recall': recall_score(y_val, preds),
        'f1': f1_score(y_val, preds)
    }
    return best_threshold, metrics