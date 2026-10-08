"""
Model artifact saving and loading.
"""

from pathlib import Path

import joblib


def save_model_artifact(
    model, feature_columns, threshold, version="v2.0.0", model_dir="models"
):
    """
    Save model, feature columns, and threshold as a single artifact.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "version": version,
        "model_type": "LightGBM + Platt Scaling",
    }
    filepath = model_dir / f"url_only_lgb_{version}.joblib"
    joblib.dump(artifact, filepath)
    print(f"✅ Model artifact saved to {filepath}")
    return filepath


def load_model_artifact(filepath):
    """
    Load model artifact.
    """
    artifact = joblib.load(filepath)
    return artifact["model"], artifact["feature_columns"], artifact["threshold"]
