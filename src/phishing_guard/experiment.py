"""The original training, comparison, calibration and artifact workflow."""

import pandas as pd
from sklearn.metrics import average_precision_score, classification_report

from src.phishing_guard.data.make_dataset import prepare_frozen_splits
from src.phishing_guard.modeling.baselines import (
    LexicalHeuristicBaseline,
    LogRegCharNgramBaseline,
)
from src.phishing_guard.modeling.train import URLOnlyLightGBMModel


def load_experiment_splits():
    """Read the existing CSV path and return the four domain-grouped splits."""
    # 1. Loading the Real UCI Dataset
    data_path = "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv"
    print(f" Reading dataset: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f" Raw Data Shape: {raw_df.shape}")

    # 2. Running the Leakage-Free Split (Train, Calib, Val, Test)
    print(" Applying Domain-Grouped Split (Building Leakage Wall)...")
    train_df, calib_df, val_df, test_df = prepare_frozen_splits(raw_df, random_seed=42)

    print(f"   ➔ Train Seti: {train_df.shape}")
    print(f"   ➔ Calibration Seti: {calib_df.shape}")
    print(f"   ➔ Validation Seti: {val_df.shape}")
    print(f"   ➔ Locked Test Seti: {test_df.shape}")

    return train_df, calib_df, val_df, test_df


def evaluate_baselines(train_df, val_df):
    """Print the same heuristic and character n-gram validation reports."""
    # Getting our target labels as a numpy array (V2: 1=Phishing, 0=Legitimate)
    y_train = train_df["is_phishing"].to_numpy()
    y_val = val_df["is_phishing"].to_numpy()

    # =========================================================================
    # Duel 1: Heuristic Baseline (Rule-Based)
    # =========================================================================
    print("\n 1. Model: Testing Lexical Heuristic Baseline...")
    heuristic_model = LexicalHeuristicBaseline()
    heuristic_preds = heuristic_model.predict(val_df)

    print("--- Heuristic Baseline Results ---")
    print(
        classification_report(
            y_val, heuristic_preds, target_names=["Legitimate", "Phishing"]
        )
    )

    # =========================================================================
    # Duel 2: LogReg Char n-gram Baseline (The AI Beginning)
    # =========================================================================
    print("\n 2. Model: Training LogReg Char N-gram...")
    logreg_model = LogRegCharNgramBaseline(random_seed=42)
    logreg_model.fit(train_df, y_train)
    logreg_preds = logreg_model.predict(val_df)

    print("--- LogReg Char Ngram Baseline Results ---")
    print(
        classification_report(
            y_val, logreg_preds, target_names=["Legitimate", "Phishing"]
        )
    )


def train_candidate(train_df, val_df):
    """Tune the candidate and print its pre-calibration validation report."""
    y_train = train_df["is_phishing"].to_numpy()
    y_val = val_df["is_phishing"].to_numpy()

    # =========================================================================
    # Duel 3: URL-Only LightGBM (Optuna & Cross-Validation Powered)
    # =========================================================================
    print(
        "\n 3. Model: Training URL-Only LightGBM with Optuna Engine (14 Safe Features)..."
    )

    # You can increase n_jobs based on your CPU power (e.g., n_jobs=4 or -1)
    lgb_model = URLOnlyLightGBMModel(random_seed=42, n_jobs=-1)

    # Hyperparameter search with Optuna + CV
    lgb_model.optimize_and_fit(
        train_df,
        y_train,
        n_trials=30,  # 30 trials are ideal for stability and success
        use_groups=True,  # Domain-based leakage protection
    )

    print("\n Optuna Best Parameters:")
    for key, value in lgb_model.best_params.items():
        print(f"   {key}: {value}")
    print(f" Optuna CV PR-AUC Score: {lgb_model.best_value:.4f}")

    # Prediction and evaluation on validation set (Before Calibration)
    lgb_preds = lgb_model.predict(val_df)
    lgb_proba = lgb_model.predict_proba(val_df)[:, 1]

    print(
        "\n--- URL-Only LightGBM Optuna & CV Results (Validation - Before Calibration) ---"
    )
    print(
        classification_report(y_val, lgb_preds, target_names=["Legitimate", "Phishing"])
    )

    val_pr_auc = average_precision_score(y_val, lgb_proba)
    print(f" Validation PR-AUC Score: {val_pr_auc:.4f}")

    return lgb_model


def calibrate_candidate(lgb_model, calib_df, val_df):
    """Preserve the existing calibration and validation-threshold procedure."""
    y_val = val_df["is_phishing"].to_numpy()

    # =========================================================================
    # MODEL CALIBRATION AND THRESHOLD SELECTION
    # =========================================================================
    print("\n Performing calibration and threshold selection...")

    # Extract features for calibration and validation sets
    X_calib = lgb_model.transform(calib_df)
    y_calib = calib_df["is_phishing"].to_numpy()
    X_val_feats = lgb_model.transform(val_df)

    from src.phishing_guard.modeling.calibrate import calibrate_model, select_threshold

    calibrated_lgb = calibrate_model(
        lgb_model.model, X_calib, y_calib, method="sigmoid"
    )

    best_threshold, cal_metrics = select_threshold(
        calibrated_lgb, X_val_feats, y_val, recall_target=0.95
    )
    print(f" Selected threshold: {best_threshold:.4f}")
    print(
        f" With this threshold on Validation: Precision={cal_metrics['precision']:.3f}, Recall={cal_metrics['recall']:.3f}, F1={cal_metrics['f1']:.3f}"
    )

    return calibrated_lgb, best_threshold


def evaluate_locked_test(lgb_model, calibrated_lgb, best_threshold, test_df):
    """Print the same calibrated report if the locked test is available."""
    # =========================================================================
    #  Optional: Locked Test Set Evaluation (With Calibrated Model)
    # =========================================================================
    if test_df is not None and not test_df.empty:
        print("\n Performing final evaluation on the Locked Test Set...")
        y_test = test_df["is_phishing"].to_numpy()

        # Extracting features for the test data
        X_test_feats = lgb_model.transform(test_df)

        # Getting probabilities from the calibrated model and deciding with the selected threshold
        test_proba = calibrated_lgb.predict_proba(X_test_feats)[:, 1]
        test_preds = (test_proba >= best_threshold).astype(int)

        print("\n--- URL-Only LightGBM Test Set Results (Calibrated & Thresholded) ---")
        print(
            classification_report(
                y_test, test_preds, target_names=["Legitimate", "Phishing"]
            )
        )

        test_pr_auc = average_precision_score(y_test, test_proba)
        print(f" Test PR-AUC Score: {test_pr_auc:.4f}")


def save_candidate(lgb_model, calibrated_lgb, best_threshold):
    """Write the same artifact keys, version and default destination."""
    # =========================================================================
    # SAVE MODEL ARTIFACT
    # =========================================================================
    print("\n Saving model artifact...")
    from src.phishing_guard.modeling.artifact import save_model_artifact

    artifact_path = save_model_artifact(
        model=calibrated_lgb,  # calibrated model
        feature_columns=lgb_model.feature_columns,
        threshold=best_threshold,
        version="v2.0.0",
    )
    print(f" Model saved: {artifact_path}")


def main():
    """Run the steps in the same order as the original root-level script."""
    print(" Starting Phishing Guard V2 Duel with Real Data...")
    train_df, calib_df, val_df, test_df = load_experiment_splits()
    evaluate_baselines(train_df, val_df)
    lgb_model = train_candidate(train_df, val_df)
    calibrated_lgb, best_threshold = calibrate_candidate(lgb_model, calib_df, val_df)
    evaluate_locked_test(lgb_model, calibrated_lgb, best_threshold, test_df)
    save_candidate(lgb_model, calibrated_lgb, best_threshold)


if __name__ == "__main__":
    main()
