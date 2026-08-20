import joblib
import shap
import numpy as np
import pandas as pd
from src.config import MODEL_PATH, RANDOM_STATE
from src.data_pipeline import load_and_preprocess_data

def generate_shap_report():
    print("=== SHAP INDEPENDENT REPORTING ENGINE (CORRECTED) ===")

    print(f"[SHAP] Loading serialized pipeline artifact from: {MODEL_PATH}")
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Could not load model. Make sure training is completed: {e}")
        return

    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier_model']

    _, X_test, _, _ = load_and_preprocess_data()

    X_test_transformed = preprocessor.transform(X_test)

    transformed_feature_names = preprocessor.get_feature_names_out()

    clean_feature_names = [name.split('__')[-1] for name in transformed_feature_names]

    # Doğru sırayla DataFrame oluştur
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=clean_feature_names)

    sample_size = min(200, len(X_test_transformed_df))
    X_sample = X_test_transformed_df.sample(sample_size, random_state=RANDOM_STATE)

    print(f"[SHAP] Computing SHAP values on {sample_size} test samples...")

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)


    if isinstance(shap_values, list):
        shap_for_phishing = shap_values[1]
    else:
        if len(shap_values.shape) == 3:
            shap_for_phishing = shap_values[:, :, 1]
        else:
            shap_for_phishing = shap_values

    mean_abs_shap = np.abs(shap_for_phishing).mean(axis=0)
    mean_signed_shap = shap_for_phishing.mean(axis=0)

    shap_df = pd.DataFrame({
        'Feature': clean_feature_names,
        'Impact': mean_abs_shap,
        'Direction': np.where(mean_signed_shap > 0, 'Phishing', 'Legitimate')
    })

    shap_df = shap_df.sort_values('Impact', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("SHAP FEATURE IMPORTANCE (Phishing Class)")
    print("=" * 60)
    print(f"{'Feature':<25} {'Impact':<12} {'Direction':<15}")
    print("-" * 60)
    for _, row in shap_df.iterrows():
        print(f"{row['Feature']:<25} {row['Impact']:<12.6f} {row['Direction']:<15}")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_for_phishing, X_sample, feature_names=clean_feature_names, show=False)
        plt.tight_layout()
        plt.savefig("reports/shap_summary.png", dpi=300, bbox_inches='tight')
        print("\n[SHAP] Summary plot saved to reports/shap_summary.png")
    except Exception as e:
        print(f"\n[SHAP] Could not generate plot: {e}")

    print("\n[SHAP] Report completed successfully.")

if __name__ == "__main__":
    generate_shap_report()