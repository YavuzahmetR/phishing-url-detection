import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, average_precision_score

from src.phishing_guard.data.make_dataset import prepare_frozen_splits
from src.phishing_guard.modeling.baselines import LexicalHeuristicBaseline, LogRegCharNgramBaseline
from src.phishing_guard.modeling.train import URLOnlyLightGBMModel


def main():
    print("🚀 Gerçek Veriyle Phishing Guard V2 Düellosu Başlıyor...")

    # 1. Gerçek UCI Veri Setini Yüklüyoruz
    data_path = "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv"
    print(f"📦 Veri seti okunuyor: {data_path}")
    raw_df = pd.read_csv(data_path)
    print(f"✅ Ham Veri Boyutu: {raw_df.shape}")

    # 2. Sızıntısız Anayasal Bölmeyi Çalıştırıyoruz (Train, Calib, Val, Test)
    print("⏳ Domain-Grouped Split uygulanıyor (Sızıntı Duvarı Örülüyor)...")
    train_df, calib_df, val_df, test_df = prepare_frozen_splits(raw_df, random_seed=42)

    print(f"   ➔ Train Seti: {train_df.shape}")
    print(f"   ➔ Calibration Seti: {calib_df.shape}")
    print(f"   ➔ Validation Seti: {val_df.shape}")
    print(f"   ➔ Locked Test Seti: {test_df.shape}")

    # Hedef etiketlerimizi numpy dizisi olarak alıyoruz (V2: 1=Phishing, 0=Legitimate)
    y_train = train_df["is_phishing"].to_numpy()
    y_val = val_df["is_phishing"].to_numpy()

    # =========================================================================
    # DÜELLO 1: Heuristic Baseline (Kural Tabanlı)
    # =========================================================================
    print("\n🧠 1. Model: Lexical Heuristic Baseline test ediliyor...")
    heuristic_model = LexicalHeuristicBaseline()
    heuristic_preds = heuristic_model.predict(val_df)

    print("--- Heuristic Baseline Sonuçları ---")
    print(classification_report(y_val, heuristic_preds, target_names=["Legitimate", "Phishing"]))

    # =========================================================================
    # DÜELLO 2: LogReg Char n-gram Baseline (Yapay Zekâ Başlangıcı)
    # =========================================================================
    print("\n🤖 2. Model: LogReg Char N-gram eğitiliyor...")
    logreg_model = LogRegCharNgramBaseline(random_seed=42)
    logreg_model.fit(train_df, y_train)
    logreg_preds = logreg_model.predict(val_df)

    print("--- LogReg Char Ngram Baseline Sonuçları ---")
    print(classification_report(y_val, logreg_preds, target_names=["Legitimate", "Phishing"]))

    # =========================================================================
    # DÜELLO 3: URL-Only LightGBM (Optuna & Cross-Validation Destekli)
    # =========================================================================
    print("\n⚡ 3. Model: URL-Only LightGBM Optuna Motoruyla Eğitiliyor (14 Güvenli Özellik)...")

    # Bilgisayarının işlemci gücüne göre n_jobs değerini artırabilirsin (Örn: n_jobs=4 veya -1)
    lgb_model = URLOnlyLightGBMModel(random_seed=42, n_jobs=-1)

    # Optuna ile hiperparametre arama + CV
    lgb_model.optimize_and_fit(
        train_df,
        y_train,
        n_trials=30,          # Stabilite ve başarı için 30 trial idealdir
        use_groups=True       # Domain bazlı sızıntı koruması
    )

    # En iyi parametreleri ve CV skorunu yazdır
    print("\n🏆 Optuna En İyi Parametreler:")
    for key, value in lgb_model.best_params.items():
        print(f"   {key}: {value}")
    print(f"🥇 Optuna CV PR-AUC Skoru: {lgb_model.best_value:.4f}")

    # Validasyon seti üzerinde tahmin ve değerlendirme (Kalibrasyon Öncesi)
    lgb_preds = lgb_model.predict(val_df)
    lgb_proba = lgb_model.predict_proba(val_df)[:, 1]

    print("\n--- URL-Only LightGBM Optuna & CV Sonuçları (Validation - Kalibrasyon Öncesi) ---")
    print(classification_report(y_val, lgb_preds, target_names=["Legitimate", "Phishing"]))

    val_pr_auc = average_precision_score(y_val, lgb_proba)
    print(f"🔎 Validation PR-AUC Skoru: {val_pr_auc:.4f}")

    # =========================================================================
    # MODEL KALIBRASYONU VE THRESHOLD SEÇİMİ
    # =========================================================================
    print("\n🎯 Kalibrasyon ve threshold seçimi yapılıyor...")
    
    # Calibration ve validation setlerinin özelliklerini çıkar
    X_calib = lgb_model.transform(calib_df)
    y_calib = calib_df["is_phishing"].to_numpy()
    X_val_feats = lgb_model.transform(val_df)
    
    from src.phishing_guard.modeling.calibrate import calibrate_model, select_threshold
    calibrated_lgb = calibrate_model(lgb_model.model, X_calib, y_calib, method='sigmoid')
    
    best_threshold, cal_metrics = select_threshold(calibrated_lgb, X_val_feats, y_val, recall_target=0.95)
    print(f"✅ Seçilen threshold: {best_threshold:.4f}")
    print(f"   Validation'da bu threshold ile: Precision={cal_metrics['precision']:.3f}, Recall={cal_metrics['recall']:.3f}, F1={cal_metrics['f1']:.3f}")

    # =========================================================================
    # Opsiyonel: Locked Test Seti Değerlendirmesi (Kalibre Edilmiş Model ile)
    # =========================================================================
    if test_df is not None and not test_df.empty:
        print("\n🔒 Locked Test Seti üzerinde final değerlendirme yapılıyor...")
        y_test = test_df["is_phishing"].to_numpy()
        
        # Test verisinin özelliklerini çıkarıyoruz
        X_test_feats = lgb_model.transform(test_df)
        
        # Kalibre edilmiş modelden olasılıkları alıp seçilen threshold ile karar veriyoruz
        test_proba = calibrated_lgb.predict_proba(X_test_feats)[:, 1]
        test_preds = (test_proba >= best_threshold).astype(int)

        print("\n--- URL-Only LightGBM Test Seti Sonuçları (Kalibre Edilmiş & Eşik Ayarlı) ---")
        print(classification_report(y_test, test_preds, target_names=["Legitimate", "Phishing"]))

        test_pr_auc = average_precision_score(y_test, test_proba)
        print(f"🔎 Test PR-AUC Skoru: {test_pr_auc:.4f}")

    # =========================================================================
    # MODEL ARTEACT'INI KAYDET
    # =========================================================================
    print("\n💾 Model artefact'ı kaydediliyor...")
    from src.phishing_guard.modeling.artifact import save_model_artifact
    artifact_path = save_model_artifact(
        model=calibrated_lgb,  # kalibre edilmiş model
        feature_columns=lgb_model.feature_columns,
        threshold=best_threshold,
        version="v2.0.0"
    )
    print(f"📁 Model kaydedildi: {artifact_path}")


if __name__ == "__main__":
    main()
