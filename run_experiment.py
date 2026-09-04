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
    # Hızlı bitmesi için varsayılan olarak n_jobs=1 kalabilir.
    lgb_model = URLOnlyLightGBMModel(random_seed=42, n_jobs=1)

    # Optuna ile hiperparametre arama + CV
    lgb_model.optimize_and_fit(
        train_df,
        y_train,
        n_trials=30,          # Hızlı sonuç için 10; daha iyi sonuç için 20-30 yapılabilir
        use_groups=True       # Domain bazlı sızıntı koruması
    )

    # En iyi parametreleri ve CV skorunu yazdır
    print("\n🏆 Optuna En İyi Parametreler:")
    for key, value in lgb_model.best_params.items():
        print(f"   {key}: {value}")
    print(f"🥇 Optuna CV PR-AUC Skoru: {lgb_model.best_value:.4f}")

    # Validasyon seti üzerinde tahmin ve değerlendirme
    lgb_preds = lgb_model.predict(val_df)
    lgb_proba = lgb_model.predict_proba(val_df)[:, 1]

    print("\n--- URL-Only LightGBM Optuna & CV Sonuçları (Validation) ---")
    print(classification_report(y_val, lgb_preds, target_names=["Legitimate", "Phishing"]))

    # PR-AUC skorunu da raporla
    val_pr_auc = average_precision_score(y_val, lgb_proba)
    print(f"🔎 Validation PR-AUC Skoru: {val_pr_auc:.4f}")

    # =========================================================================
    # Opsiyonel: Locked Test Seti Değerlendirmesi
    # =========================================================================
    if test_df is not None and not test_df.empty:
        print("\n🔒 Locked Test Seti üzerinde final değerlendirme yapılıyor...")
        y_test = test_df["is_phishing"].to_numpy()
        test_preds = lgb_model.predict(test_df)
        test_proba = lgb_model.predict_proba(test_df)[:, 1]

        print("\n--- URL-Only LightGBM Test Seti Sonuçları ---")
        print(classification_report(y_test, test_preds, target_names=["Legitimate", "Phishing"]))

        test_pr_auc = average_precision_score(y_test, test_proba)
        print(f"🔎 Test PR-AUC Skoru: {test_pr_auc:.4f}")


if __name__ == "__main__":
    main()