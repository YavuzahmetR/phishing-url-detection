import hashlib
import json
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, URL_FEATURES, RANDOM_STATE, TEST_SIZE

def run_preflight_check(csv_path: str, manifest_path: str) -> dict:
    """
    UCI Canonical veri setinin bütünlüğünü ve doğruluğunu kontrol eder.
    Herhangi bir sapmada ValueError fırlatır.
    """
    csv_file = pathlib.Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Ham veri dosyası bulunamadı: {csv_path}")
        
    # 1. SHA-256 Hash Hesaplama
    sha256_hash = hashlib.sha256()
    with open(csv_file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    calculated_hash = sha256_hash.hexdigest()
    
    # 2. Manifestoyu Oku
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # 3. Veriyi Yükle ve Şekil (Shape) Kontrolü
    df = pd.read_csv(csv_file)
    actual_rows, actual_cols = df.shape
    
    expected_rows = manifest["canonical_counts"]["total_rows"]
    expected_cols = manifest["canonical_counts"]["total_cols"]
    
    if actual_rows != expected_rows or actual_cols != expected_cols:
        raise ValueError(
            f"Veri boyutu eşleşmiyor! Beklenen: {expected_rows}x{expected_cols}, "
            f"Alınan: {actual_rows}x{actual_cols}"
        )
        
    # 4. Dinamik Kolon Yakalama (Case-Insensitive)
    # Kolon isimlerini küçük harfe çevirip 'label' kelimesini arıyoruz
    target_col = None
    for col in df.columns:
        if col.lower() == "label":
            target_col = col
            break
            
    if target_col is None:
        raise KeyError(f"Veri setinde 'Label' kolonu bulunamadı! Mevcut kolonlar: {list(df.columns[:5])}...")
        
    # Sınıf Dağılımı Kontrolü
    label_counts = df[target_col].value_counts().to_dict()
    exp_legit = manifest["canonical_counts"]["legitimate_source_label_1"]
    exp_phish = manifest["canonical_counts"]["phishing_source_label_0"]
    
    if label_counts.get(1, 0) != exp_legit or label_counts.get(0, 0) != exp_phish:
        raise ValueError(
            f"Sınıf dağılımı hatalı! Beklenen -> Legit(1): {exp_legit}, Phish(0): {exp_phish}. "
            f"Alınan -> Legit(1): {label_counts.get(1, 0)}, Phish(0): {label_counts.get(0, 0)}"
        )
        
    if df.isnull().sum().sum() > 0:
        raise ValueError("Veri setinde eksik (NaN) veri tespit edildi! UCI kurallarına aykırı.")
        
    print("🚀 [PREFLIGHT SUCCESS] Veri bütünlüğü, kolonlar ve dağılım canonical UCI standartlarıyla %100 uyuşuyor!")
    
    return {
        "status": "SUCCESS",
        "sha256": calculated_hash,
        "shape": (actual_rows, actual_cols),
        "label_column_name": target_col
    }

def load_and_preprocess_data():
    """Eski yükleme mantığı (Geçici)"""
    df = pd.read_csv(DATA_PATH)
    X = df[URL_FEATURES] if URL_FEATURES in df.columns else df.iloc[:, :-1]
    
    # Gerçek kolon adını bulup hata almasını engelliyoruz
    target_col = [c for c in df.columns if c.lower() == "label"][0]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    try:
        res = run_preflight_check(
            csv_path="data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
            manifest_path="data/data_manifest.json"
        )
        print(f"Hesaplanan SHA-256 Hash Kodu: {res['sha256']}")
        print(f"Tespit Edilen Orijinal Label Kolon Adı: '{res['label_column_name']}'")
    except Exception as e:
        print(f"❌ Preflight Hatası: {e}")
