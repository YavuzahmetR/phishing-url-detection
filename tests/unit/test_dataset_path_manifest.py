import pytest
from src.data_pipeline import run_preflight_check

def test_run_preflight_check_success():
    """Gerçek veri setinin preflight testinden başarıyla geçtiğini doğrular."""
    csv_path = "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv"
    manifest_path = "data/data_manifest.json"
    
    result = run_preflight_check(csv_path, manifest_path)
    
    assert result["status"] == "SUCCESS"
    assert result["shape"] == (235795, 56)
    assert result["label_column_name"] == "label"

def test_run_preflight_check_file_not_found():
    """Hatalı dosya yolunda FileNotFoundError fırlatıldığını doğrular."""
    with pytest.raises(FileNotFoundError):
        run_preflight_check("invalid_path.csv", "data/data_manifest.json")
