import pandas as pd
from src.phishing_guard.data.contract import transform_labels
from src.phishing_guard.data.split import group_based_split


def prepare_frozen_splits(raw_df: pd.DataFrame, random_seed : int=42):
    """
    Uçtan uca veri anayasasını uygular ve veriyi sızıntısız şekilde
    Train (%70), Calibration (%10), Validation (%10) ve Locked Test (%10) olarak böler.
    """
    # 1. Adım: Etiketleri V2 anayasasına göre düzelt (0->1, 1->0)
    processed_df = transform_labels(raw_df, label_column="label")

    # 2. Adım: Önce verinin %20'sini gelecekteki Validation ve Locked Test için ayırıyoruz
    # Geriye kalan %80 geçici olarak Train + Calibration olacak
    train_calib_df, val_test_df = group_based_split(
        processed_df,
        test_size=0.20,
        random_seed=random_seed
    )

    # 3. Adım: Train + Calibration (%80) içinden Calibration'ı (%10) ayırıyoruz.
    # %80'in %12.5'i toplam verinin tam %10'una denk gelir (0.80 * 0.125 = 0.10)
    train_df, calib_df = group_based_split(
        train_calib_df,
        test_size=0.125,
        random_seed=random_seed
    )

    # 4. Adım: Ayırdığımız %20'lik val_test_df'i yarı yarıya bölüyoruz (%10 Validation, %10 Test)
    val_df, test_df = group_based_split(
        val_test_df,
        test_size=0.50,
        random_seed=random_seed
    )

    return train_df, calib_df, val_df, test_df



