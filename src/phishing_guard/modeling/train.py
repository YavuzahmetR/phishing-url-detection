import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.phishing_guard.features.url_lexical import extract_url_only_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


class URLOnlyLightGBMModel:
    """
    Sadece ham URL stringinden 14 offline özellik türeten,
    StratifiedGroupKFold cross-validation ve PR-AUC optimizasyonu içeren
    gelişmiş Optuna motorlu LightGBM model sınıfı.
    """

    def __init__(self, random_seed: int = 42, n_jobs: int = 1):
        self.random_seed = random_seed
        self.n_jobs = n_jobs                # Paralel optimizasyon kontrolü
        self.best_params = None
        self.best_value = None              # En iyi CV PR-AUC değeri
        self.model = None
        self.feature_columns = None         # Eğitim sırasında oluşturulan özellik isimleri

    def optimize_and_fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_trials: int = 15,
        use_groups: bool = True
    ):
        """
        Train seti içindeki domain gruplarını kaybetmeden StratifiedGroupKFold uygular,
        Optuna Bayesian arama uzayında en yüksek PR-AUC skorunu veren parametreleri bulur.

        Args:
            X_train: URL ve group sütunlarını içeren DataFrame.
            y_train: Hedef etiketler (1: phishing, 0: legitimate).
            n_trials: Optuna deneme sayısı.
            use_groups: Grup bilgisini kullanıp kullanmayacağı. Eğer `registrable_domain`
                        sütunu yoksa False yapılabilir.
        """
        print(f"🎯 Optuna Bayesian Arama Başlatılıyor ({n_trials} Trial, CV + Grup Korumalı)...")

        print("⏳ Eğitim verisinden 14 yapısal özellik türetiliyor...")
        X_feats = extract_url_only_features(X_train, url_column="URL")
        self.feature_columns = X_feats.columns.tolist()

        # Domain gruplarını al (opsiyonel)
        groups = None
        if use_groups and "registrable_domain" in X_train.columns:
            groups = X_train["registrable_domain"].to_numpy()
            print("   → Domain grupları kullanılıyor (StratifiedGroupKFold).")
        else:
            print("   → Grup bilgisi yok veya kullanılmıyor, StratifiedKFold'a düşülecek.")

        def objective(trial):
            params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "random_state": self.random_seed,
                "n_estimators": 1000,        # Erken durdurma işi devralacak
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 128),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbose": -1
            }

            # Grup bilgisi varsa StratifiedGroupKFold, yoksa normal StratifiedKFold
            if groups is not None:
                cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
                splitter = cv.split(X_feats, y_train, groups=groups)
            else:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
                splitter = cv.split(X_feats, y_train)

            pr_aucs = []
            for train_idx, val_idx in splitter:
                X_tr, X_va = X_feats.iloc[train_idx], X_feats.iloc[val_idx]
                y_tr, y_va = y_train[train_idx], y_train[val_idx]

                clf = lgb.LGBMClassifier(**params)
                # DÜZELTME: eval_X ve eval_y liste olmadan doğrudan verilir
                clf.fit(
                    X_tr, y_tr,
                    eval_X=X_va,
                    eval_y=y_va,
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )

                # PR-AUC'yi sklearn'in average_precision_score ile hesapla
                y_proba = clf.predict_proba(X_va)[:, 1]
                pr_aucs.append(average_precision_score(y_va, y_proba))

            return np.mean(pr_aucs)

        # Optuna çalışması
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        self.best_params = study.best_params
        self.best_value = study.best_value
        print(f"\n🏆 Optuna İle En İyi Parametreler Bulundu: {self.best_params}")
        print(f"🥇 En İyi Validation PR-AUC Skoru: {self.best_value:.4f}")

        # Final modeli tüm eğitim verisiyle eğit
        self.model = lgb.LGBMClassifier(
            **self.best_params,
            random_state=self.random_seed,
            verbose=-1
        )
        self.model.fit(X_feats, y_train)
        return self

    def transform(self, X: pd.DataFrame, url_column: str = "URL") -> pd.DataFrame:
        """
        Ham URL içeren DataFrame'i özellik matrisine dönüştürür.
        Bu metodu kullanarak aynı veri üzerinde birden çok tahmin yapmadan önce
        özellikleri bir kez hesaplayıp saklayabilirsiniz.
        """
        return extract_url_only_features(X, url_column=url_column)

    def predict(
        self,
        X: pd.DataFrame,
        url_column: str = "URL",
        use_precomputed_features: bool = False
    ) -> np.ndarray:
        """
        Tahmin yapar. Eğer X zaten özellik matrisi ise use_precomputed_features=True verin.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi! Önce optimize_and_fit çalıştırın.")
        if use_precomputed_features:
            X_features = X
        else:
            X_features = self.transform(X, url_column=url_column)
        return self.model.predict(X_features)

    def predict_proba(
        self,
        X: pd.DataFrame,
        url_column: str = "URL",
        use_precomputed_features: bool = False
    ) -> np.ndarray:
        """
        Olasılık tahmini yapar. Aynı şekilde özellik matrisi verilebilir.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi! Önce optimize_and_fit çalıştırın.")
        if use_precomputed_features:
            X_features = X
        else:
            X_features = self.transform(X, url_column=url_column)
        return self.model.predict_proba(X_features)