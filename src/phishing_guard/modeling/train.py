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
    Advanced Optuna-powered LightGBM model class that derives 14 offline features from the raw URL string, incorporating StratifiedGroupKFold cross-validation and PR-AUC optimization.
    """

    def __init__(self, random_seed: int = 42, n_jobs: int = 1):
        self.random_seed = random_seed
        self.n_jobs = n_jobs                # Parallel optimization control
        self.best_params = None
        self.best_value = None              # Best CV PR-AUC value
        self.model = None
        self.feature_columns = None         # Feature names created during training

    def optimize_and_fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_trials: int = 15,
        use_groups: bool = True
    ):
        """
        Applies StratifiedGroupKFold without losing domain groups in the training set, and finds the parameters that yield the highest PR-AUC score in the Optuna Bayesian search space. Args: X_train: DataFrame containing URL and group columns. y_train: Target labels (1: phishing, 0: legitimate). n_trials: Number of Optuna trials. use_groups: Whether to use group information. Can be set to False if theregistrable_domaincolumn is not present.
        """
        print(f" Starting Optuna Bayesian Search ({n_trials} Trials, CV + Group Protected)...")

        print(" Deriving 14 structural features from training data...")
        X_feats = extract_url_only_features(X_train, url_column="URL")
        self.feature_columns = X_feats.columns.tolist()

        # Get domain groups (optional)
        groups = None
        if use_groups and "registrable_domain" in X_train.columns:
            groups = X_train["registrable_domain"].to_numpy()
            print(" → Using domain groups (StratifiedGroupKFold).")
        else:
            print(" → No group information or not using it, falling back to StratifiedKFold.")

        def objective(trial):
            params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "random_state": self.random_seed,
                "n_estimators": 1000,        # Early stopping will handle this
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

            # Use StratifiedGroupKFold if groups exist, otherwise regular StratifiedKFold
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
                # FIX: eval_X and eval_y are passed directly, not as a list
                clf.fit(
                    X_tr, y_tr,
                    eval_X=X_va,
                    eval_y=y_va,
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )

                # Calculate PR-AUC using sklearn's average_precision_score
                y_proba = clf.predict_proba(X_va)[:, 1]
                pr_aucs.append(average_precision_score(y_va, y_proba))

            return np.mean(pr_aucs)

        # Optuna 
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        self.best_params = study.best_params
        self.best_value = study.best_value
        print(f"\n Best Parameters Found by Optuna: {self.best_params}")
        print(f" Best Validation PR-AUC Score: {self.best_value:.4f}")

        # # Train the final model on all training data
        self.model = lgb.LGBMClassifier(
            **self.best_params,
            random_state=self.random_seed,
            verbose=-1
        )
        self.model.fit(X_feats, y_train)
        return self

    def transform(self, X: pd.DataFrame, url_column: str = "URL") -> pd.DataFrame:
        """
        Transforms a DataFrame containing raw URLs into a feature matrix. Using this method, you can compute and cache the features once before making multiple predictions on the same data.
        """
        return extract_url_only_features(X, url_column=url_column)

    def predict(
        self,
        X: pd.DataFrame,
        url_column: str = "URL",
        use_precomputed_features: bool = False
    ) -> np.ndarray:
        """
        Makes predictions. Set use_precomputed_features=True if X is already a feature matrix.
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Run optimize_and_fit first.")
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
        Makes probability predictions. A feature matrix can be provided in the same way. 
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Run optimize_and_fit first.")
        if use_precomputed_features:
            X_features = X
        else:
            X_features = self.transform(X, url_column=url_column)
        return self.model.predict_proba(X_features)