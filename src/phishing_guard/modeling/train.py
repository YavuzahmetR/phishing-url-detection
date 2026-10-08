import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.phishing_guard.features.url_lexical import extract_url_only_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _trial_parameters(trial, random_seed: int) -> dict:
    """Build the original LightGBM search space for one Optuna trial."""
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": random_seed,
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "verbose": -1,
    }


def _mean_cv_average_precision(features, labels, groups, params, random_seed: int):
    """Fit three folds and return their mean average-precision score.

    Keep the existing fit arguments and stopping rule. Moving this loop out
    of the Optuna callback makes fold training readable independently.
    """
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_seed)
        splitter = cv.split(features, labels, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        splitter = cv.split(features, labels)

    fold_scores = []
    for train_idx, val_idx in splitter:
        train_features = features.iloc[train_idx]
        val_features = features.iloc[val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        classifier = lgb.LGBMClassifier(**params)
        classifier.fit(
            train_features,
            train_labels,
            eval_X=val_features,
            eval_y=val_labels,
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        probabilities = classifier.predict_proba(val_features)[:, 1]
        fold_scores.append(average_precision_score(val_labels, probabilities))

    return np.mean(fold_scores)


class URLOnlyLightGBMModel:
    """
    Train LightGBM on 14 URL-only features using Optuna and three-fold CV.
    """

    def __init__(self, random_seed: int = 42, n_jobs: int = 1):
        self.random_seed = random_seed
        self.n_jobs = n_jobs  # Parallel optimization control
        self.best_params = None
        self.best_value = None  # Best CV PR-AUC value
        self.model = None
        self.feature_columns = None  # Feature names created during training

    def optimize_and_fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_trials: int = 15,
        use_groups: bool = True,
    ):
        """
        Search parameters by average precision, then fit on all training rows.

        ``X_train`` contains URLs and optionally ``registrable_domain``.
        ``y_train`` is a NumPy array with 1=phishing and 0=legitimate.
        When grouping is disabled or absent, use ordinary stratified folds.
        """
        print(
            f" Starting Optuna Bayesian Search ({n_trials} Trials, CV + Group Protected)..."
        )

        print(" Deriving 14 structural features from training data...")
        X_feats = extract_url_only_features(X_train, url_column="URL")
        self.feature_columns = X_feats.columns.tolist()

        # Get domain groups (optional)
        groups = None
        if use_groups and "registrable_domain" in X_train.columns:
            groups = X_train["registrable_domain"].to_numpy()
            print(" → Using domain groups (StratifiedGroupKFold).")
        else:
            print(
                " → No group information or not using it, falling back to StratifiedKFold."
            )

        def objective(trial):
            params = _trial_parameters(trial, self.random_seed)
            return _mean_cv_average_precision(
                X_feats, y_train, groups, params, self.random_seed
            )

        # Optuna
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=self.n_jobs)

        self.best_params = study.best_params
        self.best_value = study.best_value
        print(f"\n Best Parameters Found by Optuna: {self.best_params}")
        print(f" Best Validation PR-AUC Score: {self.best_value:.4f}")

        # Preserve the original final-fit defaults (not all CV parameters).
        self.model = lgb.LGBMClassifier(
            **self.best_params, random_state=self.random_seed, verbose=-1
        )
        self.model.fit(X_feats, y_train)
        return self

    def transform(self, X: pd.DataFrame, url_column: str = "URL") -> pd.DataFrame:
        """
        Extract features once for reuse across predictions.
        """
        return extract_url_only_features(X, url_column=url_column)

    def predict(
        self,
        X: pd.DataFrame,
        url_column: str = "URL",
        use_precomputed_features: bool = False,
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
        use_precomputed_features: bool = False,
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
