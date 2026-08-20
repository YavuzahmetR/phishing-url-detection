import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

from src.config import URL_FEATURES, RANDOM_STATE, MODEL_PATH
from src.data_pipeline import load_and_preprocess_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Binary Features (scaling won't be applied on these)
BINARY_FEATURES = ['IsDomainIP', 'IsHTTPS', 'HasObfuscation']
CONTINUOUS_FEATURES = [f for f in URL_FEATURES if f not in BINARY_FEATURES]

preprocessor = ColumnTransformer(transformers=[
    ('scale', RobustScaler(), CONTINUOUS_FEATURES),
    ('passthrough', 'passthrough', BINARY_FEATURES)
])

def run_pipeline():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # 1. Base Models Benchmarking
    base_models = {
        "Logistic Regression": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
    }
    
    print("=== BASE MODELS CROSS-VALIDATION ===")
    for name, model in base_models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier_model', model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=['f1'], n_jobs=-1)
        print(f"[{name}] Mean F1: {scores['test_f1'].mean():.4f}")

    # 2. Optuna Optimization
    def objective(trial):
        rf_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'n_jobs': -1,
            'random_state': RANDOM_STATE
        }
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier_model', RandomForestClassifier(**rf_params))])
        return cross_val_score(pipe, X_train, y_train, cv=cv_strategy, scoring='f1', n_jobs=-1).mean()

    print("\n=== OPTUNA HYPERPARAMETER TUNING ===")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    print(f"Best F1: {study.best_value:.4f}")

    # 3. Final Production Model Training
    best_params_prefixed = {f'classifier_model__{k}': v for k, v in study.best_params.items()}
    production_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier_model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    production_pipeline.set_params(**best_params_prefixed)
    
    print("-> Training final model with safe ColumnTransformer structure...")
    production_pipeline.fit(X_train, y_train)

    # 4. Holdout Evaluation
    y_pred = production_pipeline.predict(X_test)
    print("\n=== TEST RESULTS ===")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # 5. Serialization
    joblib.dump(production_pipeline, MODEL_PATH)
    print(f"\nModel saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    run_pipeline()