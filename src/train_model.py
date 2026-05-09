import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import load_data
from features import prepare_features
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os
import optuna
from sklearn.model_selection import StratifiedKFold


def objective(x, y, trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    weight = y.value_counts()[0] / y.value_counts()[1]

    model = XGBClassifier(scale_pos_weight=weight, **params)
  
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, Y_train, cv=kf, scoring='f1', n_jobs=-1)
    return scores.mean()



def train_model(x, y, best_params):
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    weight = y.value_counts()[0] / y.value_counts()[1]
    param = best_params
    model = XGBClassifier(scale_pos_weight=weight, **param)
    cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring="f1")
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print("Cross-validation F1 scores:", cv_scores)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.joblib")


if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    x, y = prepare_features(df)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(x, y, trial), n_trials=100, timeout=600)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    train_model(x, y, best_params)