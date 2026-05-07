import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data import load_data
from features import prepare_features
from xgboost import XGBClassifier
import joblib
import os





def train_model(x, y):
    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.joblib")


if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    x, y = prepare_features(df)
    train_model(x,y)