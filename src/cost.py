from data import load_data
from features import prepare_features
from train_model import objective
from train_model import train_model
from sklearn.metrics import confusion_matrix
import optuna
import numpy as np


def cost_analysis(y_true, y_pred, cost_fn=500, cost_fp=50):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    no_action = (tn + tp) * cost_fn
    print(f"Total cost of misclassified is {total_cost}")
    print(f"Total cost of no action is {no_action}")
    return total_cost, no_action

    return print(f"Total cost of misclassified is {total_cost} and no action cost {no_action}")

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    x, y = prepare_features(df)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(x, y, trial), n_trials=100, timeout=600)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    y_true, y_pred = train_model(x, y, best_params)
    cost = cost_analysis(y_true, y_pred)