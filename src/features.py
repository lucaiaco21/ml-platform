import pandas as pd
from data import load_data 

def prepare_features(df):
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    x = df.drop("Exited", axis=1)
    y = df["Exited"]
    return x, y

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    x, y = prepare_features(df)
    print(f"Features shape: {x.shape}, Target shape: {y.shape}")
    print(f"Features columns: {x.columns.tolist()}")