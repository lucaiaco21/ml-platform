import pandas as pd 
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    shape = df.shape
    missing_values = df.isnull().sum()
    target = df['Exited'].value_counts()
    print(f"Data shape: {shape}, \nMissing Values:{missing_values}, \ntarget distribution: {target} and \ndata description: {df.describe()}")

if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    explore_data(df)
