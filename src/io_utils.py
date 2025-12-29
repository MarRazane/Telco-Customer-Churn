import pandas as pd

def load_telco_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
