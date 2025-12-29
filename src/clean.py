import numpy as np
import pandas as pd

def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges clean + convert
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace({"": np.nan, "nan": np.nan}), errors="coerce")

    # Drop missing TotalCharges rows (often tenure==0)
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Target encoding
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    return df
