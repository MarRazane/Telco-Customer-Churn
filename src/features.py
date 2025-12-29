import pandas as pd

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

def add_analytics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
        include_lowest=True
    )

   
    def yes_to_1(x):
        return 1 if str(x).strip().lower() == "yes" else 0

    df["addon_service_count"] = df[ADDON_COLS].applymap(yes_to_1).sum(axis=1)
    return df
