import pandas as pd

def clean_data(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Rainfall_mm"] = pd.to_numeric(df["Rainfall_mm"], errors="coerce")
    df["Streamflow_cfs"] = pd.to_numeric(df["Streamflow_cfs"], errors="coerce")
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["Date", "Rainfall_mm", "Streamflow_cfs"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df
