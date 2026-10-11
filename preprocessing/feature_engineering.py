import numpy as np
import pandas as pd


def create_lag_features(df, rainfall_lag, streamflow_lag):
    data = df.copy()

    for lag in range(1, rainfall_lag + 1):
        data[f"Rainfall_t-{lag}"] = data["Rainfall_mm"].shift(lag)

    for lag in range(1, streamflow_lag + 1):
        data[f"Streamflow_t-{lag}"] = data["Streamflow_cfs"].shift(lag)

    data["Target_Streamflow"] = data["Streamflow_cfs"]

    return data.dropna().reset_index(drop=True)


def create_lstm_features(df, sequence_length):
    """Build oldest-to-newest rainfall/flow sequences to predict Q(t)."""
    rain = df["Rainfall_mm"].to_numpy(dtype=float)
    flow = df["Streamflow_cfs"].to_numpy(dtype=float)
    sequences = []
    targets = []

    for target_idx in range(sequence_length, len(df)):
        start = target_idx - sequence_length
        sequences.append(
            np.column_stack((rain[start:target_idx], flow[start:target_idx]))
        )
        targets.append(flow[target_idx])

    dates = pd.DatetimeIndex(df["Date"].iloc[sequence_length:])
    return np.asarray(sequences), np.asarray(targets), dates
