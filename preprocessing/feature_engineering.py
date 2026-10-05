def create_lag_features(df, rainfall_lag, streamflow_lag):
    data = df.copy()

    for lag in range(1, rainfall_lag + 1):
        data[f"Rainfall_t-{lag}"] = data["Rainfall_mm"].shift(lag)

    for lag in range(1, streamflow_lag + 1):
        data[f"Streamflow_t-{lag}"] = data["Streamflow_cfs"].shift(lag)

    data["Target_Streamflow"] = data["Streamflow_cfs"]

    return data.dropna().reset_index(drop=True)
