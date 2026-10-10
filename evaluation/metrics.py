import numpy as np
from sklearn.metrics import mean_squared_error

PEAK_THRESHOLD = 1500.0
MAX_STREAMFLOW = 29700.0


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def percentage_rmse(y_true, y_pred, max_flow=MAX_STREAMFLOW):
    return (rmse(y_true, y_pred) / max_flow) * 100


def split_rmse(y_true, y_pred, threshold=PEAK_THRESHOLD):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    peak = y_true > threshold
    nonpeak = ~peak

    peak_rmse = (
        rmse(y_true[peak], y_pred[peak])
        if peak.any() else np.nan
    )
    nonpeak_rmse = (
        rmse(y_true[nonpeak], y_pred[nonpeak])
        if nonpeak.any() else np.nan
    )
    overall_rmse = rmse(y_true, y_pred)

    return {
        "Peak_RMSE": peak_rmse,
        "Peak_RMSE_Percent": peak_rmse / MAX_STREAMFLOW * 100,
        "NonPeak_RMSE": nonpeak_rmse,
        "NonPeak_RMSE_Percent": nonpeak_rmse / MAX_STREAMFLOW * 100,
        "Overall_RMSE": overall_rmse,
        "Overall_RMSE_Percent": overall_rmse / MAX_STREAMFLOW * 100
    }

