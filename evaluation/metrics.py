import numpy as np
from sklearn.metrics import mean_squared_error

PEAK_THRESHOLD = 1500.0

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def split_rmse(y_true, y_pred, threshold=PEAK_THRESHOLD):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    peak = y_true > threshold
    nonpeak = ~peak

    return {
        "Peak_RMSE": rmse(y_true[peak], y_pred[peak]) if peak.any() else np.nan,
        "NonPeak_RMSE": rmse(y_true[nonpeak], y_pred[nonpeak]) if nonpeak.any() else np.nan,
        "Overall_RMSE": rmse(y_true, y_pred)
    }
