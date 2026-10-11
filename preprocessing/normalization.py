import numpy as np


def normalize_train_validation(X_train, X_val, y_train, y_val, feature_cols):
    """Divide each variable by its maximum in the training period.

    Rainfall lags share one scale and streamflow lags/targets share another.
    Using training-period maxima follows the paper's scaling method without
    assuming that its unavailable source dataset has the same maxima.
    """
    rainfall_idx = np.array([
        i for i, name in enumerate(feature_cols)
        if name.startswith("Rainfall")
    ])
    flow_idx = np.array([
        i for i, name in enumerate(feature_cols)
        if name.startswith("Streamflow")
    ])
    if rainfall_idx.size == 0 or flow_idx.size == 0:
        raise ValueError("Expected rainfall and streamflow lag features.")

    rainfall_max = float(np.max(X_train[..., rainfall_idx]))
    streamflow_max = float(max(np.max(X_train[..., flow_idx]), np.max(y_train)))
    if rainfall_max <= 0 or streamflow_max <= 0:
        raise ValueError("Training maxima must be positive for scaling.")

    X_train_s = X_train.astype(float, copy=True)
    X_val_s = X_val.astype(float, copy=True)
    X_train_s[..., rainfall_idx] /= rainfall_max
    X_val_s[..., rainfall_idx] /= rainfall_max
    X_train_s[..., flow_idx] /= streamflow_max
    X_val_s[..., flow_idx] /= streamflow_max
    y_train_s = y_train.astype(float) / streamflow_max
    y_val_s = y_val.astype(float) / streamflow_max
    return X_train_s, X_val_s, y_train_s, y_val_s, {
        "rainfall_max": rainfall_max,
        "streamflow_max": streamflow_max,
    }
