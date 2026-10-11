from itertools import product

import numpy as np

from models.ann_feed_forward import build_ann
from preprocessing.feature_engineering import create_lag_features
from preprocessing.normalization import normalize_train_validation
from evaluation.metrics import rmse


def tune_ann(
    df,
    rainfall_lags=range(2, 8),
    streamflow_lags=range(2, 8),
    hidden_sizes=(8, 12, 16, 24, 31),
    learning_rate=0.009,
    epochs=200
):
    results = []
    best = None

    # Total number of ANN configurations
    total = (
        len(rainfall_lags)
        * len(streamflow_lags)
        * len(hidden_sizes)
    )

    count = 0

    for rlag, slag, hidden in product(
        rainfall_lags,
        streamflow_lags,
        hidden_sizes
    ):
        count += 1

        print(
            f"\n[{count}/{total}] "
            f"Rainfall lag={rlag}, "
            f"Streamflow lag={slag}, "
            f"Hidden={hidden}, "
            f"LR={learning_rate}"
        )

        # Create lagged input features
        data = create_lag_features(df, rlag, slag)

        # Chronological train-validation split
        train = data[data["Date"].dt.year <= 2007]
        val = data[data["Date"].dt.year >= 2008]

        # Select rainfall and streamflow lag features
        feature_cols = [
            c for c in data.columns
            if c.startswith("Rainfall_t-")
            or c.startswith("Streamflow_t-")
        ]

        X_train = train[feature_cols].to_numpy()
        X_val = val[feature_cols].to_numpy()

        y_train = train["Target_Streamflow"].to_numpy()
        y_val = val["Target_Streamflow"].to_numpy()

        # Normalize using training data
        (
            X_train_s,
            X_val_s,
            y_train_s,
            y_val_s,
            scales
        ) = normalize_train_validation(
            X_train,
            X_val,
            y_train,
            y_val,
            feature_cols
        )

        # Build ANN
        model = build_ann(
            len(feature_cols),
            hidden,
            learning_rate
        )

        # Train ANN
        model.fit(
            X_train_s,
            y_train_s,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )

        # Predict validation data
        pred_s = model.predict(
            X_val_s,
            verbose=0
        ).ravel()

        # Convert predictions back to original streamflow units
        pred = pred_s * scales["streamflow_max"]

        # Calculate validation RMSE
        score = rmse(y_val, pred)

        print(
            f"    Validation RMSE: {score:.4f}"
        )

        # Store configuration and result
        row = {
            "rainfall_lag": rlag,
            "streamflow_lag": slag,
            "hidden_size": hidden,
            "learning_rate": learning_rate,
            "tuning_epochs": epochs,
            "validation_rmse": score
        }

        results.append(row)

        # Select configuration with lowest validation RMSE
        if best is None or score < best["validation_rmse"]:
            best = row

            print(
                f"    >>> New best RMSE: {score:.4f}"
            )

    return best, results
