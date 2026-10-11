import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import numpy as np
import pandas as pd

from preprocessing.cleaning import clean_data
from preprocessing.feature_engineering import create_lag_features, create_lstm_features
from preprocessing.normalization import normalize_train_validation
from models.ann_feed_forward import build_ann
from models.lstm import build_lstm
from models.linear_regression import build_linear_regression
from models.random_forest import build_random_forest
from tuning.ann_tuning import tune_ann
from evaluation.metrics import split_rmse
from evaluation.plots import save_plots

DATA = "data/leaf_river_2003_2012.csv"

def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            "Dataset not found. Run: python download_dataset.py"
        )

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    df = pd.read_csv(DATA)
    df = clean_data(df)

    print("Dataset rows:", len(df))
    print("Training period: 2003-2007")
    print("Validation period: 2008-2012")

    print("\nTuning ANN...")
    best, tuning_results = tune_ann(df)

    pd.DataFrame(tuning_results).sort_values(
        "validation_rmse"
    ).to_csv("outputs/results/ann_tuning_results.csv", index=False)

    print("\nBest ANN configuration:")
    print(best)

    data = create_lag_features(
        df, best["rainfall_lag"], best["streamflow_lag"]
    )

    train = data[data["Date"].dt.year <= 2007].copy()
    val = data[data["Date"].dt.year >= 2008].copy()

    feature_cols = [
        c for c in data.columns
        if c.startswith("Rainfall_t-") or c.startswith("Streamflow_t-")
    ]

    X_train = train[feature_cols].to_numpy()
    X_val = val[feature_cols].to_numpy()
    y_train = train["Target_Streamflow"].to_numpy()
    y_val = val["Target_Streamflow"].to_numpy()

    X_train_s, X_val_s, y_train_s, y_val_s, scales = normalize_train_validation(
        X_train, X_val, y_train, y_val, feature_cols
    )

    # Final ANN with selected hyperparameters.
    ann = build_ann(
        len(feature_cols),
        int(best["hidden_size"]),
        float(best["learning_rate"])
    )

    history = ann.fit(
        X_train_s, y_train_s,
        epochs=200,
        batch_size=32,
        verbose=0
    )

    ann_pred_s = ann.predict(X_val_s, verbose=0).ravel()
    # Streamflow cannot be negative; apply this physical constraint before evaluation.
    ann_pred = np.maximum(ann_pred_s * scales["streamflow_max"], 0.0)

    ann_metrics = split_rmse(y_val, ann_pred)

    # Comparison baselines use the same selected lag features and same chronological split.
    lr = build_linear_regression()
    lr.fit(X_train_s, y_train_s)
    lr_pred_s = lr.predict(X_val_s)
    lr_pred = np.maximum(lr_pred_s * scales["streamflow_max"], 0.0)
    lr_metrics = split_rmse(y_val, lr_pred)

    rf = build_random_forest()
    rf.fit(X_train_s, y_train_s)
    rf_pred_s = rf.predict(X_val_s)
    rf_pred = np.maximum(rf_pred_s * scales["streamflow_max"], 0.0)
    rf_metrics = split_rmse(y_val, rf_pred)

    # Paper's best LSTM configuration uses equal five-day rainfall/flow sequences.
    lstm_lag = 5
    lstm_hidden_size = 12
    lstm_learning_rate = 0.001
    X_lstm, y_lstm, lstm_dates = create_lstm_features(df, lstm_lag)
    lstm_train_mask = lstm_dates.year <= 2007
    lstm_val_mask = lstm_dates.year >= 2008
    X_lstm_train, X_lstm_val = X_lstm[lstm_train_mask], X_lstm[lstm_val_mask]
    y_lstm_train, y_lstm_val = y_lstm[lstm_train_mask], y_lstm[lstm_val_mask]
    (
        X_lstm_train_s,
        X_lstm_val_s,
        y_lstm_train_s,
        _,
        lstm_scales,
    ) = normalize_train_validation(
        X_lstm_train,
        X_lstm_val,
        y_lstm_train,
        y_lstm_val,
        ["Rainfall", "Streamflow"],
    )

    lstm = build_lstm(lstm_lag, lstm_hidden_size, lstm_learning_rate)
    lstm_history = lstm.fit(
        X_lstm_train_s,
        y_lstm_train_s,
        epochs=200,
        batch_size=32,
        verbose=0,
    )
    lstm_pred_s = lstm.predict(X_lstm_val_s, verbose=0).ravel()
    lstm_pred = np.maximum(lstm_pred_s * lstm_scales["streamflow_max"], 0.0)
    lstm_metrics = split_rmse(y_lstm_val, lstm_pred)

    comparison = pd.DataFrame([
        {"Model": "ANN", **ann_metrics},
        {"Model": "LSTM", **lstm_metrics},
        {"Model": "Linear Regression", **lr_metrics},
        {"Model": "Random Forest", **rf_metrics}
    ])
    comparison.to_csv("outputs/results/model_comparison.csv", index=False)

    predictions = pd.DataFrame({
        "Date": val["Date"].values,
        "Actual_Streamflow_cfs": y_val,
        "ANN_Predicted_Streamflow_cfs": ann_pred,
        "LinearRegression_Predicted_Streamflow_cfs": lr_pred,
        "RandomForest_Predicted_Streamflow_cfs": rf_pred
    })
    predictions.to_csv("outputs/results/validation_predictions.csv", index=False)

    with open("outputs/results/best_ann_config.json", "w") as f:
        json.dump(best, f, indent=4)

    with open("outputs/results/lstm_config.json", "w") as f:
        json.dump({
            "sequence_length": lstm_lag,
            "hidden_size": lstm_hidden_size,
            "learning_rate": lstm_learning_rate,
            "epochs": 200,
            "validation_rmse": lstm_metrics["Overall_RMSE"],
        }, f, indent=4)

    pd.DataFrame({
        "Date": lstm_dates[lstm_val_mask],
        "Actual_Streamflow_cfs": y_lstm_val,
        "LSTM_Predicted_Streamflow_cfs": lstm_pred,
    }).to_csv("outputs/results/lstm_validation_predictions.csv", index=False)

    save_plots(
        val["Date"].values,
        y_val,
        ann_pred,
        history
    )
    save_plots(
        lstm_dates[lstm_val_mask],
        y_lstm_val,
        lstm_pred,
        lstm_history,
        out_dir="outputs/plots/lstm",
        model_name="LSTM",
    )

    print("\n=== ANN RMSE ===")
    print(ann_metrics)
    print("\n=== LSTM RMSE ===")
    print(lstm_metrics)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))
    print("\nOutputs saved under outputs/")

if __name__ == "__main__":
    main()
