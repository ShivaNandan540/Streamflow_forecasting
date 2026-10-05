import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import pandas as pd

from preprocessing.cleaning import clean_data
from preprocessing.feature_engineering import create_lag_features
from preprocessing.normalization import normalize_train_validation
from models.ann import build_ann
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

    X_train_s, X_val_s, y_train_s, y_val_s, x_scaler, y_scaler =         normalize_train_validation(X_train, X_val, y_train, y_val)

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
    ann_pred = y_scaler.inverse_transform(
        ann_pred_s.reshape(-1,1)
    ).ravel()

    ann_metrics = split_rmse(y_val, ann_pred)

    # Comparison baselines use the same selected lag features and same chronological split.
    lr = build_linear_regression()
    lr.fit(X_train_s, y_train_s)
    lr_pred_s = lr.predict(X_val_s)
    lr_pred = y_scaler.inverse_transform(
        lr_pred_s.reshape(-1,1)
    ).ravel()
    lr_metrics = split_rmse(y_val, lr_pred)

    rf = build_random_forest()
    rf.fit(X_train_s, y_train_s)
    rf_pred_s = rf.predict(X_val_s)
    rf_pred = y_scaler.inverse_transform(
        rf_pred_s.reshape(-1,1)
    ).ravel()
    rf_metrics = split_rmse(y_val, rf_pred)

    comparison = pd.DataFrame([
        {"Model": "ANN", **ann_metrics},
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

    save_plots(
        val["Date"].values,
        y_val,
        ann_pred,
        history
    )

    print("\n=== ANN RMSE ===")
    print(ann_metrics)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))
    print("\nOutputs saved under outputs/")

if __name__ == "__main__":
    main()
