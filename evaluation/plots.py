import os
import matplotlib.pyplot as plt

def save_plots(dates, actual, predicted, history, out_dir="outputs/plots"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (ft³/s)")
    plt.title("Actual vs Predicted Streamflow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual - predicted)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Prediction Error (ft³/s)")
    plt.title("Streamflow Prediction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prediction_error.png"), dpi=200)
    plt.close()

    if history is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("ANN Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ann_training_loss.png"), dpi=200)
        plt.close()
