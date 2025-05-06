import mlflow
import os
import matplotlib.pyplot as plt

def log_forecast_metrics(model_name, rmse, mape):
    mlflow.log_param("model", model_name)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

def log_forecast_plot(dates, actuals, predictions, model_name, output_dir="models"):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actuals, label="Actual", marker='o')
    plt.plot(dates, predictions, label="Forecast", marker='x')
    plt.title(f"{model_name} - Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Payout")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_name}_forecast_plot.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()
