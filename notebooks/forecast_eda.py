import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from scipy.stats import zscore
import os

# Load data
forecast_df = pd.read_csv("data/forecast_data_clean.csv", parse_dates=["date"])
forecast_df.set_index("date", inplace=True)

y = forecast_df["total_payout"]

# STL decomposition
stl = STL(y, period=30)
res = stl.fit()
forecast_df["trend"] = res.trend
forecast_df["seasonal"] = res.seasonal
forecast_df["resid"] = res.resid

# Anomaly tagging (Z-score)
forecast_df["resid_z"] = zscore(forecast_df["resid"].fillna(0))
forecast_df["anomaly"] = forecast_df["resid_z"].abs() > 2.5

# Plot seasonal + trend + anomaly
plt.figure(figsize=(14, 6))
plt.plot(forecast_df.index, forecast_df["total_payout"], label="Actual")
plt.plot(forecast_df.index, forecast_df["trend"], label="Trend")
plt.scatter(forecast_df.index[forecast_df["anomaly"]], forecast_df["total_payout"][forecast_df["anomaly"]], color='red', label='Anomaly')
plt.title("Forecast Time Series Decomposition & Anomalies")
plt.legend()
plt.tight_layout()
plt.savefig("models/forecast_anomaly_plot.png")
plt.show()

print("✅ Decomposition + Anomaly plot saved to models/forecast_anomaly_plot.png")
