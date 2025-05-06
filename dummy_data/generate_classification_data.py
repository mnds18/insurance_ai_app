import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# --- Settings ---
n_days = 365 * 2
start_date = datetime.today() - timedelta(days=n_days)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# --- Synthetic Features ---
data = {
    "date": dates,
    "claims_count": np.random.poisson(lam=20, size=n_days),
    "total_payout": np.random.gamma(shape=2.0, scale=1000.0, size=n_days)
}
df = pd.DataFrame(data)

# --- Inject seasonality ---
df["dayofyear"] = df["date"].dt.dayofyear
seasonal_effect = 1000 + 500 * np.sin(2 * np.pi * df["dayofyear"] / 365)
df["total_payout"] += seasonal_effect

# --- Target Generation ---
df["target"] = (df["total_payout"] > df["total_payout"].rolling(30, min_periods=1).mean() + 500).astype(int)

# --- Drop helper column ---
df.drop(columns=["dayofyear"], inplace=True)

# --- Save CSV ---
os.makedirs("data", exist_ok=True)
df.to_csv("data/classification_data.csv", index=False)

print("✅ classification_data.csv generated in ./data/")
