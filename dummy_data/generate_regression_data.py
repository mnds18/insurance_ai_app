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
    "claims_count": np.random.poisson(lam=20, size=n_days)
}
df = pd.DataFrame(data)

# --- Add seasonality and randomness to target ---
df["dayofyear"] = df["date"].dt.dayofyear
seasonal_pattern = 2000 + 400 * np.sin(2 * np.pi * df["dayofyear"] / 365)
rng_noise = np.random.normal(0, 300, size=n_days)
df["target"] = seasonal_pattern + df["claims_count"] * 10 + rng_noise

# --- Inject anomalies ---
anomaly_indices = np.random.choice(n_days, size=20, replace=False)
df.loc[anomaly_indices, "target"] += np.random.choice([1000, -1000], size=20)

# --- Drop helper column ---
df.drop(columns=["dayofyear"], inplace=True)

# --- Save CSV ---
os.makedirs("data", exist_ok=True)
df.to_csv("data/regression_data.csv", index=False)

print("✅ regression_data.csv generated in ./data/")
