import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(123)

# Create date range
start_date = datetime(2022, 1, 1)
num_days = 730  # Two years of daily data

dates = [start_date + timedelta(days=i) for i in range(num_days)]
claims = np.random.poisson(lam=50, size=num_days).astype(float)
payouts = np.random.normal(loc=20000, scale=4000, size=num_days)

# Inject real-world problems
for i in range(0, num_days, 90):
    if i < num_days:
        claims[i] = np.nan  # Missing claims
    if i + 10 < num_days:
        payouts[i + 10] = np.nan  # Missing payout
    if i + 20 < num_days:
        payouts[i + 20] = payouts[i + 20] * 4  # Large spike
    if i + 25 < num_days:
        payouts[i + 25] = -abs(payouts[i + 25])  # Negative payout

# Add random noise
claims += np.random.normal(0, 5, size=num_days)
payouts += np.random.normal(0, 1000, size=num_days)

# Inject junk data
claims[50] = 999
payouts[100] = 999999

# Assemble DataFrame
df = pd.DataFrame({
    "date": dates,
    "claims_count": np.round(claims, 2),
    "total_payout": np.round(payouts, 2)
})

# Shuffle and remove 3% of days (simulate missing days)
df = df.sample(frac=1).reset_index(drop=True)
df = df.drop(df.sample(frac=0.03).index).sort_values("date")

# Save to data folder
output_path = "data/forecast_data_dirty.csv"
os.makedirs("data", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Dirty time series data saved to {output_path}")
