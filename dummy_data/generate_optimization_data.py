import pandas as pd
import numpy as np
import os
import json

# Output paths
os.makedirs("data", exist_ok=True)
claims_output_path = "data/optimization_input.csv"
constraints_output_path = "data/optimization_constraints.json"

# Set seed for reproducibility
np.random.seed(42)

# Simulate 100 insurance claims
num_claims = 100
claims = pd.DataFrame({
    "claim_id": [f"CLAIM{i:04d}" for i in range(num_claims)],
    "estimated_payout": np.random.uniform(5000, 25000, num_claims).round(2),
    "risk_score": np.random.uniform(0.1, 0.95, num_claims).round(2),
    "resources_required": np.random.randint(1, 4, num_claims),  # 1 to 3 assessors
    "priority_flag": np.random.choice(["High", "Medium", "Low"], p=[0.3, 0.5, 0.2], size=num_claims)
})

# Define constraints
constraints = {
    "total_assessors_available": 100,
    "max_total_payout": 1200000,
    "max_avg_risk_score": 0.6
}

# Save to disk
claims.to_csv(claims_output_path, index=False)
with open(constraints_output_path, "w") as f:
    json.dump(constraints, f, indent=4)

print(f"✅ Claims data saved to: {claims_output_path}")
print(f"📌 Constraints saved to: {constraints_output_path}")
