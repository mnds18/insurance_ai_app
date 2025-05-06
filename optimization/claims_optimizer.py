import pandas as pd
import json
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import os

# Paths
data_path = "data/optimization_input.csv"
constraints_path = "data/optimization_constraints.json"

# Load data
claims = pd.read_csv(data_path)
with open(constraints_path) as f:
    constraints = json.load(f)

# Objective: Maximize utility = estimated_payout / risk_score
# (i.e., get more payout per unit risk)
claims["utility"] = claims["estimated_payout"] / claims["risk_score"]

# Optimization Variables: x_i in {0,1} (claim selected or not)
c = -claims["utility"].values  # Negative because linprog minimizes

# Constraints
A = []
b = []

# 1. Total resource constraint (assessors)
A.append(claims["resources_required"].values)
b.append(constraints["total_assessors_available"])

# 2. Max payout constraint
A.append(claims["estimated_payout"].values)
b.append(constraints["max_total_payout"])

# Bounds for binary decision variables
bounds = [(0, 1) for _ in range(len(claims))]

# linprog can’t handle integer/binary constraints directly,
# so we relax and round the result (or use MILP via another lib if needed)
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

claims["selected"] = np.round(res.x)

# Evaluation metrics
total_selected = claims["selected"].sum()
total_payout = (claims["selected"] * claims["estimated_payout"]).sum()
avg_risk = (claims["selected"] * claims["risk_score"]).sum() / max(total_selected, 1)

# MLflow logging
mlflow.set_experiment("Claims_Optimization")
with mlflow.start_run():
    mlflow.log_param("solver", "scipy.linprog")
    mlflow.log_metric("objective_value", -res.fun)
    mlflow.log_metric("total_selected", total_selected)
    mlflow.log_metric("total_payout", total_payout)
    mlflow.log_metric("avg_risk_score", avg_risk)

    # Save plot
    fig, ax = plt.subplots()
    ax.hist(claims[claims["selected"] == 1]["risk_score"], bins=10, alpha=0.7, label="Selected")
    ax.hist(claims[claims["selected"] == 0]["risk_score"], bins=10, alpha=0.5, label="Rejected")
    ax.set_title("Risk Score Distribution by Selection")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plot_path = "reports/optimization_result.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

print("\n✅ Optimization complete")
print(f"Selected claims: {int(total_selected)}")
print(f"Total payout: ${total_payout:,.2f}")
print(f"Average risk score: {avg_risk:.2f}")
