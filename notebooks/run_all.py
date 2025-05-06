import subprocess
import os

print("🔁 Running Forecast EDA...")
subprocess.run(["python", "notebooks/forecast_eda.py"], check=True)

print("\n🔁 Running Classification Explainability...")
subprocess.run(["python", "notebooks/classification_explainability.py"], check=True)

print("\n✅ All notebooks executed. Visuals saved in models/")
