import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import mlflow
import matplotlib.pyplot as plt
import os

# --- Load Data ---
df = pd.read_csv("data/regression_data.csv", parse_dates=["date"])

# --- Feature Engineering ---
df = df.sort_values("date").copy()
df["dayofyear"] = df["date"].dt.dayofyear

# Lag Features
df["target_lag_1"] = df["target"].shift(1)
df["claims_lag_1"] = df["claims_count"].shift(1)

# Rolling Stats
df["target_roll_mean_7"] = df["target"].rolling(window=7).mean()
df["claims_roll_std_7"] = df["claims_count"].rolling(window=7).std()

# Holiday Feature
df["is_holiday"] = df["date"].dt.weekday >= 5

df.dropna(inplace=True)

# --- Define Features/Target ---
features = [
    "dayofyear", "claims_count", "is_holiday",
    "target_lag_1", "claims_lag_1",
    "target_roll_mean_7", "claims_roll_std_7"
]
X = df[features]
y = df["target"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Models ---
models = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

# --- MLflow Logging ---
mlflow.set_experiment("Claims_Regression")
os.makedirs("models", exist_ok=True)

with mlflow.start_run():
    best_model_name = None
    best_mape = float("inf")
    best_model = None

    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="neg_mean_absolute_error")
        mape_cv = -np.mean(scores) / np.mean(y_train) * 100
        mlflow.log_metric(f"{name}_cv_mape", mape_cv)

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        mlflow.log_metric(f"{name}_rmse", rmse)
        mlflow.log_metric(f"{name}_mape", mape)

        if mape < best_mape:
            best_mape = mape
            best_model_name = name
            best_model = model

        print(f"\n{name} Results:")
        print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    mlflow.log_param("best_model", best_model_name)
    print(f"\n✅ Best Model: {best_model_name} with MAPE: {best_mape:.2f}%")
