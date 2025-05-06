import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mlflow
import matplotlib.pyplot as plt
import os

# --- Load and preprocess data ---
df = pd.read_csv("data/classification_data.csv", parse_dates=["date"])

# --- Feature Engineering ---
df = df.sort_values("date").copy()
df["dayofyear"] = df["date"].dt.dayofyear

# Lag Features
df["target_lag_1"] = df["target"].shift(1)
df["claims_lag_1"] = df["claims_count"].shift(1)

# Rolling Stats
df["target_roll_mean_7"] = df["target"].rolling(window=7).mean()
df["claims_roll_std_7"] = df["claims_count"].rolling(window=7).std()

# Holiday Feature (1 if weekend as proxy for holiday)
df["is_holiday"] = df["date"].dt.weekday >= 5

# Drop missing from lag/roll
df.dropna(inplace=True)

# --- Define features and target ---
features = [
    "dayofyear", "claims_count", "is_holiday", 
    "target_lag_1", "claims_lag_1", 
    "target_roll_mean_7", "claims_roll_std_7"
]
X = df[features]
y = df["target"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Models ---
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# --- MLflow Tracking ---
mlflow.set_experiment("Claims_Classification")
os.makedirs("models", exist_ok=True)

with mlflow.start_run():
    best_model_name = None
    best_score = 0
    best_model = None

    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
        mean_score = np.mean(scores)
        mlflow.log_metric(f"{name}_cv_roc_auc", mean_score)

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        mlflow.log_metric(f"{name}_test_roc_auc", auc)

        if auc > best_score:
            best_model_name = name
            best_score = auc
            best_model = model

        print(f"\n{name} Results:")
        print(classification_report(y_test, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    mlflow.log_param("best_model", best_model_name)
    print(f"\n✅ Best Model: {best_model_name} with AUC: {best_score:.4f}")
