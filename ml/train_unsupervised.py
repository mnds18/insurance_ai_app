import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.data_preprocessing import preprocess_claims_data, load_data

# --- CONFIG ---
DATA_PATH = "data/claims_data.csv"
MODEL_PATH = "models/clustering_model.pkl"
EXPERIMENT_NAME = "Claims_Clustering_Model"
N_CLUSTERS = 4

# --- LOAD & CLEAN ---
df_raw = load_data(DATA_PATH)
df = preprocess_claims_data(df_raw)

# --- FEATURES ---
features = ["location_enc", "damage_type_enc", "estimated_payout", "is_fraud"]
df_cluster = df[features].dropna()

# --- SCALE ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# --- KMEANS CLUSTERING ---
model = KMeans(n_clusters=N_CLUSTERS, random_state=42)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    model.fit(X_scaled)
    labels = model.labels_
    df_cluster["cluster"] = labels

    inertia = model.inertia_

    print(f"✅ KMeans trained with {N_CLUSTERS} clusters")
    print(f"📉 Inertia: {inertia:.2f}")

    mlflow.log_param("n_clusters", N_CLUSTERS)
    mlflow.log_metric("inertia", inertia)
    mlflow.sklearn.log_model(model, "model")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("KMeans Clustering of Insurance Claims")
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Clusters")

    plot_path = "models/cluster_plot.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    print(f"📊 Cluster plot saved and logged to MLflow: {plot_path}")
