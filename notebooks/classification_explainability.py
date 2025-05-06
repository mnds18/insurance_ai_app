import pandas as pd
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load feature matrix
df = pd.read_csv("models/lgb_feature_matrix.csv")
X = df.drop(columns=["target", "date"], errors="ignore")
y = df["target"]

# Split for local explainability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LGBM model for SHAP
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
preds = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, preds))

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test)

# Save SHAP bar plot
plt.title("SHAP Feature Importance")
shap.plots.bar(shap_values[1], max_display=10, show=False)
plt.savefig("models/classification_shap_bar.png")
print("✅ SHAP bar plot saved to models/classification_shap_bar.png")
