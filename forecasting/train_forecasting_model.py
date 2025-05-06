import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import os

# --- Load Cleaned Data ---
df = pd.read_csv("data/forecast_data_clean.csv", parse_dates=["date"])
df = df[["date", "total_payout", "claims_count"]].rename(columns={"date": "ds", "total_payout": "y"})

# --- Generate AU Holidays ---
years = list(range(df["ds"].min().year, df["ds"].max().year + 1))
au_holidays = make_holidays_df(year_list=years, country="AU")
holiday_dates = set(au_holidays["ds"])
df["is_holiday"] = df["ds"].isin(holiday_dates).astype(int)

# --- Train/Test Split ---
split_date = df["ds"].max() - timedelta(days=90)
train_df = df[df["ds"] <= split_date]
test_df = df[df["ds"] > split_date]

# --- Feature Engineering ---
def add_features(df):
    df = df.sort_values("ds").copy()
    df["dayofyear"] = df["ds"].dt.dayofyear
    df["y_lag_1"] = df["y"].shift(1)
    df["y_lag_7"] = df["y"].shift(7)
    df["claims_lag_1"] = df["claims_count"].shift(1)
    df["claims_lag_7"] = df["claims_count"].shift(7)
    df["y_roll_mean_7"] = df["y"].rolling(window=7).mean()
    df["claims_roll_std_7"] = df["claims_count"].rolling(window=7).std()
    df = df.dropna()
    return df

# --- Prophet (Regressors + AU Holidays) ---
def run_prophet(train, test):
    model = Prophet()
    model.add_country_holidays(country_name="AU")
    model.add_regressor("claims_count")
    model.add_regressor("is_holiday")
    model.fit(train[["ds", "y", "claims_count", "is_holiday"]])
    forecast = model.predict(test[["ds", "claims_count", "is_holiday"]])
    y_pred = forecast["yhat"].values
    rmse = np.sqrt(mean_squared_error(test["y"], y_pred))

    # Plot Prophet components
    plt.figure(figsize=(10, 6))
    model.plot_components(forecast)
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/prophet_components.png")
    plt.close()

    return rmse, model, y_pred

# --- XGBoost ---
def run_xgboost(train, test):
    df_full = pd.concat([train, test]).reset_index(drop=True)
    df_feat = add_features(df_full)
    train_feat = df_feat[df_feat["ds"] <= train["ds"].max()]
    test_feat = df_feat[df_feat["ds"] > train["ds"].max()]
    features = ["dayofyear", "claims_count", "is_holiday", "y_lag_1", "y_lag_7",
                "claims_lag_1", "claims_lag_7", "y_roll_mean_7", "claims_roll_std_7"]
    model = xgb.XGBRegressor()
    model.fit(train_feat[features], train_feat["y"])
    y_pred = model.predict(test_feat[features])
    rmse = np.sqrt(mean_squared_error(test_feat["y"], y_pred))
    plt.figure(figsize=(8, 5))
    xgb.plot_importance(model, importance_type='gain', title='XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig("models/xgb_feature_importance.png")
    plt.close()
    return rmse, model, y_pred, train_feat[features + ["y"]]

# --- LightGBM ---
def run_lightgbm(train, test):
    df_full = pd.concat([train, test]).reset_index(drop=True)
    df_feat = add_features(df_full)
    train_feat = df_feat[df_feat["ds"] <= train["ds"].max()]
    test_feat = df_feat[df_feat["ds"] > train["ds"].max()]
    features = ["dayofyear", "claims_count", "is_holiday", "y_lag_1", "y_lag_7",
                "claims_lag_1", "claims_lag_7", "y_roll_mean_7", "claims_roll_std_7"]
    X_train = train_feat[features]
    y_train = train_feat["y"]
    X_test = test_feat[features]
    y_test = test_feat["y"]
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    plt.figure(figsize=(8, 5))
    lgb.plot_importance(model, importance_type='gain', title='LightGBM Feature Importance')
    plt.tight_layout()
    plt.savefig("models/lgb_feature_importance.png")
    plt.close()
    return rmse, model, y_pred, train_feat[features + ["y"]]

# --- MLflow Logging and Model Selection ---
mlflow.set_experiment("Forecasting_Model_Selector")

with mlflow.start_run(run_name="With_Prophet_Regressors_Holidays"):
    rmse_p, model_p, pred_p = run_prophet(train_df, test_df)
    mape_p = np.mean(np.abs((test_df["y"] - pred_p) / test_df["y"])) * 100
    mlflow.log_metric("prophet_rmse", rmse_p)
    mlflow.log_metric("prophet_mape", mape_p)
    mlflow.log_artifact("models/prophet_components.png")

    rmse_x, model_x, pred_x, xgb_features = run_xgboost(train_df, test_df)
    mape_x = np.mean(np.abs((test_df["y"] - pred_x) / test_df["y"])) * 100
    mlflow.log_metric("xgboost_rmse", rmse_x)
    mlflow.log_metric("xgboost_mape", mape_x)

    rmse_l, model_l, pred_l, lgb_features = run_lightgbm(train_df, test_df)
    mape_l = np.mean(np.abs((test_df["y"] - pred_l) / test_df["y"])) * 100
    mlflow.log_metric("lightgbm_rmse", rmse_l)
    mlflow.log_metric("lightgbm_mape", mape_l)

    xgb_features.to_csv("models/xgb_feature_matrix.csv", index=False)
    lgb_features.to_csv("models/lgb_feature_matrix.csv", index=False)
    mlflow.log_artifact("models/xgb_feature_matrix.csv")
    mlflow.log_artifact("models/lgb_feature_matrix.csv")

    results = {
        "Prophet": (mape_p, rmse_p, pred_p),
        "XGBoost": (mape_x, rmse_x, pred_x),
        "LightGBM": (mape_l, rmse_l, pred_l),
    }

    best_model_name = min(results, key=lambda k: results[k][0])
    best_mape, best_rmse, best_pred = results[best_model_name]

    mlflow.log_param("best_model_by", "MAPE")
    mlflow.log_param("best_model", best_model_name)
    mlflow.set_tags({
        "prophet_rmse": f"{rmse_p:.2f}", "prophet_mape": f"{mape_p:.2f}",
        "xgboost_rmse": f"{rmse_x:.2f}", "xgboost_mape": f"{mape_x:.2f}",
        "lightgbm_rmse": f"{rmse_l:.2f}", "lightgbm_mape": f"{mape_l:.2f}",
        "best_model": best_model_name
    })

    plt.figure(figsize=(12, 6))
    plt.plot(test_df["ds"], test_df["y"], label="Actual", color="black", linewidth=2)
    plt.plot(test_df["ds"], pred_p, label=f"Prophet (MAPE: {mape_p:.1f}%)", linestyle="--")
    plt.plot(test_df["ds"], pred_x, label=f"XGBoost (MAPE: {mape_x:.1f}%)", linestyle="-.")
    plt.plot(test_df["ds"], pred_l, label=f"LightGBM (MAPE: {mape_l:.1f}%)", linestyle=":")

    plt.title("Forecast Comparison with Holidays & Regressors")
    plt.xlabel("Date")
    plt.ylabel("Total Payout")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/forecast_comparison.png")
    mlflow.log_artifact("models/forecast_comparison.png")
    plt.close()

    print("📊 Model Performance Summary:")
    print(f"  Prophet   - RMSE: {rmse_p:.2f}, MAPE: {mape_p:.2f}%")
    print(f"  XGBoost   - RMSE: {rmse_x:.2f}, MAPE: {mape_x:.2f}%")
    print(f"  LightGBM  - RMSE: {rmse_l:.2f}, MAPE: {mape_l:.2f}%")
    print(f"\n✅ Best model based on MAPE: {best_model_name} | MAPE: {best_mape:.2f}%, RMSE: {best_rmse:.2f}")
