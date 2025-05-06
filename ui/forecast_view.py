import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Paths
FORECAST_IMG = "models/forecast_comparison.png"
XGB_FEATS = "models/xgb_feature_matrix.csv"
LGB_FEATS = "models/lgb_feature_matrix.csv"

def render():
    st.header("📈 Forecasting Dashboard")

    # --- Section 1: Forecast Comparison ---
    st.subheader("Model Comparison: Actual vs Forecast")
    if os.path.exists(FORECAST_IMG):
        st.image(FORECAST_IMG, caption="Forecast Comparison with RMSE/MAPE", use_container_width=True)
    else:
        st.warning("Forecast plot not found. Please run the training script first.")

    # --- Section 2: Feature Data Explorer ---
    with st.expander("🔍 Explore Feature Data"):
        tabs = st.tabs(["XGBoost Features", "LightGBM Features"])
        with tabs[0]:
            if os.path.exists(XGB_FEATS):
                df_xgb = pd.read_csv(XGB_FEATS)
                st.dataframe(df_xgb.tail(20))
            else:
                st.warning("XGBoost feature matrix not found.")
        with tabs[1]:
            if os.path.exists(LGB_FEATS):
                df_lgb = pd.read_csv(LGB_FEATS)
                st.dataframe(df_lgb.tail(20))
            else:
                st.warning("LightGBM feature matrix not found.")

    # --- Section 3: Upload New Data ---
    st.subheader("📤 Upload CSV for Prediction (Optional)")
    uploaded_file = st.file_uploader("Upload a cleaned time series CSV (date, total_payout, claims_count)", type=["csv"])
    if uploaded_file:
        user_df = pd.read_csv(uploaded_file, parse_dates=["date"])
        st.write("✅ Uploaded Preview:", user_df.head())
        st.info("Run your forecasting model script separately to generate updated forecasts from this input.")

    # --- Section 4: Trigger Training ---
    st.subheader("⚙️ Run Forecast Training")
    if st.button("Run train_forecasting_model.py"):
        with st.spinner("Training in progress..."):
            exit_code = os.system("python forecasting/train_forecasting_model.py")
            if exit_code == 0:
                st.success("✅ Forecasting complete. Refresh the page to see updated results.")
            else:
                st.error("❌ Forecasting script failed. Check terminal logs.")

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
