import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from agents.claims_investigator_agent import create_claims_investigator_agent
from agents.risk_analyst_agent import create_risk_analyst_agent

# Paths
FORECAST_IMG = "models/forecast_comparison.png"
XGB_FEATS = "models/xgb_feature_matrix.csv"
LGB_FEATS = "models/lgb_feature_matrix.csv"

st.set_page_config(page_title="Insurance Forecast Dashboard", layout="wide")
st.title("📈 Insurance Claims Forecasting & Investigation Dashboard")

# --- Section 1: Forecast Comparison ---
st.header("Model Comparison: Actual vs Forecast")
if os.path.exists(FORECAST_IMG):
    st.image(FORECAST_IMG, caption="Forecast Comparison with RMSE/MAPE", use_column_width=True)
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

# --- Section 3: Upload + Predict New Claims CSV ---
st.header("📤 Upload CSV for Prediction (Optional)")
uploaded_file = st.file_uploader("Upload a cleaned time series CSV (date, total_payout, claims_count)", type=["csv"])

if uploaded_file:
    user_df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.write("✅ Uploaded Preview:", user_df.head())
    st.info("Run your forecasting model script separately to generate updated forecasts from this input.")

# --- Section 4: Trigger Forecast Training ---
st.header("⚙️ Run Forecast Training")
if st.button("Run train_forecasting_model.py"):
    with st.spinner("Training in progress..."):
        exit_code = os.system("python forecasting/train_forecasting_model.py")
        if exit_code == 0:
            st.success("✅ Forecasting complete. Refresh the page to see updated results.")
        else:
            st.error("❌ Forecasting script failed. Check terminal logs.")

# --- Section 5: Claims Investigator Agent ---
st.header("🕵️‍♂️ Claims Investigator Agent")
with st.form("investigator_form"):
    claim_ids = st.text_input("Enter claim IDs (comma separated, e.g., CLAIM0010, CLAIM0023)")
    submitted = st.form_submit_button("Investigate Claims")
    if submitted and claim_ids:
        with st.spinner("Running LLM-powered investigation..."):
            agent = create_claims_investigator_agent()
            query = f"For the following claims: {claim_ids}, check if they exceed their policy limits and summarize findings."
            result = agent.invoke({"input": query})
            final_answer = result.get("output", "No result returned.")
            steps = result.get("intermediate_steps", [])

            st.markdown("**Agent Final Summary:**")
            st.code(final_answer)

            if steps:
                with st.expander("🔍 Intermediate Steps"):
                    for i, (action, obs) in enumerate(steps):
                        st.markdown(f"**Step {i+1}:**")
                        st.markdown(f"- Action: `{action.tool}`")
                        st.markdown(f"- Tool Input: `{action.tool_input}`")
                        st.markdown(f"- Observation: `{obs}`")

# --- Section 6: Risk Analyst Agent ---
st.header("📊 Risk Analyst Agent")
if st.button("Run Risk Summary Agent"):
    with st.spinner("Analyzing risk data via GPT-4 agent..."):
        agent = create_risk_analyst_agent()
        result = agent.invoke({"input": "Give me a risk summary of recent insurance claims and trends"})
        final_answer = result.get("output", "No result returned.")
        steps = result.get("intermediate_steps", [])

        st.markdown("**Agent Risk Summary:**")
        st.code(final_answer)

        if steps:
            with st.expander("🔍 Intermediate Steps"):
                for i, (action, obs) in enumerate(steps):
                    st.markdown(f"**Step {i+1}:**")
                    st.markdown(f"- Action: `{action.tool}`")
                    st.markdown(f"- Tool Input: `{action.tool_input}`")
                    st.markdown(f"- Observation: `{obs}`")

st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
