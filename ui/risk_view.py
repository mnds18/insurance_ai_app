import streamlit as st
import pandas as pd
from pathlib import Path

def render():
    st.header("📚 Risk view")

    @st.cache_data
    def load_claim_data():
        path = Path("data/mock_claims_db.csv")
        if not path.exists():
            st.error("❌ Claim dataset not found.")
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=["date"])

    df = load_claim_data()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    st.markdown("## 🤖 Insurance AI Intelligence Platform")
    st.markdown("### 📊 Risk Analyst Agent")

    if df.empty:
        st.stop()

    if st.button("Run Risk Summary Agent"):
        try:
            # Use datetime date (not string) for grouping
            df["date_only"] = df["date"].dt.date

            df_grouped = df.groupby("date_only").agg({
                "claim_id": "count",
                "total_payout": "sum"
            }).rename(columns={"claim_id": "claims_count"}).reset_index()

            avg_claims_per_day = df_grouped["claims_count"].mean()
            std_claims = df_grouped["claims_count"].std()
            total_paid = df_grouped["total_payout"].sum()
            peak_day = df_grouped.loc[df_grouped["claims_count"].idxmax(), "date_only"]

            summary = (
                f"The risk summary indicates that over the recent period, the average number "
                f"of claims per day was around **{avg_claims_per_day:.0f}**, with a standard deviation "
                f"of approximately **{std_claims:.0f}**. The total payout across all claims "
                f"was **${total_paid:,.0f}**, with the highest claim load observed on **{peak_day}**."
            )

            st.markdown("### 🧾 Agent Risk Summary:")
            st.info(summary)

            with st.expander("🔍 Intermediate Steps"):
                st.markdown("**Step 1:**")
                st.markdown("- **Action:** `Risk Summary Generator`")
                st.markdown("- **Tool Input:** `Recent insurance claim data`")
                st.markdown("- **Observation:**")
                st.dataframe(df_grouped.describe(include="all"))

        except Exception as e:
            st.error(f"⚠️ An error occurred during risk analysis: {e}")
