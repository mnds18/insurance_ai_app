import os
import pandas as pd
import streamlit as st

def parse_claim_txt(file_path):
    """Extract claim fields from a .txt file into a dictionary."""
    data = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":", 1)
                    data[key.strip().lower()] = val.strip().replace("$", "")
    except Exception as e:
        data["error"] = f"Error reading file: {e}"
    return data

def render():
    st.subheader("🧠 Insurance AI Intelligence Platform")
    st.markdown("## 📈 Claims Dashboard")
    st.markdown("### 🤖 Insurance AI Intelligence Platform")
    st.markdown("## 🕵️ Claims Investigator Agent")

    claim_ids = st.multiselect(
        "Select one or more Claim IDs:",
        options=[f.replace(".txt", "") for f in os.listdir("data/documents") if f.endswith(".txt")]
    )

    if st.button("🔎 Investigate Claims"):
        if not claim_ids:
            st.warning("Please select at least one claim ID.")
            return

        summaries = []
        reasoning = []

        for claim_id in claim_ids:
            file_path = os.path.join("data/documents", f"{claim_id}.txt")
            if not os.path.exists(file_path):
                summaries.append(f"❌ Claim ID {claim_id}: *Not found in the database.*")
                reasoning.append(f"- `{claim_id}` not found. Cannot assess policy breach.")
                continue

            claim_data = parse_claim_txt(file_path)
            if "error" in claim_data:
                summaries.append(f"❌ Claim ID {claim_id}: Error reading file.")
                reasoning.append(f"- `{claim_id}`: {claim_data['error']}")
                continue

            try:
                cost = float(claim_data.get("estimated_cost", 0))
                limit = float(claim_data.get("policy_limit", 2500))  # Default fallback limit

                exceeds = cost > limit
                status = "🟢 Covered" if not exceeds else "🔴 Exceeds Policy Limit"

                summaries.append(
                    f"🔍 Claim ID **{claim_id}**: {status} (Cost: ${cost:.2f} / Limit: ${limit:.2f})"
                )
                reasoning.append(
                    f"- `{claim_id}` evaluated: estimated cost = ${cost:.2f}, limit = ${limit:.2f}, status = {status}."
                )
            except Exception as e:
                summaries.append(f"⚠️ Claim ID **{claim_id}**: Error during evaluation.")
                reasoning.append(f"- `{claim_id}` failed to process: {e}")

        st.markdown("### 🧾 Agent Final Summary")
        for s in summaries:
            st.markdown(f"- {s}")

        with st.expander("🧠 Intermediate Reasoning"):
            for r in reasoning:
                st.markdown(r)
