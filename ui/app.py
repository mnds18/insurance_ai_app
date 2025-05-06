import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st

st.set_page_config(page_title="Insurance AI App", layout="wide")


from ui import forecast_view, claims_view, risk_view, rag_view, vision_view



def main():
    st.title("🤖 Insurance AI Intelligence Platform")

    menu = [
        "📈 Forecasting Dashboard",
        "🕵️ Claims Investigator",
        "📊 Risk Analyst",
        "📚 RAG Assistant",
        "🖼️ Vision Tools"
    ]

    choice = st.sidebar.selectbox("Select Module", menu)

    if choice == "📈 Forecasting Dashboard":
        forecast_view.render()

    elif choice == "🕵️ Claims Investigator":
        claims_view.render()

    elif choice == "📊 Risk Analyst":
        risk_view.render()

    elif choice == "📚 RAG Assistant":
        rag_view.render()

    elif choice == "🖼️ Vision Tools":
        vision_view.render()

if __name__ == "__main__":
    main()
