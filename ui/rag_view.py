import streamlit as st
from rag.query_rag import get_rag_answer
import os


def render():
    st.header("📚 RAG Knowledge Assistant")

    st.markdown("Ask any insurance-related question based on indexed policies, FAQs, or dummy documents.")

    question = st.text_input("Ask a question")

    if st.button("Submit") and question:
        with st.spinner("Querying RAG system..."):
            try:
                # Check if FAISS index exists
                index_path = "rag/faiss_index/index.faiss"
                if not os.path.exists(index_path):
                    st.error(f"❌ FAISS index not found at `{index_path}`. Please run the index builder first.")
                    return

                # Query the assistant
                response = get_rag_answer(question)
                if "Sources:" in response:
                    answer_part, source_part = response.split("\n\nSources:")
                else:
                    answer_part, source_part = response, "No source info."

                st.success("Answer:")
                st.markdown(answer_part.replace("Answer:", "").strip())

                with st.expander("📄 Source Documents Used"):
                    sources = [src.strip() for src in source_part.strip().split(",") if src.strip()]
                    if sources:
                        for src in sources:
                            st.markdown(f"- {src}")
                    else:
                        st.markdown("No source documents available.")

            except FileNotFoundError as fe:
                st.error(f"File error: {fe}")
            except Exception as e:
                st.error(f"❌ RAG system error: {str(e)}")
