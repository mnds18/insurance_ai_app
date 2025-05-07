# test_rag_answer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.query_rag import get_rag_answer


def test_rag_query():
    print("✅ Running test: test_rag_query")
    try:
        query = "What is covered under accidental damage policy?"
        response = get_rag_answer(query)

        print("🔍 Query:", query)
        print("🧠 Response:\n", response)

        assert "Answer:" in response, "❌ No answer section found."
        assert "Sources:" in response, "❌ No sources section found."

        print("✅ Test passed.")
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_rag_query()
