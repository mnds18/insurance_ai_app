from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

INDEX_DIR = "rag/faiss_index"

def get_rag_answer(query: str) -> str:
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"question": query})
    answer = result.get("result", "No answer.")

    sources = ", ".join(
        doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])
    )

    return f"Answer: {answer}\n\nSources: {sources or 'No sources provided.'}"
