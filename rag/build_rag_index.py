import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from pathlib import Path

DOCS_DIR = "data/rag_documents"
INDEX_DIR = "rag/faiss_index"

os.makedirs(INDEX_DIR, exist_ok=True)

def build_rag_index():
    print("🔄 Loading documents...")
    all_docs = []
    for file in Path(DOCS_DIR).rglob("*.pdf"):
        loader = PyPDFLoader(str(file))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = file.name
        all_docs.extend(pages)

    print(f"✅ Loaded {len(all_docs)} pages.")

    print("✂️ Splitting text into chunks...")
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    print("🔍 Creating FAISS vector index...")
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(INDEX_DIR)
    print(f"✅ Saved index to {INDEX_DIR}")

if __name__ == "__main__":
    build_rag_index()
