"""
Ingest documents from the /data folder into Pinecone.

Usage:
    python -m app.ingest

Supports .txt, .md, and .pdf files placed in the data/ folder.
Creates the Pinecone index automatically if it doesn't exist yet.
"""
import os
import glob

from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSIONS,
)
from app.rag import get_vector_store

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ensure_index_exists():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' "
              f"({EMBEDDING_DIMENSIONS} dimensions, cosine, serverless us-east-1)...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists — reusing it.")


def load_documents():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*")):
        if path.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif path.endswith((".txt", ".md")):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def main():
    ensure_index_exists()

    docs = load_documents()
    if not docs:
        print(f"No documents found in {DATA_DIR}. Add .txt, .md, or .pdf files and re-run.")
        return

    print(f"Loaded {len(docs)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunk(s). Embedding and upserting to Pinecone...")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    print(f"Done. {len(chunks)} chunks indexed into '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()
