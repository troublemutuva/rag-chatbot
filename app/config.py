import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# gemini-embedding-001 defaults to 3072 dimensions
EMBEDDING_DIMENSIONS = 3072

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in.")
