"""
Core RAG logic: embeddings, vector store connection, retriever, and
the retrieval-augmented generation chain used to answer questions.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import (
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
)

# --- Embeddings (used both for ingesting docs and embedding queries) ---
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

# --- Vector store (Pinecone) ---
def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )


# --- Chat model ---
llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful support assistant. Answer the user's question using ONLY
the context below. If the answer isn't in the context, say you don't have
that information rather than guessing.

Context:
{context}

Question:
{question}

Answer:"""
)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    """Builds a runnable RAG chain: retrieve -> format -> prompt -> LLM -> parse."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def answer_question(question: str) -> str:
    chain = build_rag_chain()
    return chain.invoke(question)
