"""
FastAPI wrapper around the LangChain RAG chain.

Run locally:
    uvicorn app.main:app --reload --port 8000

Then POST to /chat:
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"question": "What are your support hours?"}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.rag import answer_question

app = FastAPI(
    title="LangChain RAG Chatbot",
    description="A small RAG chatbot using Gemini (LLM + embeddings) and Pinecone (vector store).",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer = answer_question(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(answer=answer)
