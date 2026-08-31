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

@app.get("/")
def read_root():
    return {"message": "LangChain RAG Chatbot API is running. Go to /docs to test endpoints."}

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
            #chatbox { height: 300px; border: 1px solid #ccc; overflow-y: scroll; padding: 10px; margin-bottom: 10px; }
            input { width: 75%; padding: 10px; }
            button { padding: 10px 15px; }
        </style>
    </head>
    <body>
        <h2>RAG Chatbot Interface</h2>
        <div id="chatbox"></div>
        <input type="text" id="question" placeholder="Type your question..." />
        <button onclick="sendQuestion()">Send</button>

        <script>
            async function sendQuestion() {
                const input = document.getElementById("question");
                const chatbox = document.getElementById("chatbox");
                const question = input.value;
                if (!question) return;

                chatbox.innerHTML += `<p><strong>You:</strong> ${question}</p>`;
                input.value = "";

                const res = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question: question })
                });
                const data = await res.json();
                chatbox.innerHTML += `<p><strong>Bot:</strong> ${data.answer}</p>`;
                chatbox.scrollTop = chatbox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
