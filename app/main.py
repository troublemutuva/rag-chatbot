from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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
            .error { color: red; }
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
                const question = input.value.trim();
                if (!question) return;

                chatbox.innerHTML += `<p><strong>You:</strong> ${question}</p>`;
                input.value = "";

                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question: question })
                    });

                    const data = await res.json();

                    if (!res.ok) {
                        const errorMsg = data.detail ? JSON.stringify(data.detail) : "Server error";
                        chatbox.innerHTML += `<p class="error"><strong>Bot Error (${res.status}):</strong> ${errorMsg}</p>`;
                    } else {
                        const botReply = data.answer || JSON.stringify(data);
                        chatbox.innerHTML += `<p><strong>Bot:</strong> ${botReply}</p>`;
                    }
                } catch (err) {
                    chatbox.innerHTML += `<p class="error"><strong>Network Error:</strong> Failed to reach server.</p>`;
                }

                chatbox.scrollTop = chatbox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


