# LangChain RAG Chatbot (Gemini + Pinecone, Free Tier)

A minimal but production-shaped Retrieval-Augmented Generation (RAG) chatbot:

- **LLM + embeddings:** Google Gemini (`gemini-2.5-flash` + `gemini-embedding-001`) — free tier, no credit card
- **Vector store:** Pinecone (Starter/free plan)
- **Framework:** LangChain (chains, retrievers, document loaders)
- **API layer:** FastAPI
- **Deployment:** Dockerfile included

Project structure:
```
langchain-rag-project/
├── app/
│   ├── __init__.py
│   ├── config.py     # loads env vars
│   ├── rag.py         # embeddings, vector store, RAG chain
│   ├── ingest.py      # loads /data files into Pinecone
│   └── main.py        # FastAPI app (POST /chat)
├── data/
│   └── sample_faq.txt # example doc to index
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## 1. Get your free API keys

**Gemini (Google AI Studio):**
1. Go to https://aistudio.google.com/apikey
2. Sign in, click "Create API key" (no billing needed)
3. Copy the key

**Pinecone:**
1. Go to https://app.pinecone.io and sign up (free Starter plan)
2. Left sidebar → API Keys → copy your key
3. You do NOT need to manually create the index — `ingest.py` creates it
   automatically with the correct settings (3072 dimensions, cosine, serverless
   us-east-1) the first time you run it.

## 2. Set up the project locally

```bash
# Clone/unzip the project, then:
cd langchain-rag-project

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# then edit .env and paste in your GOOGLE_API_KEY and PINECONE_API_KEY
```

## 3. Index your documents

Drop `.txt`, `.md`, or `.pdf` files into the `data/` folder (a sample FAQ file
is already there to test with), then run:

```bash
python -m app.ingest
```

This will:
- Create the Pinecone index if it doesn't exist yet
- Split your documents into chunks
- Embed them with Gemini and upsert them into Pinecone

Re-run this any time you add new documents to `data/`.

## 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are your support hours?"}'
```

Or open http://localhost:8000/docs for the interactive Swagger UI to test
directly in the browser.

## 5. Run with Docker (optional)

```bash
docker build -t rag-chatbot .

docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key_here \
  -e PINECONE_API_KEY=your_key_here \
  -e PINECONE_INDEX_NAME=rag-chatbot \
  rag-chatbot
```

Note: run `python -m app.ingest` locally first (or add an ingest step to your
deployment) since the container itself doesn't auto-index on startup.

## 6. Deploying to AWS (EC2/ECR), matching your existing stack

Since you're already comfortable with EC2 + ECR + GitHub Actions:

1. Build and push the image to ECR (same flow as your crypto-predictor project)
2. On EC2, run the container with the same `-e` env vars shown above
3. Open port 8000 in your EC2 security group's inbound rules
4. Point a GitHub Actions workflow at this repo the same way you likely have
   for your other projects, so pushes to `main` rebuild and redeploy automatically

## Troubleshooting

- **"insufficient_quota" / rate limit errors** — you're hitting Gemini's free
  tier RPM cap (1,500 req/day on Flash). Wait a minute and retry, or space out
  test calls.
- **Dimension mismatch error from Pinecone** — make sure you didn't manually
  create the index with a different dimension than 3072. Delete it in the
  Pinecone console and let `ingest.py` recreate it.
- **Empty/irrelevant answers** — check `data/` actually has content and that
  you re-ran `python -m app.ingest` after adding new files.
