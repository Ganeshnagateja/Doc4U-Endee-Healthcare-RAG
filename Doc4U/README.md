# Doc4U — Endee-powered AI Healthcare RAG Assistant

Doc4U is a Flask-based AI healthcare assistant upgraded for the Endee.io internship project evaluation. The app keeps the existing chatbot UI while adding an Endee vector database retrieval layer for semantic search, RAG, report-aware context retrieval, and source-backed answers.

This repository is forked from the official Endee repository as required for the Endee.io project-based internship evaluation.

The actual AI healthcare chatbot project is inside:

`Doc4U/`

It contains the Flask backend, Gemini integration, Endee RAG layer, semantic search APIs, frontend pages, setup instructions, and deployment files.
## Why this project fits Endee's criteria

- **Uses Endee Vector Database:** `endee_rag.py` integrates `langchain-endee`, `endee`, and `endee-model` through `EndeeVectorStore`.
- **Semantic Search:** `/api/rag/search?q=...` retrieves relevant health/report context using vector similarity.
- **RAG Pipeline:** `/ask_bot` retrieves context from Endee before calling Gemini, then injects that context into the model prompt.
- **AI-driven Application:** The assistant supports health Q&A, report/PDF analysis, image input, saved chat history, WHO vaccination data, and outbreak alerts.
- **Production-style System:** JWT auth, bcrypt password hashing, MongoDB persistence, environment-based config, modular RAG service, API endpoints, and a clear setup guide.

## Main features

1. **Endee RAG chat** — user messages are enriched with retrieved context before Gemini generates an answer.
2. **PDF report ingestion** — uploaded PDFs are extracted, chunked, embedded, and indexed for future semantic retrieval.
3. **Manual knowledge ingestion** — authenticated users can add text notes to the retrieval layer using `/api/rag/ingest-text`.
4. **Semantic search endpoint** — search the indexed medical/report knowledge directly.
5. **Source citations** — bot responses return source metadata that the UI displays under answers.
6. **Healthcare safety prompt** — responses avoid final diagnosis claims and mention emergency red flags when relevant.
7. **WHO APIs** — existing vaccine coverage and disease alert endpoints are preserved.

## System design

```text
User Browser
   |
   | HTML/CSS/JS UI
   v
Flask Backend
   |-- JWT + bcrypt authentication
   |-- MongoDB users and chat history
   |-- PDF/image handling
   |-- WHO vaccine + outbreak APIs
   |
   | RAG path
   v
endee_rag.py
   |-- chunk uploaded/seed text
   |-- embed with all-MiniLM-L6-v2
   |-- store/search vectors in Endee
   |-- fallback lexical retriever only for demo resilience
   v
Gemini Model
   |
   v
Answer + citations returned to UI
```

## Project structure

```text
.
├── app.py                         # Flask backend and API routes
├── endee_rag.py                   # Endee vector DB integration and RAG service
├── chatbot.html                   # Existing chatbot UI, citation display already supported
├── login.html                     # Login/signup UI
├── data/medical_knowledge_seed.json # Default safety/RAG seed knowledge
├── requirements.txt               # Python dependencies including Endee integration
├── .env.example                   # Safe environment variable template
├── scripts/run_local.sh           # Local setup helper
├── tests/test_endee_rag.py        # Basic retrieval utility tests
└── SUBMISSION_CHECKLIST.md        # Endee internship submission checklist
```

## Setup

### 1. Fork and star Endee

The internship description requires you to star and fork the official repository:

- Official repo: `https://github.com/endee-io/endee`
- Star it from your GitHub account.
- Fork it to your GitHub account.
- Put this project inside your fork or push this upgraded app to the fork you will submit.

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
MONGO_URI=mongodb://localhost:27017/doc4u
GOOGLE_API_KEY=your_gemini_key
JWT_SECRET_KEY=your_long_secret
ENDEE_ENABLED=true
ENDEE_INDEX_NAME=doc4u_health_index
ENDEE_DIMENSION=384
ENDEE_API_TOKEN=                # keep empty for local Endee, add token for Endee Cloud
```

### 3. Run Endee

Use the official Endee repo setup. The current Endee README says the fastest local path is:

```bash
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh
```

The Endee server listens on port `8080` by default. If you use Endee Cloud, set `ENDEE_API_TOKEN` in `.env`.

### 4. Run MongoDB

Local MongoDB example:

```bash
mongod --dbpath ./mongo-data
```

Or use MongoDB Atlas and paste the URI in `.env`.

### 5. Start the app

```bash
python app.py
```

Open:

```text
http://localhost:3000
```

## API endpoints to demonstrate

### Endee status

```bash
curl http://localhost:3000/api/endee/status
```

### Ingest custom text into Endee

```bash
curl -X POST http://localhost:3000/api/rag/ingest-text \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Fever care note","text":"For mild fever, monitor temperature, hydrate, and seek care if symptoms worsen."}'
```

### Semantic search

```bash
curl "http://localhost:3000/api/rag/search?q=fever%20hydration&top_k=4" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### RAG chat

Use the UI or call `/ask_bot`. The backend retrieves relevant Endee context first, then calls Gemini.

## What was upgraded from the original project

| Area | Before | Upgraded version |
|---|---|---|
| Vector DB | Not present | Endee VectorStore integration |
| RAG | Generic Gemini chat | Retrieval-augmented Gemini responses |
| Semantic search | Not present | `/api/rag/search` endpoint |
| Document memory | Files saved in chat history | PDF text chunked and indexed for future retrieval |
| Citations | UI support existed | Backend now returns retrieved source metadata |
| Evaluation readiness | Healthcare chatbot only | Endee-focused AI/ML project with README, design, setup, and APIs |

## Notes for evaluators

- `endee_rag.py` is the core Endee integration file.
- The fallback retriever exists only to keep the demo usable when Endee is not running on the evaluator's machine. The project still includes the real Endee integration and dependencies.
- For full evaluation, run Endee locally or configure Endee Cloud before starting the Flask app.

## Safety disclaimer

Doc4U provides general health information and report explanations. It is not a replacement for a licensed doctor, emergency care, or clinical diagnosis.
