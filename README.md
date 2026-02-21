# RAG PDF System

Retrieval-Augmented Generation (RAG) system for searching across PDF documents. Uses FastAPI, LangChain, OpenAI embeddings (text-embedding-3-large), GPT-4o-mini, and Chroma for persistent vector storage.

## Requirements

- Python 3.11+
- OpenAI API key

## Setup

1. **Clone or navigate to the project**
   ```bash
   cd rag-pdf-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/macOS
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   - Copy or edit `.env` and set your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

5. **Add PDFs**
   - Place your PDF files in `data/pdfs/`.

## Run ingestion

Ingest all PDFs from `data/pdfs` into the vector database (chunk, embed, store in Chroma):

```bash
# From project root — ensure the venv is activated (see Setup), then:
python -m uvicorn app.main:app --reload
# Then call POST /ingest (see below)
```

**If you see `No module named uvicorn`:** you're using system Python instead of the project venv. Activate the venv (`source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on Windows) then run `python -m uvicorn ...`, or run with the venv's Python: `.venv/bin/python -m uvicorn app.main:app --reload` (Linux/macOS) or `.venv\Scripts\python -m uvicorn app.main:app --reload` (Windows).

If you see `VersionConflict` or `DistributionNotFound` for uvicorn/httptools, you are likely running the system `uvicorn` instead of the project’s. Activate the project venv and use `python -m uvicorn` as above.

Or trigger ingestion via API after starting the server:

```bash
curl -X POST "http://127.0.0.1:8000/ingest"
# Force re-ingest (recreate index):
curl -X POST "http://127.0.0.1:8000/ingest?force_recreate=true"
```

## Start the API

From the project root with the venv activated:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  

## Example curl request

**Query:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What does the document say about compliance?\"}"
```

**Example response:**
```json
{
  "answer": "...",
  "sources": [
    {"source": "policy.pdf", "page": 32}
  ]
}
```

## Project structure

```
rag-pdf-system/
├── app/
│   ├── main.py          # FastAPI entrypoint, /ingest, /query
│   ├── ingestion.py     # PDF load, chunk, embed, Chroma
│   ├── rag_pipeline.py  # Retriever + GPT-4o-mini
│   ├── config.py        # Settings from .env
│   └── utils.py         # Logging, paths
├── data/pdfs/           # Put PDFs here
├── chroma_db/           # Persistent vector DB (auto-created)
├── .env
├── requirements.txt
└── README.md
```

## Notes

- Ingestion is skipped if the Chroma collection already has data (no duplicate ingestion). Use `?force_recreate=true` on `POST /ingest` to re-run.
- Logging is configured in `app.utils` and will print to the console.
