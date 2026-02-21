"""FastAPI entrypoint for RAG PDF system."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.ingestion import run_ingestion
from app.rag_pipeline import query, retrieve_embeddings_for_question
from app.utils import logger

# Run ingestion in thread so we don't block the event loop
def _run_ingestion_sync(force_recreate: bool = False, extend_only: bool = False):
    return run_ingestion(force_recreate=force_recreate, extend_only=extend_only)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("RAG PDF API starting")
    yield
    logger.info("RAG PDF API shutting down")


app = FastAPI(title="RAG PDF API", version="1.0.0", lifespan=lifespan)


class QueryBody(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., min_length=1, description="Question to answer from the documents")


class RetrieveBody(BaseModel):
    """Request body for POST /retrieve-embeddings."""

    question: str = Field(..., min_length=1, description="Question to get retrieved chunks and embeddings for")
    k: int | None = Field(default=None, ge=1, le=50, description="Number of chunks to return (default from config)")


class QueryResponse(BaseModel):
    """Response for POST /query."""

    answer: str
    sources: list[dict]  # [{"source": str, "page": int}, ...]


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with its embedding."""

    content: str
    metadata: dict
    embedding: list[float]


class RetrieveEmbeddingsResponse(BaseModel):
    """Response for POST /retrieve-embeddings."""

    question: str
    chunks: list[RetrievedChunk]


class IngestResponse(BaseModel):
    """Response for POST /ingest."""

    message: str
    pdf_count: int
    chunk_count: int | str
    errors: list[str]


@app.post("/ingest", response_model=IngestResponse)
async def ingest(force_recreate: bool = False, extend_only: bool = False):
    """Run PDF ingestion: load PDFs from data/pdfs, chunk, embed, store in Chroma. Use extend_only=true to add only new PDFs without recreating the collection."""
    try:
        result = await asyncio.to_thread(_run_ingestion_sync, force_recreate, extend_only)
        return IngestResponse(
            message=result["message"],
            pdf_count=result["pdf_count"],
            chunk_count=result["chunk_count"],
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def post_query(body: QueryBody):
    """Answer a question using RAG over ingested PDFs."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")
    try:
        result = await asyncio.to_thread(query, body.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve-embeddings", response_model=RetrieveEmbeddingsResponse)
async def retrieve_embeddings(body: RetrieveBody):
    """Retrieve all chunks and their embedding vectors for a given question (top-k by similarity)."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must be non-empty")
    try:
        chunks = await asyncio.to_thread(
            retrieve_embeddings_for_question, body.question, body.k
        )
        return RetrieveEmbeddingsResponse(
            question=body.question,
            chunks=[RetrievedChunk(content=c["content"], metadata=c["metadata"], embedding=c["embedding"]) for c in chunks],
        )
    except Exception as e:
        logger.exception("Retrieve embeddings failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}
