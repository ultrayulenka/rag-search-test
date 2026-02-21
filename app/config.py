"""Application configuration loaded from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in .env")

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"

# LLM
LLM_MODEL = "gpt-4o-mini"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval
RETRIEVER_K = 5

# Chroma collection name
CHROMA_COLLECTION_NAME = "rag_pdf_docs"
