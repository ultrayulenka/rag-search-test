"""PDF ingestion: load, chunk, embed, and store in Chroma."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PDF_DIR,
)
from app.utils import ensure_dirs, get_pdf_paths, logger

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


def _load_pdf(path: Path) -> list[Document]:
    """Load a single PDF and return documents with source and page metadata."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = path.name
        if "page" not in d.metadata and "page_number" in d.metadata:
            d.metadata["page"] = d.metadata["page_number"]
    return docs


def _chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_vector_store(embedding=None):
    """Return Chroma vector store (persistent). Reuses existing DB if present."""
    ensure_dirs()
    if embedding is None:
        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=str(CHROMA_DIR),
    )


def chroma_collection_exists() -> bool:
    """Check if Chroma collection already exists and has data."""
    try:
        store = get_vector_store()
        coll = store._collection
        if coll is None:
            return False
        return coll.count() > 0
    except Exception:
        return False


def _delete_chroma_collection() -> None:
    """Remove the Chroma collection so it can be re-created from scratch."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        client.delete_collection(CHROMA_COLLECTION_NAME)
        logger.info("Deleted existing Chroma collection %s", CHROMA_COLLECTION_NAME)
    except Exception as e:
        logger.warning("Could not delete Chroma collection: %s", e)


def _get_existing_sources() -> set[str]:
    """Return set of source filenames already in the Chroma collection."""
    try:
        store = get_vector_store()
        result = store._collection.get(include=["metadatas"], limit=100_000)
        metadatas = result.get("metadatas") or []
        return {m.get("source") for m in metadatas if m and m.get("source")}
    except Exception as e:
        logger.warning("Could not list existing sources: %s", e)
        return set()


def run_ingestion(force_recreate: bool = False, extend_only: bool = False) -> dict:
    """
    Load all PDFs from data/pdfs, chunk, embed, and store in Chroma.
    - force_recreate: if True, delete collection and re-ingest everything.
    - extend_only: if True and collection exists, add only PDFs not yet in the collection (no recreate).
    If force_recreate and extend_only are False and DB has data, skip ingestion.
    Returns summary dict with counts and any errors.
    """
    ensure_dirs()
    pdf_paths = get_pdf_paths()
    logger.info("PDF paths: %s", pdf_paths)
    if not pdf_paths:
        logger.warning("No PDFs found in %s", PDF_DIR)
        return {"pdf_count": 0, "chunk_count": 0, "message": "No PDFs to ingest", "errors": []}

    existing_sources: set[str] = set()
    if extend_only and chroma_collection_exists():
        existing_sources = _get_existing_sources()
        pdf_paths = [p for p in pdf_paths if p.name not in existing_sources]
        if not pdf_paths:
            logger.info("No new PDFs to add (all already in collection).")
            return {"pdf_count": 0, "chunk_count": 0, "message": "No new PDFs to add", "errors": []}
        logger.info("Extending collection with %d new file(s): %s", len(pdf_paths), [p.name for p in pdf_paths])
    elif not force_recreate and chroma_collection_exists():
        logger.info("Chroma collection already exists with data; skipping ingestion (use force_recreate=True or extend_only=True).")
        return {"pdf_count": len(get_pdf_paths()), "chunk_count": "skipped", "message": "DB exists; ingestion skipped", "errors": []}

    if force_recreate:
        _delete_chroma_collection()

    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    all_docs: list[Document] = []
    errors: list[str] = []

    for path in pdf_paths:
        try:
            docs = _load_pdf(path)
            for d in docs:
                d.metadata["page"] = d.metadata.get("page") or d.metadata.get("page_number", 0)
            all_docs.extend(docs)
            logger.info("Loaded %s (%d pages)", path.name, len(docs))
        except Exception as e:
            errors.append(f"{path.name}: {e}")
            logger.exception("Failed to load %s", path.name)

    if not all_docs:
        return {"pdf_count": len(pdf_paths), "chunk_count": 0, "message": "No documents loaded", "errors": errors}

    chunks = _chunk_documents(all_docs)
    logger.info("Created %d chunks from %d pages", len(chunks), len(all_docs))

    # Normalize metadata for Chroma (str values)
    for c in chunks:
        for k, v in list(c.metadata.items()):
            if not isinstance(v, (str, int, float, bool)) or (isinstance(v, (int, float)) and k != "page"):
                c.metadata[k] = str(v) if v is not None else ""
        if "page" in c.metadata and isinstance(c.metadata["page"], (int, float)):
            c.metadata["page"] = int(c.metadata["page"])

    try:
        if force_recreate or not chroma_collection_exists():
            store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=CHROMA_COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
            )
        else:
            store = get_vector_store(embedding)
            store.add_documents(chunks)
    except Exception as e:
        logger.exception("Failed to add documents to Chroma")
        errors.append(f"Chroma: {e}")
        return {"pdf_count": len(pdf_paths), "chunk_count": len(chunks), "message": "Ingestion failed", "errors": errors}

    return {
        "pdf_count": len(pdf_paths),
        "chunk_count": len(chunks),
        "message": "Ingestion complete",
        "errors": errors,
    }
