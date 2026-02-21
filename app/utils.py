"""Utility helpers."""

import logging
from pathlib import Path

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag-pdf-system")


def ensure_dirs() -> None:
    """Create data and chroma directories if they do not exist."""
    from app.config import CHROMA_DIR, PDF_DIR

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_paths() -> list[Path]:
    """Return list of PDF file paths in data/pdfs."""
    from app.config import PDF_DIR

    ensure_dirs()
    logger.info("Ensured dirs: %s", ensure_dirs())
    return sorted(PDF_DIR.glob("*.pdf"))
