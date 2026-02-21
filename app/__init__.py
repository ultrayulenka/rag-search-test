# RAG PDF application package

# Use pysqlite3-binary when system sqlite3 is < 3.35 (required by ChromaDB).
# Must run before any submodule imports chromadb.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
