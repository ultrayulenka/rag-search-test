"""RAG pipeline: retrieve relevant chunks and generate answer with GPT-4o-mini."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, LLM_MODEL, RETRIEVER_K
from app.ingestion import get_vector_store
from app.utils import logger

# Prompt for the LLM with placeholders for context and question
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions based only on the following context. "
            "If the context does not contain enough information, say so. "
            "Always cite the source (document and page) when possible.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def _format_doc(doc) -> str:
    """Format a single document for context."""
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "")
    return f"[Source: {source}, Page: {page}]\n{doc.page_content}"


def get_retriever(k: int | None = None):
    """Return a retriever over the Chroma collection."""
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": k or RETRIEVER_K})


def _format_context(docs) -> str:
    """Format retrieved documents as a single context string."""
    return "\n\n---\n\n".join(_format_doc(d) for d in docs)


def query(question: str) -> dict:
    """
    Run RAG: retrieve top-k chunks, format context, call GPT-4o-mini.
    Returns dict with 'answer' and 'sources' (list of {source, page}).
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)
    if not docs:
        return {"answer": "No relevant passages found in the documents.", "sources": []}

    context = _format_context(docs)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = RAG_PROMPT | llm
    try:
        response = chain.invoke({"context": context, "question": question})
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.exception("LLM call failed")
        return {"answer": f"Error generating answer: {e}", "sources": []}

    # Deduplicate sources by (source, page)
    seen = set()
    sources = []
    for d in docs:
        meta = d.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page", "")
        try:
            page = int(page) if page != "" else 0
        except (TypeError, ValueError):
            page = 0
        key = (src, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": src, "page": page})

    return {"answer": answer, "sources": sources}


def retrieve_embeddings_for_question(question: str, k: int | None = None) -> list[dict]:
    """
    For a given question, retrieve the top-k chunks and their embedding vectors.
    Returns a list of dicts with 'content', 'metadata', and 'embedding' for each chunk.
    """
    store = get_vector_store()
    n_results = k or RETRIEVER_K
    # Embed the question (use same model as stored embeddings)
    embedding_fn = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    query_embedding = embedding_fn.embed_query(question)
    # Query Chroma for documents, metadatas, and embeddings
    result = store._collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "embeddings"],
    )
    out = []
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]
    embeddings = result.get("embeddings") or [[]]
    docs_0 = documents[0] if documents else []
    meta_0 = metadatas[0] if metadatas else []
    emb_0 = embeddings[0] if embeddings else []
    for i in range(len(docs_0)):
        out.append({
            "content": docs_0[i],
            "metadata": meta_0[i] if i < len(meta_0) else {},
            "embedding": emb_0[i] if i < len(emb_0) else [],
        })
    return out
