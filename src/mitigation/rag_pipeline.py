"""
rag_pipeline.py — Retrieval-Augmented Generation (RAG) mitigation.
Builds a ChromaDB knowledge base and retrieves context for grounded answers.
"""

from loguru import logger
from pathlib import Path
from typing import List, Tuple, Optional
from src.utils.api_client import query_llm_sync

CHROMA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "processed" / "chroma_db")


# ── Vector Store ──────────────────────────────────────────────────────────────
def get_or_create_collection(collection_name: str = "hallucination_kb"):
    """Return a ChromaDB collection, creating it if it doesn't exist."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_or_create_collection(collection_name)
    except Exception as e:
        logger.error(f"ChromaDB init failed: {e}")
        return None


def ingest_documents(
    documents: List[str],
    metadatas: Optional[List[dict]] = None,
    collection_name: str = "hallucination_kb",
) -> int:
    """
    Add documents to the ChromaDB knowledge base.

    Args:
        documents: List of text chunks to store.
        metadatas: Optional list of metadata dicts (same length as documents).
        collection_name: ChromaDB collection to use.

    Returns:
        Number of documents ingested.
    """
    collection = get_or_create_collection(collection_name)
    if collection is None:
        return 0

    metadatas = metadatas or [{"source": f"doc_{i}"} for i in range(len(documents))]
    ids = [f"doc_{i}_{hash(d) % 1_000_000}" for i, d in enumerate(documents)]

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.success(f"[rag] Ingested {len(documents)} documents into '{collection_name}'")
    return len(documents)


def retrieve_context(
    query: str,
    top_k: int = 5,
    collection_name: str = "hallucination_kb",
) -> Tuple[str, List[str]]:
    """
    Retrieve the top-k most relevant document chunks for a query.

    Args:
        query: Search query string.
        top_k: Number of documents to retrieve.
        collection_name: ChromaDB collection to search.

    Returns:
        Tuple of (combined_context_string, list_of_individual_docs).
    """
    collection = get_or_create_collection(collection_name)
    if collection is None:
        return "", []

    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        docs: List[str] = results["documents"][0] if results["documents"] else []
        if not docs:
            logger.warning(f"[rag] No documents found for query: '{query[:50]}'")
            return "", []

        combined = "\n\n".join(f"[Source {i+1}]: {d}" for i, d in enumerate(docs))
        logger.info(f"[rag] Retrieved {len(docs)} chunks for query")
        return combined, docs
    except Exception as e:
        logger.error(f"[rag] Retrieval failed: {e}")
        return "", []


def rag_answer(
    question: str,
    model: str = "gpt-4o",
    top_k: int = 5,
    collection_name: str = "hallucination_kb",
) -> dict:
    """
    Full RAG pipeline: retrieve context → generate grounded answer.

    Returns:
        Dict with 'answer', 'context', and 'retrieved_docs'.
    """
    context, docs = retrieve_context(question, top_k, collection_name)

    if not context:
        logger.warning("[rag] Empty context — falling back to base LLM call.")

    answer = query_llm_sync(
        prompt=question,
        model=model,
        context=context or None,
        temperature=0.0,
    )

    return {"answer": answer, "context": context, "retrieved_docs": docs}
