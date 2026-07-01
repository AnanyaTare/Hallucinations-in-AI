"""
mitigation_service.py — AI Model Layer (Mitigation)
Implements RAG, self-consistency, and prompt-engineering mitigation techniques.
"""

from loguru import logger
from typing import Optional, List, Tuple
from deployment.api.config import get_settings
from deployment.api.services.llm_service import call_llm, call_llm_n_times
from deployment.api.services.evaluation_service import compute_semantic_similarity
from collections import Counter

settings = get_settings()


# ── RAG Mitigation ────────────────────────────────────────────────────────────
def get_rag_context(query: str, top_k: int = None) -> Tuple[str, List[str]]:
    """
    Retrieve relevant documents from ChromaDB vector store.
    Returns (combined_context_string, list_of_doc_snippets).
    """
    top_k = top_k or settings.TOP_K_RETRIEVAL
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        collection = client.get_or_create_collection("hallucination_kb")

        results = collection.query(query_texts=[query], n_results=top_k)
        docs: List[str] = results["documents"][0] if results["documents"] else []

        if not docs:
            logger.warning("RAG: No documents found in ChromaDB for query.")
            return "", []

        combined = "\n\n".join(f"[Doc {i+1}]: {d}" for i, d in enumerate(docs))
        logger.info(f"RAG: Retrieved {len(docs)} documents for query.")
        return combined, docs

    except Exception as e:
        logger.warning(f"RAG retrieval failed (ChromaDB may not be initialized): {e}")
        return "", []


async def rag_mitigate(
    prompt: str,
    model: str,
    knowledge_base_query: Optional[str] = None,
) -> dict:
    """
    Apply RAG: retrieve context → inject into LLM → get grounded answer.
    """
    query = knowledge_base_query or prompt
    context, docs = get_rag_context(query)

    # Get baseline response (without RAG)
    original = await call_llm(prompt, model=model, mitigation="none")

    # Get RAG-grounded response
    if context:
        mitigated = await call_llm(prompt, model=model, mitigation="rag", context=context)
    else:
        logger.warning("RAG context empty — falling back to base response.")
        mitigated = original

    improvement = compute_semantic_similarity(mitigated, original)

    return {
        "original_response": original,
        "mitigated_response": mitigated,
        "technique_used": "rag",
        "retrieved_docs": docs,
        "improvement_score": improvement,
        "consistency_votes": None,
    }


# ── Self-Consistency Mitigation ───────────────────────────────────────────────
async def self_consistency_mitigate(
    prompt: str,
    model: str,
    n: int = None,
) -> dict:
    """
    Self-consistency: generate N responses and select the most common answer.
    Returns the most voted response and vote distribution.
    """
    n = n or settings.SELF_CONSISTENCY_N

    # Get baseline (single call, temp=0)
    original = await call_llm(prompt, model=model)

    # Generate N diverse responses
    responses = await call_llm_n_times(prompt, model=model, n=n, temperature=0.7)

    # Majority voting via semantic clustering (simplified: exact-ish grouping)
    # For a production system, you'd cluster by semantic similarity
    vote_counter = Counter(responses)
    most_common = vote_counter.most_common(1)[0][0]
    total = sum(vote_counter.values())

    votes_pct = {
        resp[:80] + ("..." if len(resp) > 80 else ""): round(count / total * 100, 1)
        for resp, count in vote_counter.most_common()
    }

    improvement = compute_semantic_similarity(most_common, original)

    logger.info(
        f"Self-consistency: {n} samples, winner got {vote_counter[most_common]}/{total} votes"
    )

    return {
        "original_response": original,
        "mitigated_response": most_common,
        "technique_used": "self_consistency",
        "retrieved_docs": None,
        "improvement_score": improvement,
        "consistency_votes": votes_pct,
    }


# ── Prompt Engineering Mitigation ─────────────────────────────────────────────
async def prompt_engineering_mitigate(
    prompt: str,
    model: str,
    technique: str = "few_shot",
) -> dict:
    """
    Apply prompt engineering (few-shot or chain-of-thought) to reduce hallucinations.
    """
    # Baseline
    original = await call_llm(prompt, model=model, mitigation="none")

    # Mitigated
    mitigated = await call_llm(prompt, model=model, mitigation=technique)

    improvement = compute_semantic_similarity(mitigated, original)

    return {
        "original_response": original,
        "mitigated_response": mitigated,
        "technique_used": technique,
        "retrieved_docs": None,
        "improvement_score": improvement,
        "consistency_votes": None,
    }


# ── Dispatch ──────────────────────────────────────────────────────────────────
async def apply_mitigation(
    prompt: str,
    model: str,
    technique: str,
    knowledge_base_query: Optional[str] = None,
) -> dict:
    """Router: dispatch to the correct mitigation handler."""
    if technique == "rag":
        return await rag_mitigate(prompt, model, knowledge_base_query)
    elif technique == "self_consistency":
        return await self_consistency_mitigate(prompt, model)
    elif technique in ("few_shot", "chain_of_thought"):
        return await prompt_engineering_mitigate(prompt, model, technique)
    else:
        # "none" — just return the plain response
        response = await call_llm(prompt, model=model)
        return {
            "original_response": response,
            "mitigated_response": response,
            "technique_used": "none",
            "retrieved_docs": None,
            "improvement_score": 1.0,
            "consistency_votes": None,
        }
