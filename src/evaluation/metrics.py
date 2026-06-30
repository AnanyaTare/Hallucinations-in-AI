"""
metrics.py — Evaluation metrics for hallucination scoring.
Implements BERTScore, ROUGE-L, semantic similarity, and NLI consistency.
"""

from loguru import logger
from typing import Optional


def compute_bert_score(candidate: str, reference: str) -> float:
    """Compute BERTScore F1 between candidate response and ground truth."""
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn([candidate], [reference], lang="en", verbose=False)
        result = float(F1[0])
        logger.debug(f"BERTScore F1: {result:.4f}")
        return result
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return 0.0


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between candidate and reference."""
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        result = rouge.compute(predictions=[candidate], references=[reference])
        score = float(result.get("rougeL", 0.0))
        logger.debug(f"ROUGE-L: {score:.4f}")
        return score
    except Exception as e:
        logger.warning(f"ROUGE-L failed: {e}")
        return 0.0


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_a = model.encode(text_a, convert_to_tensor=True)
        emb_b = model.encode(text_b, convert_to_tensor=True)
        score = float(util.cos_sim(emb_a, emb_b)[0][0])
        logger.debug(f"Semantic similarity: {score:.4f}")
        return score
    except Exception as e:
        logger.warning(f"Semantic similarity failed: {e}")
        return 0.0


def compute_nli_entailment(premise: str, hypothesis: str) -> float:
    """
    Use NLI model to check if hypothesis is entailed by premise.
    Returns entailment probability (0–1).
    """
    try:
        from transformers import pipeline
        nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,
        )
        result = nli(f"{premise} [SEP] {hypothesis}")
        scores = {r["label"].lower(): r["score"] for r in result}
        entailment = scores.get("entailment", 0.0)
        logger.debug(f"NLI entailment: {entailment:.4f}")
        return float(entailment)
    except Exception as e:
        logger.warning(f"NLI scoring failed: {e}")
        return 0.0


def score_sample(
    candidate: str,
    reference: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """
    Run all applicable metrics on a single candidate response.

    Args:
        candidate: The LLM-generated response to score.
        reference: Ground-truth answer (enables BERTScore and ROUGE-L).
        context: Retrieved document context (enables NLI faithfulness).

    Returns:
        Dict with all computed metric scores.
    """
    scores = {
        "bert_score":          None,
        "rouge_l":             None,
        "semantic_similarity": None,
        "nli_faithfulness":    None,
    }

    if reference:
        scores["bert_score"]          = compute_bert_score(candidate, reference)
        scores["rouge_l"]             = compute_rouge_l(candidate, reference)
        scores["semantic_similarity"] = compute_semantic_similarity(candidate, reference)

    if context:
        scores["nli_faithfulness"] = compute_nli_entailment(
            premise=context, hypothesis=candidate
        )

    logger.info(f"Metrics: {scores}")
    return scores
