"""
evaluation_service.py — AI Model Layer (Evaluation)
Computes hallucination scores using BERTScore, ROUGE-L, NLI, and LLM-as-judge.
"""

from loguru import logger
from typing import Optional
from deployment.api.config import get_settings
from deployment.api.services.llm_service import call_llm_judge

settings = get_settings()


# ── Semantic Similarity ───────────────────────────────────────────────────────
def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using sentence-transformers.
    Returns a float between -1 and 1 (higher = more similar).
    """
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


# ── BERTScore ─────────────────────────────────────────────────────────────────
def compute_bert_score(candidate: str, reference: str) -> float:
    """
    Compute BERTScore F1 between candidate response and ground truth reference.
    """
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn([candidate], [reference], lang="en", verbose=False)
        result = float(F1[0])
        logger.debug(f"BERTScore F1: {result:.4f}")
        return result
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return 0.0


# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def compute_rouge_l(candidate: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between candidate and reference.
    """
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


# ── NLI Consistency ───────────────────────────────────────────────────────────
def compute_nli_score(premise: str, hypothesis: str) -> float:
    """
    Use a NLI model to check if the hypothesis (LLM response) is entailed by
    the premise (ground truth / context). Returns entailment probability.
    """
    try:
        from transformers import pipeline
        nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=-1,  # CPU
        )
        result = nli(f"{premise} [SEP] {hypothesis}")
        label_score = {r["label"]: r["score"] for r in result}
        entailment = label_score.get("entailment", label_score.get("ENTAILMENT", 0.0))
        logger.debug(f"NLI entailment score: {entailment:.4f}")
        return float(entailment)
    except Exception as e:
        logger.warning(f"NLI scoring failed: {e}")
        return 0.0


# ── Master Evaluation Pipeline ────────────────────────────────────────────────
async def evaluate_response(
    prompt: str,
    llm_response: str,
    ground_truth: Optional[str] = None,
    context: Optional[str] = None,
    model: str = "gpt-4o",
) -> dict:
    """
    Run the full evaluation pipeline:
    1. LLM-as-judge (G-Eval) — primary hallucination signal
    2. BERTScore — semantic faithfulness (if ground_truth available)
    3. ROUGE-L — extractive faithfulness
    4. Semantic similarity
    5. NLI consistency

    Returns a unified scores dict.
    """
    scores = {
        "hallucination_score": 0.0,
        "bert_score": None,
        "rouge_l": None,
        "faithfulness_score": None,
        "answer_relevancy": None,
        "is_hallucinated": False,
        "hallucination_type": "unknown",
        "explanation": "",
        "contradictions": [],
        "semantic_similarity": 0.0,
    }

    # 1️⃣ LLM-as-judge
    try:
        judge_result = await call_llm_judge(
            prompt=prompt,
            response=llm_response,
            context=context,
            model=model,
        )
        scores["hallucination_score"] = judge_result.get("hallucination_score", 0.0)
        scores["is_hallucinated"] = judge_result.get("is_hallucinated", False)
        scores["hallucination_type"] = judge_result.get("hallucination_type", "unknown")
        scores["explanation"] = judge_result.get("explanation", "")
        scores["contradictions"] = judge_result.get("contradictions", [])
    except Exception as e:
        logger.error(f"LLM judge failed: {e}")

    # 2️⃣ BERTScore (needs ground truth)
    if ground_truth:
        scores["bert_score"] = compute_bert_score(llm_response, ground_truth)
        scores["rouge_l"] = compute_rouge_l(llm_response, ground_truth)
        scores["semantic_similarity"] = compute_semantic_similarity(llm_response, ground_truth)

    # 3️⃣ NLI faithfulness vs context
    if context:
        nli = compute_nli_score(premise=context, hypothesis=llm_response)
        scores["faithfulness_score"] = nli

    # 4️⃣ Semantic similarity vs prompt (answer relevancy proxy)
    scores["answer_relevancy"] = compute_semantic_similarity(prompt, llm_response)

    # Override is_hallucinated if hallucination_score crosses threshold
    if scores["hallucination_score"] >= settings.HALLUCINATION_THRESHOLD:
        scores["is_hallucinated"] = True

    logger.info(
        f"Evaluation complete | hallucinated={scores['is_hallucinated']} "
        f"| score={scores['hallucination_score']:.3f}"
    )
    return scores


# ── Text Analysis ─────────────────────────────────────────────────────────────
def analyze_text(text: str, reference: Optional[str] = None) -> dict:
    """
    Quick analysis of a piece of text for hallucination signals
    without calling an LLM. Returns heuristic-based signals.
    """
    import re

    # Simple heuristics for demonstration
    hedging_phrases = [
        "i think", "i believe", "i'm not sure", "approximately",
        "around", "probably", "might be", "could be",
    ]
    text_lower = text.lower()
    hedging_count = sum(1 for ph in hedging_phrases if ph in text_lower)

    result = {
        "word_count": len(text.split()),
        "hedging_phrase_count": hedging_count,
        "contains_numbers": bool(re.search(r"\d+", text)),
        "semantic_similarity": (
            compute_semantic_similarity(text, reference) if reference else None
        ),
    }

    return result
