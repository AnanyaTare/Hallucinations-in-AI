"""
analyze.py — POST /analyze
Analyze any piece of text for hallucination signals (without calling an LLM first).
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from deployment.api.models.schemas import AnalyzeRequest, AnalyzeResponse
from deployment.api.services.evaluation_service import (
    compute_semantic_similarity,
    analyze_text,
)
from deployment.api.services.llm_service import call_llm_judge

router = APIRouter(prefix="/analyze", tags=["Analysis"])


@router.post(
    "/",
    response_model=AnalyzeResponse,
    summary="Analyze text for hallucination patterns",
    description=(
        "Given a piece of LLM-generated text (and optionally a reference/ground-truth), "
        "return hallucination type, confidence, semantic similarity, and detected contradictions."
    ),
)
async def analyze(request: AnalyzeRequest):
    logger.info(f"[POST /analyze] text_len={len(request.text)}")

    # Heuristic analysis
    heuristics = analyze_text(request.text, request.reference)

    # LLM-as-judge for deeper analysis
    try:
        judge_result = await call_llm_judge(
            prompt="(direct text analysis — no original question)",
            response=request.text,
            context=request.reference,
        )
        hallucination_type = judge_result.get("hallucination_type", "unknown")
        confidence = judge_result.get("hallucination_score", 0.0)
        explanation = judge_result.get("explanation", "")
        contradictions = judge_result.get("contradictions", [])
    except Exception as e:
        logger.warning(f"LLM judge unavailable: {e}")
        hallucination_type = "unknown"
        confidence = 0.0
        explanation = "LLM judge unavailable."
        contradictions = []

    # Semantic similarity
    sim = heuristics.get("semantic_similarity") or (
        compute_semantic_similarity(request.text, request.reference)
        if request.reference else 0.0
    )

    return AnalyzeResponse(
        hallucination_type=hallucination_type,
        confidence=confidence,
        explanation=explanation,
        semantic_similarity=sim,
        contradictions_detected=contradictions,
    )
