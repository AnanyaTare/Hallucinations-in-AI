"""
evaluate.py — POST /evaluate
Runs a full hallucination evaluation: call LLM → score → save to DB.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger
from datetime import datetime, timezone

from deployment.api.database import get_db, EvaluationRun, MitigationTechnique, HallucinationType
from deployment.api.models.schemas import EvaluateRequest, EvaluateResponse, EvaluationScores
from deployment.api.services.llm_service import call_llm
from deployment.api.services.evaluation_service import evaluate_response
from deployment.api.services.mitigation_service import get_rag_context

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.post(
    "/",
    response_model=EvaluateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Evaluate an LLM response for hallucinations",
    description=(
        "Send a prompt to the chosen LLM (with optional mitigation technique), "
        "then score the response for hallucinations using BERTScore, ROUGE-L, "
        "semantic similarity, NLI, and an LLM-as-judge pipeline."
    ),
)
async def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)):
    logger.info(
        f"[POST /evaluate] model={request.model} | mitigation={request.mitigation} "
        f"| prompt='{request.prompt[:60]}...'"
    )

    # 1️⃣ Retrieve RAG context if needed
    retrieved_docs = None
    context = request.context
    if request.mitigation == "rag" and not context:
        context, retrieved_docs = get_rag_context(request.prompt)

    # 2️⃣ Call LLM
    try:
        llm_response = await call_llm(
            prompt=request.prompt,
            model=request.model.value,
            mitigation=request.mitigation.value,
            context=context,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM API call failed: {str(e)}",
        )

    # 3️⃣ Evaluate response
    try:
        scores = await evaluate_response(
            prompt=request.prompt,
            llm_response=llm_response,
            ground_truth=request.ground_truth,
            context=context,
            model=request.model.value,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        scores = {
            "hallucination_score": 0.0,
            "bert_score": None,
            "rouge_l": None,
            "faithfulness_score": None,
            "answer_relevancy": None,
            "is_hallucinated": False,
            "hallucination_type": "unknown",
            "explanation": str(e),
            "contradictions": [],
            "semantic_similarity": 0.0,
        }

    # 4️⃣ Persist to DB
    run = EvaluationRun(
        prompt=request.prompt,
        context=context,
        llm_response=llm_response,
        ground_truth=request.ground_truth,
        model_name=request.model.value,
        mitigation=MitigationTechnique(request.mitigation.value),
        hallucination_score=scores["hallucination_score"],
        bert_score=scores["bert_score"],
        rouge_l=scores["rouge_l"],
        faithfulness_score=scores["faithfulness_score"],
        answer_relevancy=scores["answer_relevancy"],
        hallucination_type=HallucinationType(
            scores.get("hallucination_type", "unknown")
        ),
        is_hallucinated=int(scores["is_hallucinated"]),
        notes=scores.get("explanation", ""),
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    logger.success(f"Evaluation saved | run_id={run.id} | hallucinated={run.is_hallucinated}")

    return EvaluateResponse(
        run_id=run.id,
        prompt=request.prompt,
        llm_response=llm_response,
        model=request.model.value,
        mitigation=request.mitigation.value,
        is_hallucinated=bool(run.is_hallucinated),
        hallucination_type=run.hallucination_type.value,
        scores=EvaluationScores(
            hallucination_score=scores["hallucination_score"],
            bert_score=scores["bert_score"],
            rouge_l=scores["rouge_l"],
            faithfulness_score=scores["faithfulness_score"],
            answer_relevancy=scores["answer_relevancy"],
        ),
        retrieved_context=retrieved_docs,
        created_at=run.created_at,
    )
