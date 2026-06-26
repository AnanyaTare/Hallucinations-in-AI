"""
mitigate.py — POST /mitigate
Apply a chosen mitigation technique to reduce hallucinations.
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from deployment.api.models.schemas import MitigateRequest, MitigateResponse
from deployment.api.services.mitigation_service import apply_mitigation

router = APIRouter(prefix="/mitigate", tags=["Mitigation"])


@router.post(
    "/",
    response_model=MitigateResponse,
    summary="Apply a mitigation technique to reduce hallucinations",
    description=(
        "Applies the selected technique (RAG, self-consistency, few-shot, or CoT) "
        "to the given prompt and returns the original vs mitigated response along "
        "with an improvement score."
    ),
)
async def mitigate(request: MitigateRequest):
    logger.info(
        f"[POST /mitigate] technique={request.technique} | model={request.model} "
        f"| prompt='{request.prompt[:60]}...'"
    )

    try:
        result = await apply_mitigation(
            prompt=request.prompt,
            model=request.model.value,
            technique=request.technique.value,
            knowledge_base_query=request.knowledge_base_query,
        )
    except Exception as e:
        logger.error(f"Mitigation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mitigation pipeline failed: {str(e)}",
        )

    return MitigateResponse(
        original_response=result["original_response"],
        mitigated_response=result["mitigated_response"],
        technique_used=result["technique_used"],
        improvement_score=result["improvement_score"],
        retrieved_docs=result.get("retrieved_docs"),
        consistency_votes=result.get("consistency_votes"),
    )
