"""
chat.py — POST /api/chat
Send a prompt to the AI model, get a response, and automatically evaluate
it for hallucinations. Persists the entire conversation in the database.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from loguru import logger
from datetime import datetime, timezone

from deployment.api.database import (
    get_db, ChatSession, ChatMessage, MitigationTechnique
)
from deployment.api.models.schemas import ChatRequest, ChatResponse
from deployment.api.services.llm_service import call_llm
from deployment.api.services.evaluation_service import evaluate_response
from deployment.api.services.mitigation_service import get_rag_context

router = APIRouter(prefix="/api/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a prompt to the AI",
    description=(
        "**POST /api/chat** — Core AI interaction endpoint.\n\n"
        "Sends the user's prompt to the selected LLM, applies the chosen mitigation "
        "technique, evaluates the response for hallucinations, and persists the full "
        "conversation turn in the database. Supports multi-turn conversations via `session_id`."
    ),
)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    logger.info(
        f"[POST /api/chat] model={request.model} | mitigation={request.mitigation} "
        f"| session_id={request.session_id} | prompt='{request.prompt[:60]}...'"
    )

    # ── 1. Resolve or create chat session ────────────────────────────────────
    if request.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session {request.session_id} not found.",
            )
    else:
        session = ChatSession(
            session_title=request.prompt[:80],
            user_id=request.user_id,
            model_name=request.model.value,
            mitigation=request.mitigation.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(session)
        db.flush()   # get session.id before commit

    # ── 2. Save user message ──────────────────────────────────────────────────
    user_msg = ChatMessage(
        session_id=session.id,
        user_id=request.user_id,
        role="user",
        content=request.prompt,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user_msg)
    db.flush()

    # ── 3. Retrieve RAG context if needed ────────────────────────────────────
    context = None
    if request.mitigation.value == "rag":
        context_str, _ = get_rag_context(request.prompt)
        context = context_str or None

    # ── 4. Call LLM ──────────────────────────────────────────────────────────
    try:
        ai_response = await call_llm(
            prompt=request.prompt,
            model=request.model.value,
            mitigation=request.mitigation.value,
            context=context,
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM API unavailable: {str(e)}",
        )

    # ── 5. Evaluate for hallucinations ───────────────────────────────────────
    scores = {
        "hallucination_score": 0.0,
        "is_hallucinated": False,
        "hallucination_type": "none_detected",
        "explanation": "",
    }
    if request.evaluate_hallucination:
        try:
            scores = await evaluate_response(
                prompt=request.prompt,
                llm_response=ai_response,
                ground_truth=request.ground_truth,
                context=context,
                model=request.model.value,
            )
        except Exception as e:
            logger.warning(f"Evaluation failed (non-fatal): {e}")

    # ── 6. Save assistant message ─────────────────────────────────────────────
    assistant_msg = ChatMessage(
        session_id=session.id,
        user_id=request.user_id,
        role="assistant",
        content=ai_response,
        hallucination_score=scores.get("hallucination_score"),
        is_hallucinated=scores.get("is_hallucinated", False),
        hallucination_type=scores.get("hallucination_type", "unknown"),
        created_at=datetime.now(timezone.utc),
    )
    db.add(assistant_msg)

    # Update session timestamp
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(assistant_msg)

    logger.success(
        f"Chat turn saved | session={session.id} | "
        f"hallucinated={scores.get('is_hallucinated')} | "
        f"score={scores.get('hallucination_score', 0):.3f}"
    )

    return ChatResponse(
        session_id=session.id,
        user_message_id=user_msg.id,
        assistant_message_id=assistant_msg.id,
        prompt=request.prompt,
        ai_response=ai_response,
        model=request.model.value,
        mitigation=request.mitigation.value,
        is_hallucinated=bool(scores.get("is_hallucinated", False)),
        hallucination_score=float(scores.get("hallucination_score", 0.0)),
        hallucination_type=scores.get("hallucination_type", "none_detected"),
        explanation=scores.get("explanation"),
        created_at=assistant_msg.created_at,
    )
