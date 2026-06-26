"""
history.py — GET /api/history
Retrieve conversation history — all sessions with their messages,
filterable by user, model, date range, and hallucination status.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session, joinedload
from loguru import logger
from typing import Optional

from deployment.api.database import get_db, ChatSession, ChatMessage
from deployment.api.models.schemas import (
    HistoryResponse, ChatSessionOut, ChatMessageOut
)

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get(
    "",
    response_model=HistoryResponse,
    summary="Retrieve conversation history",
    description=(
        "**GET /api/history** — Returns all chat sessions with their messages.\n\n"
        "Each session contains the full turn-by-turn conversation including hallucination "
        "scores for every AI response. Supports filtering and pagination."
    ),
)
def get_history(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(10, ge=1, le=50, description="Max sessions to return"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    model: Optional[str] = Query(None, description="Filter by model name (e.g. gpt-4o)"),
    mitigation: Optional[str] = Query(None, description="Filter by mitigation technique"),
    hallucinated_only: bool = Query(False, description="Only sessions with ≥1 hallucinated response"),
    db: Session = Depends(get_db),
):
    query = (
        db.query(ChatSession)
        .options(joinedload(ChatSession.messages))
        .order_by(ChatSession.updated_at.desc())
    )

    if user_id is not None:
        query = query.filter(ChatSession.user_id == user_id)
    if model:
        query = query.filter(ChatSession.model_name == model)
    if mitigation:
        query = query.filter(ChatSession.mitigation == mitigation)

    total = query.count()
    sessions = query.offset(skip).limit(limit).all()

    # Optional post-filter: sessions that contain at least one hallucinated reply
    if hallucinated_only:
        sessions = [
            s for s in sessions
            if any(m.is_hallucinated for m in s.messages if m.role == "assistant")
        ]

    logger.info(f"[GET /api/history] returned {len(sessions)} sessions (total={total})")
    return HistoryResponse(total_sessions=total, sessions=sessions)


@router.get(
    "/{session_id}",
    response_model=ChatSessionOut,
    summary="Get a single chat session by ID",
)
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = (
        db.query(ChatSession)
        .options(joinedload(ChatSession.messages))
        .filter(ChatSession.id == session_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")

    logger.info(f"[GET /api/history/{session_id}] {len(session.messages)} messages")
    return session


@router.delete(
    "/{session_id}",
    status_code=204,
    summary="Delete a chat session and all its messages",
)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    db.delete(session)
    db.commit()
    logger.info(f"[DELETE /api/history/{session_id}] deleted")
