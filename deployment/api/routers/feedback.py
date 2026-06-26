"""
feedback.py — POST /api/feedback
Store user ratings and comments on AI responses.
Provides feedback statistics and hallucination flagging.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger
from typing import Optional

from deployment.api.database import get_db, Feedback
from deployment.api.models.schemas import (
    FeedbackCreate, FeedbackOut, FeedbackStats
)

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


# ── Submit feedback ───────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="Store a rating on an AI response",
    description=(
        "**POST /api/feedback** — Submit a rating (1–5 stars) and optional comment "
        "on any AI response from a chat session or evaluation run.\n\n"
        "Use `flagged_hallucination=true` to explicitly flag a response as incorrect. "
        "This data is used to improve evaluation metrics over time."
    ),
)
def submit_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    if payload.message_id is None and payload.run_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one of: message_id or run_id.",
        )

    fb = Feedback(
        user_id=payload.user_id,
        message_id=payload.message_id,
        run_id=payload.run_id,
        rating=payload.rating,
        comment=payload.comment,
        response_was_accurate=payload.response_was_accurate,
        response_was_helpful=payload.response_was_helpful,
        flagged_hallucination=payload.flagged_hallucination,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    logger.success(
        f"[POST /api/feedback] id={fb.id} | rating={fb.rating} "
        f"| flagged={fb.flagged_hallucination}"
    )
    return fb


# ── List all feedback ─────────────────────────────────────────────────────────
@router.get(
    "",
    response_model=list[FeedbackOut],
    summary="List all feedback entries",
)
def list_feedback(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    flagged_only: bool = Query(False, description="Only return flagged hallucination feedback"),
    min_rating: int = Query(1, ge=1, le=5),
    max_rating: int = Query(5, ge=1, le=5),
    run_id: Optional[int] = Query(None, description="Filter by evaluation run ID"),
    db: Session = Depends(get_db),
):
    query = db.query(Feedback)
    if flagged_only:
        query = query.filter(Feedback.flagged_hallucination == True)
    if run_id is not None:
        query = query.filter(Feedback.run_id == run_id)
    query = query.filter(Feedback.rating >= min_rating, Feedback.rating <= max_rating)

    results = query.order_by(Feedback.created_at.desc()).offset(skip).limit(limit).all()
    logger.info(f"[GET /api/feedback] returned {len(results)} entries")
    return results


# ── Get single feedback ───────────────────────────────────────────────────────
@router.get(
    "/{feedback_id}",
    response_model=FeedbackOut,
    summary="Get a single feedback entry by ID",
)
def get_feedback(feedback_id: int, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail=f"Feedback {feedback_id} not found.")
    return fb


# ── Aggregate statistics ──────────────────────────────────────────────────────
@router.get(
    "/stats/summary",
    response_model=FeedbackStats,
    summary="Get aggregate feedback statistics",
    description=(
        "Returns summary statistics across all feedback: average rating, "
        "hallucination flag count, and percentage of accurate/helpful responses."
    ),
)
def feedback_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(Feedback.id)).scalar() or 0
    avg_rating = db.query(func.avg(Feedback.rating)).scalar() or 0.0
    flagged = db.query(func.count(Feedback.id)).filter(
        Feedback.flagged_hallucination == True
    ).scalar() or 0

    accurate_count = db.query(func.count(Feedback.id)).filter(
        Feedback.response_was_accurate == True
    ).scalar() or 0
    helpful_count = db.query(func.count(Feedback.id)).filter(
        Feedback.response_was_helpful == True
    ).scalar() or 0

    pct_accurate = round(accurate_count / total * 100, 2) if total else None
    pct_helpful = round(helpful_count / total * 100, 2) if total else None

    logger.info(f"[GET /api/feedback/stats] total={total} | avg_rating={avg_rating:.2f}")

    return FeedbackStats(
        total_feedback=total,
        avg_rating=round(float(avg_rating), 2),
        flagged_hallucinations=flagged,
        pct_accurate=pct_accurate,
        pct_helpful=pct_helpful,
    )
