"""
api_health.py — GET /api/health
Detailed health check endpoint exposing system status, DB connectivity,
model availability, and live counters.
"""

import time
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from loguru import logger

from deployment.api.database import (
    get_db, User, ChatSession, EvaluationRun, Feedback
)
from deployment.api.models.schemas import DetailedHealthResponse
from deployment.api.config import get_settings

router = APIRouter(prefix="/api/health", tags=["Health"])

# Record server start time for uptime calculation
_SERVER_START = time.time()

SUPPORTED_MODELS = ["gpt-4o", "gpt-3.5-turbo", "gemini/gemini-1.5-pro", "mistral/mistral-7b-instruct"]
SUPPORTED_MITIGATIONS = ["none", "few_shot", "chain_of_thought", "rag", "self_consistency"]


@router.get(
    "",
    response_model=DetailedHealthResponse,
    summary="Detailed system health check",
    description=(
        "**GET /api/health** — Returns a comprehensive system health report.\n\n"
        "Includes:\n"
        "- Database connectivity\n"
        "- LLM API key availability\n"
        "- Live counters (users, sessions, evaluations, feedback)\n"
        "- Supported models and mitigation techniques\n"
        "- Server uptime in seconds\n\n"
        "Use `status=healthy` as your deployment health gate."
    ),
)
def detailed_health(db: Session = Depends(get_db)):
    settings = get_settings()
    uptime = round(time.time() - _SERVER_START, 1)

    # DB connectivity
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        logger.error(f"Health check DB error: {e}")

    # Live counters
    active_users = 0
    total_sessions = 0
    total_evals = 0
    total_fb = 0

    if db_ok:
        try:
            active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar() or 0
            total_sessions = db.query(func.count(ChatSession.id)).scalar() or 0
            total_evals = db.query(func.count(EvaluationRun.id)).scalar() or 0
            total_fb = db.query(func.count(Feedback.id)).scalar() or 0
        except Exception as e:
            logger.warning(f"Counter query failed: {e}")

    # Model key availability
    model_ok = bool(settings.OPENAI_API_KEY or settings.GEMINI_API_KEY)

    overall_status = "healthy" if (db_ok and model_ok) else ("degraded" if db_ok else "down")

    logger.info(
        f"[GET /api/health] status={overall_status} | uptime={uptime}s "
        f"| users={active_users} | sessions={total_sessions}"
    )

    return DetailedHealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        uptime_seconds=uptime,
        database_connected=db_ok,
        model_available=model_ok,
        active_users=active_users,
        total_chat_sessions=total_sessions,
        total_evaluations=total_evals,
        total_feedback=total_fb,
        supported_models=SUPPORTED_MODELS,
        supported_mitigations=SUPPORTED_MITIGATIONS,
    )
