"""
conversation_history.py — Conversation History Module (Task 5)

Objective : Store previous interactions and retrieve chat history.
Deliverable: A self-contained history module that saves {question, answer}
             pairs to both an in-memory list AND the SQLite database via the API.

Example record format:
    {
        "question": "Explain AI",
        "answer":   "Artificial Intelligence ..."
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional
from loguru import logger


# ── In-Memory Store (array) ───────────────────────────────────────────────────
# Satisfies the task activity: "Create an array to save prompts and responses"
_conversation_history: list[dict] = []


def add_to_history(
    question: str,
    answer: str,
    session_id: Optional[int] = None,
    model: Optional[str] = None,
    is_hallucinated: Optional[bool] = None,
    hallucination_score: Optional[float] = None,
) -> dict:
    """
    Save one interaction to the in-memory history array.

    Returns the saved record so callers can confirm what was stored.
    """
    record = {
        "id":                 len(_conversation_history) + 1,
        "question":           question,
        "answer":             answer,
        "session_id":         session_id,
        "model":              model,
        "is_hallucinated":    is_hallucinated,
        "hallucination_score": hallucination_score,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }
    _conversation_history.append(record)
    logger.info(
        f"[ConversationHistory] Saved record #{record['id']} | "
        f"session={session_id} | hallucinated={is_hallucinated}"
    )
    return record


def get_all_history() -> list[dict]:
    """Return the complete in-memory conversation history."""
    return list(_conversation_history)


def get_history_by_session(session_id: int) -> list[dict]:
    """Return all interactions for a specific session ID."""
    return [r for r in _conversation_history if r.get("session_id") == session_id]


def get_last_n(n: int = 5) -> list[dict]:
    """Return the most recent N interactions."""
    return _conversation_history[-n:]


def clear_history() -> None:
    """Wipe the in-memory history (does NOT delete from database)."""
    _conversation_history.clear()
    logger.info("[ConversationHistory] In-memory history cleared.")


def search_history(keyword: str) -> list[dict]:
    """
    Simple keyword search over questions and answers.
    Case-insensitive substring match.
    """
    kw = keyword.lower()
    return [
        r for r in _conversation_history
        if kw in r["question"].lower() or kw in r["answer"].lower()
    ]


def export_to_json(filepath: str) -> str:
    """
    Export all history to a JSON file.
    Returns the path to the written file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_conversation_history, f, indent=2, ensure_ascii=False)
    logger.info(f"[ConversationHistory] Exported {len(_conversation_history)} records → {filepath}")
    return filepath


def history_summary() -> dict:
    """
    Return a quick summary of the current history:
      - total interactions
      - number of hallucinated responses
      - hallucination rate
    """
    total = len(_conversation_history)
    hallucinated = sum(
        1 for r in _conversation_history
        if r.get("is_hallucinated") is True
    )
    return {
        "total_interactions":  total,
        "hallucinated_count":  hallucinated,
        "faithful_count":      total - hallucinated,
        "hallucination_rate":  round(hallucinated / total, 4) if total > 0 else 0.0,
    }
