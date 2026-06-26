"""
schemas.py — Pydantic request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


# ── Shared Enums ──────────────────────────────────────────────────────────────
class MitigationTechniqueEnum(str, Enum):
    NONE = "none"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RAG = "rag"
    SELF_CONSISTENCY = "self_consistency"


class ModelEnum(str, Enum):
    GPT4O = "gpt-4o"
    GPT35 = "gpt-3.5-turbo"
    GEMINI = "gemini/gemini-1.5-pro"
    MISTRAL = "mistral/mistral-7b-instruct"


# ── Evaluate ──────────────────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    prompt: str = Field(..., description="The question/prompt to send to the LLM", min_length=1)
    ground_truth: Optional[str] = Field(None, description="Expected factual answer for scoring")
    context: Optional[str] = Field(None, description="Retrieved context (for RAG evaluation)")
    model: ModelEnum = Field(ModelEnum.GPT4O, description="LLM to use")
    mitigation: MitigationTechniqueEnum = Field(
        MitigationTechniqueEnum.NONE, description="Mitigation technique to apply"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What year was the Eiffel Tower built?",
                "ground_truth": "The Eiffel Tower was built in 1887–1889.",
                "model": "gpt-4o",
                "mitigation": "none",
            }
        }


class EvaluationScores(BaseModel):
    hallucination_score: float = Field(..., description="0=faithful, 1=hallucinated")
    bert_score: Optional[float] = Field(None, description="BERTScore F1")
    rouge_l: Optional[float] = Field(None, description="ROUGE-L F1")
    faithfulness_score: Optional[float] = Field(None, description="RAGAS faithfulness (0–1)")
    answer_relevancy: Optional[float] = Field(None, description="RAGAS answer relevancy (0–1)")


class EvaluateResponse(BaseModel):
    run_id: int
    prompt: str
    llm_response: str
    model: str
    mitigation: str
    is_hallucinated: bool
    hallucination_type: str
    scores: EvaluationScores
    retrieved_context: Optional[List[str]] = None
    created_at: datetime


# ── Analyze ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="LLM-generated text to analyze for hallucination patterns")
    reference: Optional[str] = Field(None, description="Reference or ground-truth text")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The Eiffel Tower was built in 1650 by Napoleon Bonaparte.",
                "reference": "The Eiffel Tower was built in 1887–1889 by Gustave Eiffel.",
            }
        }


class AnalyzeResponse(BaseModel):
    hallucination_type: str
    confidence: float
    explanation: str
    semantic_similarity: float
    contradictions_detected: List[str]


# ── Mitigate ──────────────────────────────────────────────────────────────────
class MitigateRequest(BaseModel):
    prompt: str = Field(..., description="Original prompt that produced hallucinations")
    model: ModelEnum = Field(ModelEnum.GPT4O)
    technique: MitigationTechniqueEnum = Field(
        MitigationTechniqueEnum.RAG, description="Mitigation technique to apply"
    )
    knowledge_base_query: Optional[str] = Field(
        None, description="Override query for RAG retrieval"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "When was the Eiffel Tower built?",
                "model": "gpt-4o",
                "technique": "rag",
            }
        }


class MitigateResponse(BaseModel):
    original_response: str
    mitigated_response: str
    technique_used: str
    improvement_score: float
    retrieved_docs: Optional[List[str]] = None
    consistency_votes: Optional[dict] = None


# ── Results ───────────────────────────────────────────────────────────────────
class RunSummary(BaseModel):
    id: int
    prompt: str
    model_name: str
    mitigation: str
    is_hallucinated: bool
    hallucination_score: Optional[float]
    hallucination_type: str
    created_at: datetime

    class Config:
        from_attributes = True


class BenchmarkSummary(BaseModel):
    model_name: str
    mitigation: str
    dataset_name: str
    total_samples: int
    hallucination_rate: Optional[float]
    avg_bert_score: Optional[float]
    avg_faithfulness: Optional[float]
    avg_answer_relevancy: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class ComparisonResponse(BaseModel):
    results: List[BenchmarkSummary]
    best_technique: str
    lowest_hallucination_rate: float


# ── Health (basic — used by /health) ──────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    version: str
    model_available: bool
    database_connected: bool


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — New endpoint schemas: /api/chat, /api/history, /api/users,
#           /api/feedback, /api/health
# ══════════════════════════════════════════════════════════════════════════════

# ── /api/chat ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Send a prompt to the AI. Optionally attach a session for conversation context."""
    prompt: str = Field(..., min_length=1, description="User's message / question")
    session_id: Optional[int] = Field(None, description="Existing chat session ID to continue")
    user_id: Optional[int] = Field(None, description="ID of the user sending the message")
    model: ModelEnum = Field(ModelEnum.GPT4O, description="LLM to use")
    mitigation: MitigationTechniqueEnum = Field(
        MitigationTechniqueEnum.NONE, description="Mitigation technique to apply"
    )
    evaluate_hallucination: bool = Field(
        True, description="Whether to score the response for hallucinations"
    )
    ground_truth: Optional[str] = Field(None, description="Optional ground truth for scoring")

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Did the Eiffel Tower exist in the 17th century?",
                "model": "gpt-4o",
                "mitigation": "none",
                "evaluate_hallucination": True,
            }
        }


class ChatMessageOut(BaseModel):
    """A single message turn in a conversation."""
    id: int
    session_id: int
    role: str                          # "user" | "assistant"
    content: str
    hallucination_score: Optional[float] = None
    is_hallucinated: Optional[bool] = None
    hallucination_type: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    """Full response from POST /api/chat."""
    session_id: int
    user_message_id: int
    assistant_message_id: int
    prompt: str
    ai_response: str
    model: str
    mitigation: str
    is_hallucinated: bool
    hallucination_score: float
    hallucination_type: str
    explanation: Optional[str] = None
    created_at: datetime


# ── /api/history ──────────────────────────────────────────────────────────────
class ChatSessionOut(BaseModel):
    """A conversation session with all messages."""
    id: int
    session_title: Optional[str]
    user_id: Optional[int]
    model_name: str
    mitigation: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessageOut] = []

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """Response from GET /api/history."""
    total_sessions: int
    sessions: List[ChatSessionOut]


# ── /api/users ────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=80)
    email: str = Field(..., description="Valid email address")
    full_name: Optional[str] = Field(None, max_length=150)
    role: str = Field("researcher", description="researcher | admin")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "sabeena_researcher",
                "email": "sabeena@aisafety.ai",
                "full_name": "Sabeena V",
                "role": "researcher",
            }
        }


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    last_seen: Optional[datetime]

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UsersListResponse(BaseModel):
    total: int
    users: List[UserOut]


# ── /api/feedback ─────────────────────────────────────────────────────────────
class FeedbackCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (bad) to 5 (excellent)")
    comment: Optional[str] = Field(None, description="Free-text feedback")
    user_id: Optional[int] = Field(None, description="ID of the user giving feedback")
    message_id: Optional[int] = Field(None, description="Chat message being rated")
    run_id: Optional[int] = Field(None, description="Evaluation run being rated")
    response_was_accurate: Optional[bool] = Field(None, description="Was the AI response factually accurate?")
    response_was_helpful: Optional[bool] = Field(None, description="Was the response helpful?")
    flagged_hallucination: bool = Field(False, description="Flag this response as a hallucination")

    class Config:
        json_schema_extra = {
            "example": {
                "rating": 4,
                "comment": "The response was mostly accurate but contained one wrong date.",
                "run_id": 1,
                "response_was_accurate": False,
                "response_was_helpful": True,
                "flagged_hallucination": True,
            }
        }


class FeedbackOut(BaseModel):
    id: int
    rating: int
    comment: Optional[str]
    user_id: Optional[int]
    message_id: Optional[int]
    run_id: Optional[int]
    response_was_accurate: Optional[bool]
    response_was_helpful: Optional[bool]
    flagged_hallucination: bool
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackStats(BaseModel):
    total_feedback: int
    avg_rating: float
    flagged_hallucinations: int
    pct_accurate: Optional[float]
    pct_helpful: Optional[float]


# ── /api/health (detailed) ────────────────────────────────────────────────────
class DetailedHealthResponse(BaseModel):
    status: str                            # healthy | degraded | down
    version: str
    uptime_seconds: float
    database_connected: bool
    model_available: bool
    active_users: int
    total_chat_sessions: int
    total_evaluations: int
    total_feedback: int
    supported_models: List[str]
    supported_mitigations: List[str]
