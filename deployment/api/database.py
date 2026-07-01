"""
database.py — Database Layer
SQLAlchemy + SQLite setup for storing evaluation runs, chat history,
users, feedback, and benchmark results.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, Enum as SAEnum, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
import enum

from deployment.api.config import get_settings

settings = get_settings()

# ── Engine & Session ──────────────────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Enums ─────────────────────────────────────────────────────────────────────
class MitigationTechnique(str, enum.Enum):
    NONE = "none"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    RAG = "rag"
    SELF_CONSISTENCY = "self_consistency"


class HallucinationType(str, enum.Enum):
    FACTUAL = "factual"
    FAITHFULNESS = "faithfulness"
    INTRINSIC = "intrinsic"
    NONE_DETECTED = "none_detected"
    UNKNOWN = "unknown"


# ── ORM Models ────────────────────────────────────────────────────────────────
class EvaluationRun(Base):
    """Stores one complete evaluation run (prompt → response → scores)."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    context = Column(Text, nullable=True)          # optional RAG context
    llm_response = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)

    # Model & technique used
    model_name = Column(String(100), nullable=False)
    mitigation = Column(
        SAEnum(MitigationTechnique),
        default=MitigationTechnique.NONE,
        nullable=False,
    )

    # Evaluation scores
    hallucination_score = Column(Float, nullable=True)   # 0=faithful, 1=hallucinated
    bert_score = Column(Float, nullable=True)
    rouge_l = Column(Float, nullable=True)
    faithfulness_score = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)

    # Classification
    hallucination_type = Column(
        SAEnum(HallucinationType),
        default=HallucinationType.UNKNOWN,
        nullable=False,
    )
    is_hallucinated = Column(Integer, default=0)  # 0=no, 1=yes

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    notes = Column(Text, nullable=True)


class BenchmarkResult(Base):
    """Aggregate benchmark stats per technique+model combination."""
    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    mitigation = Column(String(50), nullable=False)
    dataset_name = Column(String(100), nullable=False)

    total_samples = Column(Integer, default=0)
    hallucination_rate = Column(Float, nullable=True)
    avg_bert_score = Column(Float, nullable=True)
    avg_faithfulness = Column(Float, nullable=True)
    avg_answer_relevancy = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── User ─────────────────────────────────────────────────────────────────────
class User(Base):
    """Registered users / researchers interacting with the API."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False)
    full_name = Column(String(150), nullable=True)
    role = Column(String(30), default="researcher")  # researcher | admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, nullable=True)

    # Relationships
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete-orphan")


# ── Chat / Conversation History ───────────────────────────────────────────────
class ChatSession(Base):
    """A conversation session (groups multiple messages)."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_title = Column(String(200), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_name = Column(String(100), nullable=False, default="gpt-4o")
    mitigation = Column(String(50), default="none")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan",
                            order_by="ChatMessage.id")


class ChatMessage(Base):
    """A single turn in a chat session (user prompt + AI reply + scores)."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    role = Column(String(20), nullable=False)       # "user" | "assistant"
    content = Column(Text, nullable=False)           # message text

    # Hallucination metadata (populated for assistant messages)
    hallucination_score = Column(Float, nullable=True)
    is_hallucinated = Column(Boolean, default=False)
    hallucination_type = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")


# ── Feedback ──────────────────────────────────────────────────────────────────
class Feedback(Base):
    """User ratings and comments on AI responses."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=True)

    rating = Column(Integer, nullable=False)         # 1–5 stars
    comment = Column(Text, nullable=True)
    response_was_accurate = Column(Boolean, nullable=True)
    response_was_helpful = Column(Boolean, nullable=True)
    flagged_hallucination = Column(Boolean, default=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="feedbacks")


# ── Helpers ───────────────────────────────────────────────────────────────────
def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
