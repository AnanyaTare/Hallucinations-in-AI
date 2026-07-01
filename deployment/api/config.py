"""
config.py — Centralized app configuration via .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "Hallucinations-in-AI API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── LLM API Keys ─────────────────────────────────────
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    DEFAULT_MODEL: str = "groq/llama3-8b-8192"

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./hallucinations.db"

    # ── Evaluation ───────────────────────────────────────
    HALLUCINATION_THRESHOLD: float = 0.5
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.0

    # ── RAG ──────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "data/processed/chroma_db"
    TOP_K_RETRIEVAL: int = 5

    # ── Self-Consistency ─────────────────────────────────
    SELF_CONSISTENCY_N: int = 5


@lru_cache()
def get_settings() -> Settings:
    return Settings()
