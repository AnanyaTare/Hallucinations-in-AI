"""
main.py — FastAPI Application Entry Point (Server Layer)

Architecture:
  Client Layer  →  FastAPI REST API (this file)
  Server Layer  →  Routers (evaluate / analyze / mitigate / results)
  AI Model Layer→  Services (llm_service / evaluation_service / mitigation_service)
  Database Layer→  SQLAlchemy + SQLite (database.py)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
import time

from deployment.api.config import get_settings
from deployment.api.database import create_tables
# Task 1 — Evaluation routers
from deployment.api.routers import evaluate, analyze, mitigate, results
# Task 2 — New API endpoint routers
from deployment.api.routers import chat, history, users, feedback, api_health

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Hallucinations-in-AI API...")
    logger.info(f"   Default model : {settings.DEFAULT_MODEL}")
    logger.info(f"   Database URL  : {settings.DATABASE_URL}")
    create_tables()
    logger.success("✅ Database tables created / verified.")
    yield
    # Shutdown
    logger.info("🛑 Shutting down API.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "**U2UInnovate Project 22** — REST API for detecting, evaluating, and mitigating "
        "hallucinations in Large Language Models.\n\n"
        "### Architecture Layers\n"
        "| Layer | Implementation |\n"
        "|---|---|\n"
        "| **Client** | Any HTTP client, curl, Streamlit frontend |\n"
        "| **Server** | FastAPI + Uvicorn |\n"
        "| **AI Model** | LLM APIs (OpenAI/Gemini) + BERTScore + NLI |\n"
        "| **Database** | SQLite via SQLAlchemy |\n"
    ),
    contact={"name": "AI Safety Lab", "email": "mentor@aisafetylab.ai"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── CORS (Client Layer access) ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


# ── Routers — Task 1 (Evaluation) ────────────────────────────────────────────
app.include_router(evaluate.router, prefix="/api/v1")
app.include_router(analyze.router,  prefix="/api/v1")
app.include_router(mitigate.router, prefix="/api/v1")
app.include_router(results.router,  prefix="/api/v1")

# ── Routers — Task 2 (Core AI endpoints) ─────────────────────────────────────
app.include_router(chat.router)        # POST /api/chat
app.include_router(history.router)     # GET  /api/history
app.include_router(users.router)       # GET  /api/users
app.include_router(feedback.router)    # POST /api/feedback
app.include_router(api_health.router)  # GET  /api/health


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"], summary="API health check")
async def health_check():
    from deployment.api.models.schemas import HealthResponse
    from deployment.api.database import SessionLocal

    db_ok = False
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        pass

    model_ok = bool(settings.OPENAI_API_KEY or settings.GEMINI_API_KEY)

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version=settings.APP_VERSION,
        model_available=model_ok,
        database_connected=db_ok,
    )


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "message": "Hallucinations-in-AI API is running 🧠",
        "docs": "/docs",
        "version": settings.APP_VERSION,
    }
