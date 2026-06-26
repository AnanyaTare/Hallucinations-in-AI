"""
results.py — GET /results
Query stored evaluation runs and benchmark summaries from the database.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger
from typing import List, Optional

from deployment.api.database import get_db, EvaluationRun, BenchmarkResult
from deployment.api.models.schemas import RunSummary, BenchmarkSummary, ComparisonResponse

router = APIRouter(prefix="/results", tags=["Results"])


@router.get(
    "/runs",
    response_model=List[RunSummary],
    summary="List evaluation runs",
    description="Retrieve a paginated list of all evaluation runs stored in the database.",
)
def get_runs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max records to return"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    hallucinated_only: bool = Query(False, description="Return only hallucinated runs"),
    db: Session = Depends(get_db),
):
    query = db.query(EvaluationRun)
    if model:
        query = query.filter(EvaluationRun.model_name == model)
    if hallucinated_only:
        query = query.filter(EvaluationRun.is_hallucinated == 1)

    runs = query.order_by(EvaluationRun.created_at.desc()).offset(skip).limit(limit).all()
    logger.info(f"[GET /results/runs] returned {len(runs)} records")
    return runs


@router.get(
    "/runs/{run_id}",
    response_model=RunSummary,
    summary="Get a single evaluation run by ID",
)
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@router.get(
    "/stats",
    summary="Get aggregate statistics across all runs",
)
def get_stats(db: Session = Depends(get_db)):
    total = db.query(func.count(EvaluationRun.id)).scalar()
    hallucinated = db.query(func.count(EvaluationRun.id)).filter(
        EvaluationRun.is_hallucinated == 1
    ).scalar()
    avg_score = db.query(func.avg(EvaluationRun.hallucination_score)).scalar()
    avg_bert = db.query(func.avg(EvaluationRun.bert_score)).scalar()

    hallucination_rate = round(hallucinated / total * 100, 2) if total else 0.0

    # Per-technique breakdown
    technique_stats = (
        db.query(
            EvaluationRun.mitigation,
            func.count(EvaluationRun.id).label("count"),
            func.avg(EvaluationRun.hallucination_score).label("avg_score"),
        )
        .group_by(EvaluationRun.mitigation)
        .all()
    )

    return {
        "total_runs": total,
        "hallucinated_runs": hallucinated,
        "hallucination_rate_pct": hallucination_rate,
        "avg_hallucination_score": round(avg_score or 0.0, 4),
        "avg_bert_score": round(avg_bert or 0.0, 4),
        "per_technique": [
            {
                "mitigation": row.mitigation.value if hasattr(row.mitigation, "value") else str(row.mitigation),
                "count": row.count,
                "avg_hallucination_score": round(row.avg_score or 0.0, 4),
            }
            for row in technique_stats
        ],
    }


@router.get(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare mitigation techniques via benchmark results",
)
def compare_techniques(db: Session = Depends(get_db)):
    benchmarks = db.query(BenchmarkResult).all()

    if not benchmarks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark results found. Run a benchmark first.",
        )

    # Find best technique (lowest hallucination rate)
    best = min(
        benchmarks,
        key=lambda b: b.hallucination_rate if b.hallucination_rate is not None else 1.0,
    )

    return ComparisonResponse(
        results=benchmarks,
        best_technique=best.mitigation,
        lowest_hallucination_rate=best.hallucination_rate or 0.0,
    )
