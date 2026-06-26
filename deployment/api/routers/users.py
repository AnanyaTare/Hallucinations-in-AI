"""
users.py — GET/POST /api/users
User management: register researchers, fetch user info, update profiles.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from loguru import logger
from typing import Optional
from datetime import datetime, timezone

from deployment.api.database import get_db, User
from deployment.api.models.schemas import (
    UserCreate, UserOut, UserUpdate, UsersListResponse
)

router = APIRouter(prefix="/api/users", tags=["Users"])


# ── List all users ────────────────────────────────────────────────────────────
@router.get(
    "",
    response_model=UsersListResponse,
    summary="Fetch user information",
    description=(
        "**GET /api/users** — Retrieve a list of all registered researchers/users.\n\n"
        "Supports filtering by role, active status, and pagination."
    ),
)
def get_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    role: Optional[str] = Query(None, description="Filter by role (researcher | admin)"),
    active_only: bool = Query(True, description="Only return active users"),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if active_only:
        query = query.filter(User.is_active == True)
    if role:
        query = query.filter(User.role == role)

    total = query.count()
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
    logger.info(f"[GET /api/users] total={total} | returned={len(users)}")
    return UsersListResponse(total=total, users=users)


# ── Get single user ───────────────────────────────────────────────────────────
@router.get(
    "/{user_id}",
    response_model=UserOut,
    summary="Get a user by ID",
)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    return user


# ── Register new user ─────────────────────────────────────────────────────────
@router.post(
    "",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user / researcher",
    description=(
        "**POST /api/users** — Register a new researcher on the platform.\n\n"
        "Returns the created user object. Username and email must be unique."
    ),
)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    # Check uniqueness
    existing = (
        db.query(User)
        .filter((User.username == payload.username) | (User.email == payload.email))
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this username or email already exists.",
        )

    user = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        role=payload.role,
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.success(f"[POST /api/users] Created user id={user.id} username={user.username}")
    return user


# ── Update user ───────────────────────────────────────────────────────────────
@router.patch(
    "/{user_id}",
    response_model=UserOut,
    summary="Update user profile",
)
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")

    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.role is not None:
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active

    db.commit()
    db.refresh(user)
    logger.info(f"[PATCH /api/users/{user_id}] Updated")
    return user


# ── Update last_seen ──────────────────────────────────────────────────────────
@router.post(
    "/{user_id}/ping",
    response_model=UserOut,
    summary="Update user's last_seen timestamp",
)
def ping_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    user.last_seen = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


# ── Delete user ───────────────────────────────────────────────────────────────
@router.delete("/{user_id}", status_code=204, summary="Delete a user")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    db.delete(user)
    db.commit()
    logger.info(f"[DELETE /api/users/{user_id}] deleted")
