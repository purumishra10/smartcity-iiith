from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import HTTPException, Response
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Analysis, User

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def make_token(user: User) -> str:
    payload = {
        "sub": user.id,
        "email": user.email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expire_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        settings.auth_cookie,
        token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_hours * 3600,
        path="/",
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(settings.auth_cookie, path="/")


def current_user(db: Session, token: str | None) -> User | None:
    if not token:
        return None
    data = decode_token(token)
    if not data or "sub" not in data:
        return None
    return db.get(User, data["sub"])


def require_user(user: User | None) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="login_required")
    return user


def validate_credentials(email: str, password: str) -> tuple[str, str]:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="invalid_email")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="password_too_short")
    return email, password


def migrate_guest_exams(db: Session, user: User, guest_id: str | None) -> int:
    if not guest_id:
        return 0
    rows = db.query(Analysis).filter(Analysis.guest_session_id == guest_id).all()
    n = 0
    for row in rows:
        row.user_id = user.id
        row.guest_session_id = None
        n += 1
    if n:
        db.commit()
    return n


def can_access(row: Analysis, user: User | None, guest_id: str | None) -> bool:
    if user and row.user_id == user.id:
        return True
    if guest_id and row.guest_session_id == guest_id and row.user_id is None:
        return True
    return False


def new_user(email: str, password: str) -> User:
    return User(id=str(uuid.uuid4()), email=email, password_hash=hash_password(password))
