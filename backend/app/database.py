from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    quality_score: Mapped[float] = mapped_column()
    quality_label: Mapped[str] = mapped_column(String(32))
    context: Mapped[str | None] = mapped_column(String(32), nullable=True)
    original_path: Mapped[str] = mapped_column(Text)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    guest_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


def _ensure_sqlite_parent_dir(url: str) -> None:
    parsed = make_url(url)
    if not parsed.drivername.startswith("sqlite") or not parsed.database or parsed.database == ":memory:":
        return
    Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent_dir(settings.database_url)
engine = create_engine(settings.database_url, **_engine_kwargs(settings.database_url))
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _sqlite_add_columns() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(analyses)").fetchall()}
        if cols and "user_id" not in cols:
            conn.exec_driver_sql("ALTER TABLE analyses ADD COLUMN user_id VARCHAR(36)")
            conn.exec_driver_sql("ALTER TABLE analyses ADD COLUMN guest_session_id VARCHAR(36)")


def init_db() -> None:
    _ensure_sqlite_parent_dir(settings.database_url)
    Base.metadata.create_all(bind=engine)
    _sqlite_add_columns()
