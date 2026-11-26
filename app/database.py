"""SQLAlchemy 2.x engine, session factory, and SQLite WAL initialization."""

from collections.abc import Generator
from typing import Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base declarative class for all ORM models."""


def create_db_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with SQLite WAL/FK pragmas or serverless Postgres pooling."""
    normalized_url = (
        database_url.replace("postgres://", "postgresql://", 1)
        if database_url.startswith("postgres://")
        else database_url
    )
    is_sqlite = normalized_url.startswith("sqlite")
    engine_kwargs: dict[str, object] = {
        "pool_pre_ping": True,
    }
    if is_sqlite:
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Prevent stale idle connections on serverless Postgres (Neon/Supabase) after cold starts or cron gaps
        engine_kwargs["pool_recycle"] = 300

    db_engine = create_engine(normalized_url, **engine_kwargs)

    if db_engine.dialect.name == "sqlite":

        @event.listens_for(db_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return db_engine


engine = create_db_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def migrate_sqlite_schema(target_engine: Engine) -> None:
    """Idempotently add Phase 5 partial scale-out columns to existing SQLite trades tables."""
    if target_engine.dialect.name != "sqlite":
        return

    expected_columns: dict[str, str] = {
        "initial_qty": "FLOAT",
        "partial_tp_taken": "BOOLEAN DEFAULT 0",
        "realized_partial_pnl_usd": "FLOAT DEFAULT 0.0",
    }

    with target_engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(trades)")).fetchall()
        if not rows:
            return
        existing_cols = {str(row[1]) for row in rows}
        for col_name, col_def in expected_columns.items():
            if col_name not in existing_cols:
                conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}"))


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all database tables if they do not exist and run idempotent SQLite migrations."""
    import app.models  # noqa: F401 - Ensure ORM models are registered on Base.metadata

    resolved_engine = target_engine or engine
    Base.metadata.create_all(bind=resolved_engine)
    migrate_sqlite_schema(resolved_engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
