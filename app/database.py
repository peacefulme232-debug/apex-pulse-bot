"""SQLAlchemy 2.x engine, session factory, and SQLite WAL initialization."""

from collections.abc import Generator
from typing import Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base declarative class for all ORM models."""


def create_db_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with SQLite WAL/FK pragmas when using SQLite."""
    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    db_engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    if is_sqlite:

        @event.listens_for(db_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return db_engine


engine = create_db_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all database tables if they do not exist."""
    import app.models  # noqa: F401 - Ensure ORM models are registered on Base.metadata

    Base.metadata.create_all(bind=target_engine or engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
