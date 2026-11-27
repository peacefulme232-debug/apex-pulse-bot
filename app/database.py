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


from decimal import Decimal, ROUND_HALF_UP

DEFAULT_INITIAL_BYBIT_BALANCE = Decimal("10000.000000")
DEFAULT_INITIAL_FOREX_BALANCE = Decimal("10000.000000")

_TRADES_MIGRATION_COLUMNS: list[tuple[str, str, str]] = [
    # (column_name, sqlite_type_def, postgres_type_def)
    ("initial_qty", "FLOAT", "DOUBLE PRECISION"),
    ("partial_tp_taken", "BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE"),
    ("realized_partial_pnl_usd", "FLOAT DEFAULT 0.0", "DOUBLE PRECISION DEFAULT 0.0"),
    ("order_entry_type", "VARCHAR(16) DEFAULT 'MARKET'", "VARCHAR(16) DEFAULT 'MARKET'"),
    ("order_exit_type", "VARCHAR(16) DEFAULT 'MARKET'", "VARCHAR(16) DEFAULT 'MARKET'"),
    ("requested_entry_price", "NUMERIC(18, 8)", "NUMERIC(18, 8)"),
    ("filled_entry_price", "NUMERIC(18, 8)", "NUMERIC(18, 8)"),
    ("requested_exit_price", "NUMERIC(18, 8)", "NUMERIC(18, 8)"),
    ("filled_exit_price", "NUMERIC(18, 8)", "NUMERIC(18, 8)"),
    ("initial_margin", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("maintenance_margin", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("liquidation_price", "NUMERIC(18, 8)", "NUMERIC(18, 8)"),
    ("gross_pnl", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("entry_fee", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("exit_fee", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("spread_cost", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("slippage_cost", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("funding_or_swap_fee", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("net_pnl", "NUMERIC(18, 6) DEFAULT 0.000000", "NUMERIC(18, 6) DEFAULT 0.000000"),
    ("breakeven_applied", "BOOLEAN DEFAULT 0", "BOOLEAN DEFAULT FALSE"),
    ("last_funding_applied_at", "DATETIME", "TIMESTAMP WITH TIME ZONE"),
    ("last_swap_applied_date", "DATE", "DATE"),
]

_WALLET_ACCOUNTS_MIGRATION_COLUMNS: list[tuple[str, str, str]] = [
    ("consecutive_losses", "INTEGER NOT NULL DEFAULT 0", "INTEGER NOT NULL DEFAULT 0"),
    ("circuit_breaker_until", "DATETIME", "TIMESTAMP WITH TIME ZONE"),
]

_EXECUTION_REJECTIONS_MIGRATION_COLUMNS: list[tuple[str, str, str]] = [
    ("strategy", "VARCHAR(48)", "VARCHAR(48)"),
    ("confluence_score", "NUMERIC(6, 2)", "NUMERIC(6, 2)"),
]

_TABLE_MIGRATION_COLUMNS: dict[str, list[tuple[str, str, str]]] = {
    "trades": _TRADES_MIGRATION_COLUMNS,
    "wallet_accounts": _WALLET_ACCOUNTS_MIGRATION_COLUMNS,
    "execution_rejections": _EXECUTION_REJECTIONS_MIGRATION_COLUMNS,
}


def _q6(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def _migrate_execution_rejections_defaults(conn) -> None:
    """Idempotently rebuild legacy SQLite execution_rejections if qty/margin columns are NOT NULL without default."""
    rows = conn.execute(text("PRAGMA table_info(execution_rejections)")).fetchall()
    if not rows:
        return
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    guarded = {"requested_qty", "required_min_qty", "required_margin", "available_free_margin"}
    needs_rebuild = any(
        str(row[1]) in guarded and int(row[3]) == 1 and row[4] is None
        for row in rows
    )
    if not needs_rebuild:
        return

    existing_cols = {str(row[1]) for row in rows}
    conn.execute(
        text(
            """
            CREATE TABLE execution_rejections__migrated (
                id INTEGER NOT NULL PRIMARY KEY,
                market VARCHAR(16) NOT NULL,
                symbol VARCHAR(32) NOT NULL,
                side VARCHAR(16) NOT NULL,
                strategy VARCHAR(48),
                confluence_score NUMERIC(6, 2),
                reason_code VARCHAR(48) NOT NULL,
                requested_qty NUMERIC(18, 8) DEFAULT 0,
                required_min_qty NUMERIC(18, 8) DEFAULT 0,
                required_margin NUMERIC(18, 6) DEFAULT 0,
                available_free_margin NUMERIC(18, 6) DEFAULT 0,
                details TEXT NOT NULL DEFAULT '',
                created_at DATETIME NOT NULL
            )
            """
        )
    )
    strategy_expr = "strategy" if "strategy" in existing_cols else "NULL"
    score_expr = "confluence_score" if "confluence_score" in existing_cols else "NULL"
    conn.execute(
        text(
            f"""
            INSERT INTO execution_rejections__migrated (
                id, market, symbol, side, strategy, confluence_score,
                reason_code, requested_qty, required_min_qty,
                required_margin, available_free_margin, details, created_at
            )
            SELECT
                id, market, symbol, side, {strategy_expr}, {score_expr},
                reason_code,
                COALESCE(requested_qty, 0),
                COALESCE(required_min_qty, 0),
                COALESCE(required_margin, 0),
                COALESCE(available_free_margin, 0),
                COALESCE(details, ''),
                created_at
            FROM execution_rejections
            """
        )
    )
    conn.execute(text("DROP TABLE execution_rejections"))
    conn.execute(text("ALTER TABLE execution_rejections__migrated RENAME TO execution_rejections"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_execution_rejections_id ON execution_rejections (id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_execution_rejections_market ON execution_rejections (market)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_execution_rejections_symbol ON execution_rejections (symbol)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_execution_rejections_reason_code ON execution_rejections (reason_code)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_execution_rejections_created_at ON execution_rejections (created_at)"))


def migrate_sqlite_schema(target_engine: Engine) -> None:
    """Idempotently add missing columns to existing SQLite tables."""
    if target_engine.dialect.name != "sqlite":
        return

    with target_engine.begin() as conn:
        for table_name, columns in _TABLE_MIGRATION_COLUMNS.items():
            rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            if not rows:
                continue
            existing_cols = {str(row[1]) for row in rows}
            for col_name, sqlite_def, _pg_def in columns:
                if col_name not in existing_cols:
                    conn.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {sqlite_def}")
                    )
        _migrate_execution_rejections_defaults(conn)


def migrate_schema(target_engine: Engine) -> None:
    """Idempotently add missing columns to existing tables on both PostgreSQL and SQLite."""
    if target_engine.dialect.name == "sqlite":
        migrate_sqlite_schema(target_engine)
        return

    if target_engine.dialect.name == "postgresql":
        with target_engine.begin() as conn:
            for table_name, columns in _TABLE_MIGRATION_COLUMNS.items():
                exists = conn.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name = :table_name"
                    ),
                    {"table_name": table_name},
                ).fetchone()
                if not exists:
                    continue
                for col_name, _sqlite_def, pg_def in columns:
                    conn.execute(
                        text(
                            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {pg_def}"
                        )
                    )


def seed_wallet_accounts(target_engine: Engine) -> None:
    """Seamlessly seed wallet_accounts rows for 'bybit' and 'forex' if absent, preserving existing balances."""
    from app.models import Trade, WalletAccount

    session_cls = sessionmaker(autocommit=False, autoflush=False, bind=target_engine)
    db = session_cls()
    try:
        existing_markets = {
            row.market for row in db.query(WalletAccount.market).all()
        }
        missing_markets = [m for m in ("bybit", "forex") if m not in existing_markets]
        if not missing_markets:
            return

        trades = db.query(Trade).all()
        defaults = {
            "bybit": {
                "upper": "BYBIT",
                "base_balance": DEFAULT_INITIAL_BYBIT_BALANCE,
                "account_profile": "bybit_vip0_linear",
            },
            "forex": {
                "upper": "FOREX",
                "base_balance": DEFAULT_INITIAL_FOREX_BALANCE,
                "account_profile": "mt5_raw_ecn",
            },
        }

        for market in missing_markets:
            cfg = defaults[market]
            market_upper = str(cfg["upper"])
            base_balance = Decimal(cfg["base_balance"])  # type: ignore[arg-type]

            closed_pnl = Decimal("0.000000")
            open_partial_pnl = Decimal("0.000000")
            open_unrealized = Decimal("0.000000")

            for t in trades:
                if (t.market or "").upper() != market_upper:
                    continue
                t_pnl = Decimal(str(t.pnl_usd if t.pnl_usd is not None else 0))
                t_partial = Decimal(str(t.realized_partial_pnl_usd if t.realized_partial_pnl_usd is not None else 0))
                if (t.status or "").upper() == "CLOSED":
                    closed_pnl += t_pnl
                else:
                    open_partial_pnl += t_partial
                    open_unrealized += (t_pnl - t_partial)

            active_bal = _q6(max(Decimal("0.000000"), base_balance + closed_pnl + open_partial_pnl))
            equity_baseline = _q6(max(Decimal("0.000000"), active_bal + open_unrealized))

            db.add(
                WalletAccount(
                    market=market,
                    seed_capital=active_bal,
                    active_balance=active_bal,
                    vault_balance=Decimal("0.000000"),
                    equity_baseline_day=equity_baseline,
                    cumulative_net_cashflow_day=Decimal("0.000000"),
                    account_profile=str(cfg["account_profile"]),
                    realism_enabled=True,
                    forex_leverage=500,
                    bybit_leverage=5,
                    allow_min_lot_override=False,
                    auto_sweep_enabled=False,
                    auto_sweep_base_capital=Decimal("20.000000"),
                    auto_sweep_threshold=Decimal("50.000000"),
                    auto_sweep_network="INTERNAL",
                    consecutive_losses=0,
                    circuit_breaker_until=None,
                )
            )
        db.commit()
    finally:
        db.close()


def init_db(target_engine: Optional[Engine] = None) -> None:
    """Create all database tables if they do not exist, run idempotent migrations, and seed wallets."""
    import app.models  # noqa: F401 - Ensure ORM models are registered on Base.metadata

    resolved_engine = target_engine or engine
    Base.metadata.create_all(bind=resolved_engine)
    migrate_schema(resolved_engine)
    seed_wallet_accounts(resolved_engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
