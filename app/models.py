"""SQLAlchemy 2.x ORM models for trades, equity, bot state, strategies, and logs."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Trade(Base):
    """Represents an open or closed trade across Bybit or Forex markets."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)  # BYBIT | FOREX
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))  # LONG | SHORT
    strategy_name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="OPEN")  # OPEN | CLOSED
    qty: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    entry_reason: Mapped[str] = mapped_column(Text, default="")
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class EquitySnapshot(Base):
    """Time-series snapshot of account equity across markets."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=utcnow)
    total_equity: Mapped[float] = mapped_column(Float)
    bybit_equity: Mapped[float] = mapped_column(Float)
    forex_equity: Mapped[float] = mapped_column(Float)
    daily_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    daily_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)


class BotState(Base):
    """Singleton state table tracking bot execution, regimes, and kill switch."""

    __tablename__ = "bot_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    trading_mode: Mapped[str] = mapped_column(String(16), default="PAPER")
    is_running: Mapped[bool] = mapped_column(Boolean, default=True)
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False)
    kill_switch_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active_regime_bybit: Mapped[str] = mapped_column(String(32), default="TRENDING_UP")
    active_regime_forex: Mapped[str] = mapped_column(String(32), default="RANGE_BOUND")
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StrategyConfig(Base):
    """Configuration and performance summary for each trading strategy."""

    __tablename__ = "strategy_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text)
    target_regime: Mapped[str] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)


class SystemLog(Base):
    """Structured operational log entries surfaced on the dashboard."""

    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, default=utcnow)
    level: Mapped[str] = mapped_column(String(16), index=True)  # INFO | WARN | ERROR | CRITICAL
    source: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
