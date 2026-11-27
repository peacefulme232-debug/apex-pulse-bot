"""SQLAlchemy 2.x ORM models for trades, equity, bot state, strategies, and logs."""

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, Numeric, String, Text
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
    initial_qty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    partial_tp_taken: Mapped[bool] = mapped_column(Boolean, default=False)
    realized_partial_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Phase 1 Realism Engine columns (exact Decimal precision)
    order_entry_type: Mapped[str] = mapped_column(String(16), default="MARKET")
    order_exit_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, default="MARKET")
    requested_entry_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8, asdecimal=True), nullable=True)
    filled_entry_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8, asdecimal=True), nullable=True)
    requested_exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8, asdecimal=True), nullable=True)
    filled_exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8, asdecimal=True), nullable=True)
    initial_margin: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    maintenance_margin: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    liquidation_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 8, asdecimal=True), nullable=True)
    gross_pnl: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    entry_fee: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    exit_fee: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    spread_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    slippage_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    funding_or_swap_fee: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    net_pnl: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    breakeven_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    last_funding_applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_swap_applied_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)


class WalletAccount(Base):
    """Per-market wallet account (one row per market: 'bybit', 'forex') with realism & sweep settings."""

    __tablename__ = "wallet_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    market: Mapped[str] = mapped_column(String(16), unique=True, index=True)  # bybit | forex
    seed_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    active_balance: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    vault_balance: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    equity_baseline_day: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    cumulative_net_cashflow_day: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    account_profile: Mapped[str] = mapped_column(String(32), default="bybit_vip0_linear")
    realism_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    forex_leverage: Mapped[int] = mapped_column(Integer, default=500)
    bybit_leverage: Mapped[int] = mapped_column(Integer, default=5)
    allow_min_lot_override: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_sweep_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_sweep_base_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("20.000000")
    )
    auto_sweep_threshold: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("50.000000")
    )
    auto_sweep_network: Mapped[str] = mapped_column(String(32), default="INTERNAL")
    consecutive_losses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    circuit_breaker_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class WalletTransaction(Base):
    """Immutable ledger of deposits, withdrawals, sweeps, and transfers per market."""

    __tablename__ = "wallet_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)  # bybit | forex
    tx_type: Mapped[str] = mapped_column(String(32))
    network: Mapped[str] = mapped_column(String(32), default="INTERNAL")
    gross_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    network_fee: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    fx_spread_fee: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    net_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    active_balance_before: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    active_balance_after: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    vault_balance_before: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    vault_balance_after: Mapped[Decimal] = mapped_column(
        Numeric(18, 6, asdecimal=True), default=Decimal("0.000000")
    )
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=utcnow
    )


class ExecutionRejection(Base):
    """Audit log of orders rejected due to realism constraints or strategy vetoes."""

    __tablename__ = "execution_rejections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)  # bybit | forex
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16))
    strategy: Mapped[Optional[str]] = mapped_column(String(48), nullable=True)
    confluence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(6, 2, asdecimal=True), nullable=True
    )
    reason_code: Mapped[str] = mapped_column(String(48), index=True)
    requested_qty: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 8, asdecimal=True), nullable=True
    )
    required_min_qty: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 8, asdecimal=True), nullable=True
    )
    required_margin: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 6, asdecimal=True), nullable=True
    )
    available_free_margin: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 6, asdecimal=True), nullable=True
    )
    details: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, default=utcnow
    )

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("requested_qty", Decimal("0.00000000"))
        kwargs.setdefault("required_min_qty", Decimal("0.00000000"))
        kwargs.setdefault("required_margin", Decimal("0.000000"))
        kwargs.setdefault("available_free_margin", Decimal("0.000000"))
        super().__init__(**kwargs)


class ArmedRetestSetup(Base):
    """Persisted state for Breakout + Retest setups across container restarts."""

    __tablename__ = "armed_retest_setups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    market: Mapped[str] = mapped_column(String(16), index=True)  # bybit | forex
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(16))  # LONG | SHORT
    zone_low: Mapped[Decimal] = mapped_column(Numeric(18, 8, asdecimal=True))
    zone_high: Mapped[Decimal] = mapped_column(Numeric(18, 8, asdecimal=True))
    breakout_candle_range: Mapped[Decimal] = mapped_column(Numeric(18, 8, asdecimal=True))
    breakout_atr: Mapped[Decimal] = mapped_column(Numeric(18, 8, asdecimal=True))
    bars_remaining: Mapped[int] = mapped_column(Integer, default=10)
    status: Mapped[str] = mapped_column(String(24), default="ARMED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


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
