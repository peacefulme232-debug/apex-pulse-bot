"""FastAPI application entrypoint and API routes for APEX-PULSE."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from app.brokers.bybit_broker import BybitBrokerError
from app.brokers.forex_deriv import DerivBrokerError
from app.brokers.forex_oanda import OandaBrokerError
from app.brokers.paper_sim import INITIAL_BYBIT_BALANCE, INITIAL_FOREX_BALANCE, PaperSimBroker
from app.config import settings
from app.database import SessionLocal, create_db_engine, get_db, init_db
from app.engine.backtester import Backtester
from app.engine.historical_data import HistoricalDataFetcher
from app.engine.runner import BotRunner
from app.engine.summarizer import BotSummarizer
from app.models import BotState, EquitySnapshot, StrategyConfig, SystemLog, Trade, WalletTransaction
from app.security import (
    clear_session_cookie,
    extract_client_ip,
    is_auth_enabled,
    is_request_authenticated,
    log_security_startup_warnings,
    rate_limiter,
    require_dashboard_auth,
    set_session_cookie,
    verify_cron_secret,
    verify_pin,
)
from app.services.realism_engine import ZERO, ensure_utc, q6, to_decimal
from app.services.wallet_service import (
    WalletError,
    build_wallet_summary,
    compute_daily_pnl_isolated,
    execute_leave_balance,
    execute_reset_challenge,
    execute_transfer,
    list_execution_rejections,
    list_wallet_transactions,
    update_wallet_settings,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class PinLoginRequest(BaseModel):
    """Payload for unlocking the dashboard with the configured Phone PIN."""

    pin: str = Field(min_length=1, max_length=64)


class StrategyToggleRequest(BaseModel):
    """Payload for toggling a strategy's enabled state."""

    name: str
    enabled: Optional[bool] = None


class KillSwitchRequest(BaseModel):
    """Optional payload for activating the emergency kill switch."""

    reason: Optional[str] = None


class BacktestRequest(BaseModel):
    """Validated payload for running an on-demand zero-lookahead backtest."""

    symbol: str = Field(default="BTCUSDT", min_length=2, max_length=24)
    interval: Literal["15m", "1h", "4h", "1d"] = "1h"
    days: int = Field(default=60, ge=7, le=180)
    starting_balance: float = Field(default=10000.0, ge=100.0, le=1000000.0)
    enable_partial_tp: bool = True
    enable_confluence: bool = True


class WalletTransferRequest(BaseModel):
    """Payload for manual transfers between Trading, Vault, and External rails."""

    market: Literal["bybit", "forex"]
    tx_type: Literal["TRADING_TO_VAULT", "VAULT_TO_TRADING", "TRADING_TO_EXTERNAL", "EXTERNAL_TO_TRADING"]
    amount: Decimal = Field(gt=Decimal("0"))
    network: str = Field(default="INTERNAL")
    note: str = Field(default="", max_length=256)


class LeaveBalanceRequest(BaseModel):
    """Payload for 1-click Withdraw Excess & Leave $X ($20 Challenge)."""

    market: Literal["bybit", "forex", "all"] = "all"
    target_active_balance: Decimal = Field(gt=Decimal("0"))
    network: str = Field(default="INTERNAL")
    split_mode: Literal["equal", "proportional", "bybit_only", "forex_only"] = "equal"
    note: str = Field(default="", max_length=256)


class ResetChallengeRequest(BaseModel):
    """Payload for clean-slate challenge reset (active=target, vault=0)."""

    market: Literal["bybit", "forex", "all"] = "all"
    target_active_balance: Decimal = Field(gt=Decimal("0"))
    note: str = Field(default="", max_length=256)


class WalletSettingsPatchRequest(BaseModel):
    """Payload for updating auto-sweep and realism/leverage settings for a market wallet."""

    market: Literal["bybit", "forex"]
    account_profile: Optional[str] = None
    realism_enabled: Optional[bool] = None
    forex_leverage: Optional[int] = Field(default=None, ge=1, le=2000)
    bybit_leverage: Optional[int] = Field(default=None, ge=1, le=100)
    allow_min_lot_override: Optional[bool] = None
    auto_sweep_enabled: Optional[bool] = None
    auto_sweep_base_capital: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    auto_sweep_threshold: Optional[Decimal] = Field(default=None, gt=Decimal("0"))
    auto_sweep_network: Optional[str] = None


class BrainConfigPatchRequest(BaseModel):
    """Payload for updating Strategy Brain toggles, confluence/veto thresholds, and breaker reset."""

    ema_50_pullback_enabled: Optional[bool] = None
    breakout_retest_enabled: Optional[bool] = None
    rsi_divergence_sniper_enabled: Optional[bool] = None
    trend_rider_enabled: Optional[bool] = None
    mean_reversion_enabled: Optional[bool] = None
    volatility_breakout_enabled: Optional[bool] = None
    min_confluence_score: Optional[int] = Field(default=None, ge=65, le=90)
    max_fee_drag_pct: Optional[float] = Field(default=None, gt=0.0, le=50.0)
    breakeven_r_trigger: Optional[float] = Field(default=None, ge=0.5, le=5.0)
    max_consecutive_losses: Optional[int] = Field(default=None, ge=1, le=10)
    micro_shield_enabled: Optional[bool] = None
    reset_circuit_breaker: Optional[bool] = None


class CloseAllPositionsRequest(BaseModel):
    """Optional payload for MT5-style Close All positions endpoint."""

    market: Optional[Literal["all", "bybit", "forex", "ALL", "BYBIT", "FOREX"]] = None
    reason: Optional[str] = Field(default=None, max_length=256)


class WalkForwardRequest(BaseModel):
    """Validated payload for running an out-of-sample Walk-Forward validation."""

    symbol: str = Field(min_length=3, max_length=32)
    market: Optional[Literal["bybit", "forex"]] = None
    starting_capital: Decimal = Field(default=Decimal("20.00"), gt=Decimal("0"))
    strategies: Optional[list[str]] = None
    folds: Literal[3, 5] = 3
    leverage: Optional[int] = Field(default=None, ge=1, le=2000)
    profile: Optional[str] = None
    realism_enabled: bool = True
    min_confluence_score: int = Field(default=75, ge=65, le=90)
    max_fee_drag_pct: float = Field(default=15.0, gt=0.0, le=50.0)
    days: int = Field(default=60, ge=14, le=365)


def _ensure_bot_state(db: Session, broker: PaperSimBroker) -> BotState:
    state = db.query(BotState).first()
    if state is None:
        broker.seed_demo_data(db)
        state = db.query(BotState).first()
    assert state is not None
    return state


def create_app(database_url: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if database_url:
        custom_engine = create_db_engine(database_url)
        session_factory = sessionmaker(autocommit=False, autoflush=False, bind=custom_engine)
        target_engine = custom_engine
    else:
        session_factory = SessionLocal
        target_engine = None

    broker = PaperSimBroker(session_factory=session_factory)
    bot_runner = BotRunner(broker=broker, session_factory=session_factory)
    summarizer = BotSummarizer()

    def _get_scoped_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    db_dependency = _get_scoped_db if database_url else get_db

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        log_security_startup_warnings()
        init_db(target_engine=target_engine)
        await bot_runner.start()
        fastapi_app.state.bot_runner = bot_runner
        fastapi_app.state.broker = broker
        try:
            yield
        except asyncio.CancelledError:
            pass
        finally:
            await bot_runner.stop()

    fastapi_app = FastAPI(
        title="APEX-PULSE",
        description="24/7 Cloud-Ready Multi-Market Algorithmic Trading Bot (Phase 6 Production)",
        version="0.6.0",
        lifespan=lifespan,
    )

    @fastapi_app.get("/", response_class=HTMLResponse)
    def serve_dashboard() -> HTMLResponse:
        """Serve the single-file mobile-first monitoring dashboard."""
        index_path = TEMPLATES_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard template not found")
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    def _build_brokers_status() -> dict[str, Any]:
        forex_symbols = [s.strip() for s in settings.FOREX_SYMBOLS.split(",") if s.strip()]
        primary_fx = forex_symbols[0] if forex_symbols else "EUR_USD"
        forex_session = bot_runner.get_forex_session_status(primary_fx)
        return {
            "bybit": bot_runner.bybit_router.get_status(),
            "forex": bot_runner.forex_router.get_status(),
            "forex_session": forex_session,
        }

    def _compute_strategy_metrics(db: Session, strat: StrategyConfig) -> tuple[float, float]:
        closed_trades = (
            db.query(Trade)
            .filter(Trade.status == "CLOSED", Trade.strategy_name == strat.name)
            .order_by(Trade.closed_at.asc())
            .all()
        )
        if not closed_trades:
            pf = round(1.0 + (strat.win_rate / 100.0), 2) if strat.net_pnl_usd > 0 else 0.0
            return pf, 0.0

        gross_profit = sum(float(t.pnl_usd) for t in closed_trades if float(t.pnl_usd) > 0)
        gross_loss = abs(sum(float(t.pnl_usd) for t in closed_trades if float(t.pnl_usd) < 0))
        if gross_loss > 0:
            pf = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            pf = round(gross_profit, 2)
        else:
            pf = 0.0

        equity = INITIAL_BYBIT_BALANCE
        peak = equity
        max_dd = 0.0
        for t in closed_trades:
            equity += float(t.pnl_usd)
            if equity > peak:
                peak = equity
            dd = ((peak - equity) / peak) * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return pf, round(max_dd, 2)

    @fastapi_app.get("/health")
    def health_check(db: Session = Depends(db_dependency)) -> dict[str, Any]:
        """Return minimal non-sensitive service health, mode, kill switch, auth, and loop status."""
        state = _ensure_bot_state(db, broker)
        return {
            "status": "ok",
            "mode": settings.TRADING_MODE,
            "kill_switch": bool(state.kill_switch_active),
            "auth_enabled": is_auth_enabled(),
            "background_loop": bool(settings.ENABLE_BACKGROUND_LOOP),
        }

    @fastapi_app.get("/api/auth/status")
    def get_auth_status(request: Request) -> dict[str, bool]:
        """Return whether PIN auth is enabled and whether the caller has a valid session."""
        return {
            "auth_enabled": is_auth_enabled(),
            "authenticated": is_request_authenticated(request),
        }

    @fastapi_app.post("/api/auth/login")
    def login_with_pin(
        payload: PinLoginRequest,
        request: Request,
        response: Response,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Validate Dashboard PIN with dual-window brute-force protection and set HttpOnly cookie."""
        if not is_auth_enabled():
            return {"status": "ok", "authenticated": True, "auth_enabled": False}

        client_ip = extract_client_ip(request)
        allowed, retry_after = rate_limiter.check_allowed(client_ip)
        if not allowed:
            db.add(
                SystemLog(
                    level="WARN",
                    source="AuthGuard",
                    message=f"PIN login locked out for {client_ip} (Retry-After: {retry_after}s).",
                )
            )
            db.commit()
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed PIN attempts. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        if not verify_pin(payload.pin):
            rate_limiter.record_failure(client_ip)
            db.add(
                SystemLog(
                    level="WARN",
                    source="AuthGuard",
                    message=f"Failed dashboard PIN login attempt from {client_ip}.",
                )
            )
            db.commit()
            raise HTTPException(status_code=401, detail="Invalid PIN.")

        set_session_cookie(response, request)
        return {"status": "ok", "authenticated": True, "auth_enabled": True}

    @fastapi_app.post("/api/auth/logout")
    def logout_session(request: Request, response: Response) -> dict[str, Any]:
        """Clear the apex_session cookie to lock the dashboard."""
        clear_session_cookie(response, request)
        return {"status": "ok", "authenticated": False}

    @fastapi_app.get("/api/brokers/status", dependencies=[Depends(require_dashboard_auth)])
    def get_brokers_status() -> dict[str, Any]:
        """Return live connection health for Bybit, Forex brokers, and Forex session/news guard."""
        return _build_brokers_status()

    @fastapi_app.get("/api/summary", dependencies=[Depends(require_dashboard_auth)])
    def get_summary(
        market: Literal["ALL", "BYBIT", "FOREX"] = Query(default="ALL"),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Return plain-English bot accuracy, edge math, strategy rankings, and risk shield audit."""
        _ensure_bot_state(db, broker)
        balances = broker.get_balance(db=db)
        forex_symbols = [s.strip() for s in settings.FOREX_SYMBOLS.split(",") if s.strip()]
        primary_fx = forex_symbols[0] if forex_symbols else "EUR_USD"
        forex_session = bot_runner.get_forex_session_status(primary_fx)
        return summarizer.build_summary(
            db=db,
            market=market,
            balances=balances,
            forex_session=forex_session,
        )

    @fastapi_app.get("/api/dashboard", dependencies=[Depends(require_dashboard_auth)])
    def get_dashboard(
        market: Literal["ALL", "BYBIT", "FOREX"] = Query(default="ALL"),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Return complete dashboard telemetry filtered by market."""
        state = _ensure_bot_state(db, broker)
        balances = broker.get_balance(db=db)

        bybit_wallet = broker.get_wallet_account(db, "bybit")
        forex_wallet = broker.get_wallet_account(db, "forex")

        if market == "BYBIT":
            selected_equity = balances["bybit_equity"]
            baseline_dec = to_decimal(bybit_wallet.equity_baseline_day)
            fallback_cap = to_decimal(bybit_wallet.active_balance)
            active_regime = state.active_regime_bybit
        elif market == "FOREX":
            selected_equity = balances["forex_equity"]
            baseline_dec = to_decimal(forex_wallet.equity_baseline_day)
            fallback_cap = to_decimal(forex_wallet.active_balance)
            active_regime = state.active_regime_forex
        else:
            selected_equity = balances["total_equity"]
            baseline_dec = q6(
                to_decimal(bybit_wallet.equity_baseline_day)
                + to_decimal(forex_wallet.equity_baseline_day)
            )
            fallback_cap = q6(
                to_decimal(bybit_wallet.active_balance)
                + to_decimal(forex_wallet.active_balance)
            )
            active_regime = f"{state.active_regime_bybit} / {state.active_regime_forex}"

        daily_pnl_usd_dec, daily_pnl_pct_dec = compute_daily_pnl_isolated(
            current_active_equity=to_decimal(selected_equity),
            equity_baseline_day=baseline_dec,
            fallback_capital=fallback_cap,
        )
        daily_pnl_usd = float(round(daily_pnl_usd_dec, 2))
        daily_pnl_pct = float(round(daily_pnl_pct_dec, 2))
        daily_loss_used_pct = round(max(0.0, -daily_pnl_pct), 2)

        open_q = db.query(Trade).filter(Trade.status == "OPEN")
        closed_q = db.query(Trade).filter(Trade.status == "CLOSED")
        if market != "ALL":
            open_q = open_q.filter(Trade.market == market)
            closed_q = closed_q.filter(Trade.market == market)

        open_positions = open_q.order_by(Trade.opened_at.desc()).all()
        closed_trades = closed_q.order_by(Trade.closed_at.desc()).limit(25).all()

        all_closed_for_wr = closed_q.all()
        if all_closed_for_wr:
            wins = sum(1 for t in all_closed_for_wr if t.pnl_usd > 0)
            win_rate = round((wins / len(all_closed_for_wr)) * 100.0, 1)
        else:
            win_rate = 0.0

        snapshots = (
            db.query(EquitySnapshot)
            .order_by(EquitySnapshot.timestamp.desc())
            .limit(60)
            .all()
        )
        snapshots.reverse()

        # Fetch wallet cashflow events to compute Cash-Flow Adjusted Cumulative Trading PnL series
        tx_q = db.query(WalletTransaction).order_by(WalletTransaction.created_at.asc())
        if market == "BYBIT":
            tx_q = tx_q.filter(WalletTransaction.market == "bybit")
        elif market == "FOREX":
            tx_q = tx_q.filter(WalletTransaction.market == "forex")
        wallet_txs = tx_q.all()

        equity_curve = []
        adjusted_trading_pnl_curve = []
        raw_active_balance_curve = []
        first_val = (
            (
                snapshots[0].bybit_equity
                if market == "BYBIT"
                else snapshots[0].forex_equity
                if market == "FOREX"
                else snapshots[0].total_equity
            )
            if snapshots
            else 0.0
        )
        for snap in snapshots:
            if market == "BYBIT":
                val = snap.bybit_equity
            elif market == "FOREX":
                val = snap.forex_equity
            else:
                val = snap.total_equity

            snap_ts = ensure_utc(snap.timestamp)
            cum_cashflow = ZERO
            for tx in wallet_txs:
                if tx.created_at and ensure_utc(tx.created_at) <= snap_ts:
                    cum_cashflow += (
                        to_decimal(tx.active_balance_after)
                        - to_decimal(tx.active_balance_before)
                    )
            adjusted_equity = round(float(to_decimal(val) - cum_cashflow), 2)
            adjusted_pnl_usd = round(adjusted_equity - round(first_val, 2), 2)
            iso_ts = snap.timestamp.isoformat()
            raw_val = round(val, 2)

            equity_curve.append(
                {
                    "timestamp": iso_ts,
                    "total_equity": round(snap.total_equity, 2),
                    "bybit_equity": round(snap.bybit_equity, 2),
                    "forex_equity": round(snap.forex_equity, 2),
                    "value": raw_val,
                    "adjusted_equity": adjusted_equity,
                    "adjusted_pnl_usd": adjusted_pnl_usd,
                }
            )
            adjusted_trading_pnl_curve.append(
                {
                    "timestamp": iso_ts,
                    "value": adjusted_pnl_usd,
                    "adjusted_equity": adjusted_equity,
                }
            )
            raw_active_balance_curve.append(
                {
                    "timestamp": iso_ts,
                    "value": raw_val,
                }
            )

        strategies = db.query(StrategyConfig).order_by(StrategyConfig.id.asc()).all()
        logs = db.query(SystemLog).order_by(SystemLog.id.desc()).limit(50).all()

        strategy_payload = []
        for s in strategies:
            pf, max_dd = _compute_strategy_metrics(db, s)
            strategy_payload.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "display_name": s.display_name,
                    "description": s.description,
                    "target_regime": s.target_regime,
                    "enabled": s.enabled,
                    "total_trades": s.total_trades,
                    "win_rate": round(s.win_rate, 1),
                    "net_pnl_usd": round(s.net_pnl_usd, 2),
                    "profit_factor": pf,
                    "max_drawdown_pct": max_dd,
                }
            )

        return {
            "market": market,
            "trading_mode": state.trading_mode,
            "brokers_status": _build_brokers_status(),
            "last_confluence": bot_runner.last_confluence,
            "bot_state": {
                "is_running": state.is_running,
                "kill_switch_active": state.kill_switch_active,
                "kill_switch_reason": state.kill_switch_reason,
                "active_regime_bybit": state.active_regime_bybit,
                "active_regime_forex": state.active_regime_forex,
                "last_heartbeat": state.last_heartbeat.isoformat() if state.last_heartbeat else None,
            },
            "kpis": {
                "total_equity": round(selected_equity, 2),
                "bybit_equity": balances["bybit_equity"],
                "forex_equity": balances["forex_equity"],
                "daily_pnl_usd": daily_pnl_usd,
                "daily_pnl_pct": daily_pnl_pct,
                "win_rate": win_rate,
                "active_regime": active_regime,
                "daily_loss_used_pct": daily_loss_used_pct,
                "max_daily_loss_pct": settings.MAX_DAILY_LOSS_PCT,
            },
            "equity_curve": equity_curve,
            "adjusted_trading_pnl_curve": adjusted_trading_pnl_curve,
            "raw_active_balance_curve": raw_active_balance_curve,
            "strategies": strategy_payload,
            "open_positions": [broker._trade_to_dict(t) for t in open_positions],
            "closed_trades": [broker._trade_to_dict(t) for t in closed_trades],
            "system_logs": [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level,
                    "source": log.source,
                    "message": log.message,
                }
                for log in logs
            ],
        }

    @fastapi_app.post("/api/strategy/toggle", dependencies=[Depends(require_dashboard_auth)])
    def toggle_strategy(
        payload: StrategyToggleRequest,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Toggle or set a strategy's enabled flag and record an audit log."""
        strat = db.query(StrategyConfig).filter(StrategyConfig.name == payload.name).first()
        if strat is None:
            raise HTTPException(status_code=404, detail=f"Strategy '{payload.name}' not found")

        strat.enabled = (not strat.enabled) if payload.enabled is None else payload.enabled
        state_label = "ENABLED" if strat.enabled else "DISABLED"
        db.add(
            SystemLog(
                level="INFO",
                source="Dashboard",
                message=f"Strategy '{strat.display_name}' ({strat.name}) {state_label}.",
            )
        )
        db.commit()
        db.refresh(strat)
        return {
            "status": "ok",
            "name": strat.name,
            "display_name": strat.display_name,
            "enabled": strat.enabled,
        }

    @fastapi_app.post("/api/kill-switch", dependencies=[Depends(require_dashboard_auth)])
    async def activate_kill_switch(
        payload: Optional[KillSwitchRequest] = Body(default=None),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Activate Emergency Kill Switch, close all open positions, and log CRITICAL event."""
        async with bot_runner._tick_lock:
            state = _ensure_bot_state(db, broker)
            reason = (
                payload.reason
                if payload and payload.reason
                else "Manual Emergency Kill Switch activated from dashboard."
            )

            state.kill_switch_active = True
            state.is_running = False
            state.kill_switch_reason = reason
            state.last_heartbeat = datetime.now(timezone.utc)

            open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
            closed_count = 0
            for trade in open_trades:
                broker.close_position(trade.id, "Emergency Kill Switch", db=db)
                closed_count += 1

            db.add(
                SystemLog(
                    level="CRITICAL",
                    source="KillSwitch",
                    message=f"KILL SWITCH ACTIVATED: {reason} Closed {closed_count} open position(s).",
                )
            )
            db.commit()
        await bot_runner.notifier.notify_kill_switch(reason=reason, closed_positions=closed_count)
        return {
            "status": "ok",
            "kill_switch_active": True,
            "is_running": False,
            "closed_positions": closed_count,
            "reason": reason,
        }

    @fastapi_app.post("/api/bot/resume", dependencies=[Depends(require_dashboard_auth)])
    def resume_bot(db: Session = Depends(db_dependency)) -> dict[str, Any]:
        """Safely reset Kill Switch and resume the paper trading loop."""
        state = _ensure_bot_state(db, broker)
        state.kill_switch_active = False
        state.kill_switch_reason = None
        state.is_running = True
        state.last_heartbeat = datetime.now(timezone.utc)

        db.add(
            SystemLog(
                level="INFO",
                source="BotRunner",
                message="Kill Switch cleared. Paper trading heartbeat resumed.",
            )
        )
        db.commit()
        return {
            "status": "ok",
            "kill_switch_active": False,
            "is_running": True,
        }

    @fastapi_app.post("/api/backtest/run", dependencies=[Depends(require_dashboard_auth)])
    async def run_backtest(payload: BacktestRequest) -> dict[str, Any]:
        """Run a zero-lookahead historical backtest off the async event loop."""
        symbol = payload.symbol.strip()

        def _execute_backtest() -> dict[str, Any]:
            fetcher = HistoricalDataFetcher()
            try:
                df, source = fetcher.fetch(
                    symbol=symbol,
                    interval=payload.interval,
                    days=payload.days,
                )
            finally:
                fetcher.close()
            engine = Backtester()
            result = engine.run(
                df=df,
                symbol=symbol,
                interval=payload.interval,
                starting_balance=payload.starting_balance,
                enable_partial_tp=payload.enable_partial_tp,
                enable_confluence=payload.enable_confluence,
                data_source=source,
            )
            return result.to_dict()

        return await asyncio.to_thread(_execute_backtest)

    @fastapi_app.get("/api/chart/{symbol}", dependencies=[Depends(require_dashboard_auth)])
    def get_chart_data(
        symbol: str,
        market: Optional[Literal["BYBIT", "FOREX"]] = Query(default=None),
        interval: Literal["15m", "1h", "4h"] = Query(default="15m"),
        limit: int = Query(default=150, ge=20, le=500),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Return ascending deduplicated OHLCV candles and any active open trade levels for symbol."""
        _ensure_bot_state(db, broker)
        cleaned_symbol = symbol.strip()
        resolved_market = market or ("FOREX" if "_" in cleaned_symbol else "BYBIT")
        source = "PAPER_SIM"
        df = None

        if resolved_market == "BYBIT" and bot_runner.bybit_router.is_demo_active():
            try:
                df = bot_runner.bybit_router.get_broker().get_ohlcv(
                    cleaned_symbol, interval=interval, limit=limit
                )
                source = "BYBIT_DEMO"
            except BybitBrokerError:
                df = None
        elif resolved_market == "FOREX" and bot_runner.forex_router.is_cloud_active():
            try:
                df = bot_runner.forex_router.get_broker().get_ohlcv(
                    cleaned_symbol, interval=interval, limit=limit
                )
                source = "FOREX_CLOUD"
            except (OandaBrokerError, DerivBrokerError):
                df = None

        if df is None or df.empty:
            df = broker.get_ohlcv(cleaned_symbol, interval=interval, limit=limit)
            source = "PAPER_SIM"

        candles_by_time: dict[int, dict[str, Any]] = {}
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if hasattr(ts, "timestamp"):
                epoch_sec = int(ts.timestamp())
            else:
                epoch_sec = int(datetime.fromisoformat(str(ts)).timestamp())
            candles_by_time[epoch_sec] = {
                "time": epoch_sec,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0)),
            }

        candles = [candles_by_time[t] for t in sorted(candles_by_time.keys())]

        open_trade_row = (
            db.query(Trade)
            .filter(Trade.status == "OPEN", Trade.symbol == cleaned_symbol)
            .order_by(Trade.opened_at.desc())
            .first()
        )
        open_trade: Optional[dict[str, Any]] = None
        if open_trade_row is not None:
            open_trade = {
                "id": open_trade_row.id,
                "symbol": open_trade_row.symbol,
                "side": open_trade_row.side,
                "strategy_name": open_trade_row.strategy_name,
                "entry_price": float(open_trade_row.entry_price),
                "stop_loss": float(open_trade_row.stop_loss),
                "take_profit": float(open_trade_row.take_profit),
                "qty": float(open_trade_row.qty),
                "partial_tp_taken": bool(open_trade_row.partial_tp_taken),
            }

        return {
            "symbol": cleaned_symbol,
            "market": resolved_market,
            "interval": interval,
            "source": source,
            "candles": candles,
            "open_trade": open_trade,
        }

    @fastapi_app.post("/api/alerts/test", dependencies=[Depends(require_dashboard_auth)])
    async def send_test_alert() -> dict[str, Any]:
        """Dispatch a test Telegram HTML alert (PIN-protected)."""
        sent, reason = await bot_runner.notifier.send_test_alert()
        return {"sent": sent, "reason": reason}

    @fastapi_app.post("/api/cron/tick", dependencies=[Depends(verify_cron_secret)])
    async def cron_tick() -> dict[str, Any]:
        """Run one mutually-excluded trading tick triggered by Cloud Scheduler / external cron."""
        return await bot_runner.run_locked_tick()

    @fastapi_app.post("/api/cron/daily-summary", dependencies=[Depends(verify_cron_secret)])
    async def cron_daily_summary(db: Session = Depends(db_dependency)) -> dict[str, Any]:
        """Build BotSummarizer daily report and dispatch it via TelegramNotifier."""
        _ensure_bot_state(db, broker)
        balances = broker.get_balance(db=db)
        forex_symbols = [s.strip() for s in settings.FOREX_SYMBOLS.split(",") if s.strip()]
        primary_fx = forex_symbols[0] if forex_symbols else "EUR_USD"
        forex_session = bot_runner.get_forex_session_status(primary_fx)
        summary = summarizer.build_summary(
            db=db,
            market="ALL",
            balances=balances,
            forex_session=forex_session,
        )
        sent = await bot_runner.notifier.notify_daily_summary(summary)
        return {
            "status": "ok",
            "sent": sent,
            "health_grade": summary["health_grade"],
            "net_pnl_usd": summary["net_pnl_usd"],
        }

    # ------------------------------------------------------------------
    # Phase 2: Virtual Wallet / Vault, Withdrawal Fees, & $20 Challenge API
    # ------------------------------------------------------------------

    @fastapi_app.get("/api/wallet/summary", dependencies=[Depends(require_dashboard_auth)])
    def get_wallet_summary(db: Session = Depends(db_dependency)) -> dict[str, Any]:
        """Return per-market + combined balances, margin, withdrawable active, friction breakdown & reconciliation."""
        _ensure_bot_state(db, broker)
        return build_wallet_summary(db)

    @fastapi_app.post("/api/wallet/transfer", dependencies=[Depends(require_dashboard_auth)])
    def post_wallet_transfer(
        payload: WalletTransferRequest,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Execute manual transfer between Trading, Vault, and External rails with realistic fee calculation."""
        _ensure_bot_state(db, broker)
        try:
            return execute_transfer(
                db=db,
                market=payload.market,
                tx_type=payload.tx_type,
                amount=payload.amount,
                network=payload.network,
                note=payload.note,
            )
        except WalletError as exc:
            db.rollback()
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @fastapi_app.post("/api/wallet/leave-balance", dependencies=[Depends(require_dashboard_auth)])
    def post_wallet_leave_balance(
        payload: LeaveBalanceRequest,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """1-Click Withdraw Excess & Leave $X ($20 / $50 / $100 Challenge). Returns 409 if open positions exist."""
        _ensure_bot_state(db, broker)
        try:
            return execute_leave_balance(
                db=db,
                market=payload.market,
                target_active_balance=payload.target_active_balance,
                network=payload.network,
                split_mode=payload.split_mode,
                note=payload.note,
            )
        except WalletError as exc:
            db.rollback()
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @fastapi_app.post("/api/wallet/reset-challenge", dependencies=[Depends(require_dashboard_auth)])
    def post_wallet_reset_challenge(
        payload: ResetChallengeRequest,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Clean-slate challenge reset: active_balance = target, vault_balance = 0."""
        _ensure_bot_state(db, broker)
        try:
            return execute_reset_challenge(
                db=db,
                market=payload.market,
                target_active_balance=payload.target_active_balance,
                note=payload.note,
            )
        except WalletError as exc:
            db.rollback()
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @fastapi_app.patch("/api/wallet/settings", dependencies=[Depends(require_dashboard_auth)])
    def patch_wallet_settings(
        payload: WalletSettingsPatchRequest,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Update auto-sweep, account_profile, realism_enabled, leverage, and allow_min_lot_override."""
        _ensure_bot_state(db, broker)
        try:
            updates = payload.model_dump(exclude={"market"}, exclude_none=True)
            return update_wallet_settings(db=db, market=payload.market, updates=updates)
        except WalletError as exc:
            db.rollback()
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @fastapi_app.get("/api/wallet/transactions", dependencies=[Depends(require_dashboard_auth)])
    def get_wallet_transactions(
        market: Optional[Literal["bybit", "forex"]] = Query(default=None),
        tx_type: Optional[str] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Return paginated wallet transaction ledger."""
        _ensure_bot_state(db, broker)
        try:
            return list_wallet_transactions(
                db=db,
                market=market,
                tx_type=tx_type,
                limit=limit,
                offset=offset,
            )
        except WalletError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @fastapi_app.get("/api/wallet/rejections", dependencies=[Depends(require_dashboard_auth)])
    def get_wallet_rejections(
        market: Optional[Literal["bybit", "forex"]] = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Return paginated execution_rejections audit rows."""
        _ensure_bot_state(db, broker)
        try:
            return list_execution_rejections(db=db, market=market, limit=limit, offset=offset)
        except WalletError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @fastapi_app.get("/api/brain/telemetry", dependencies=[Depends(require_dashboard_auth)])
    def get_brain_telemetry(
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Return per-symbol regime, 0-100 confluence breakdown, armed setups, divergences, and circuit breaker."""
        _ensure_bot_state(db, broker)
        return bot_runner.get_brain_telemetry(db)

    @fastapi_app.patch("/api/brain/config", dependencies=[Depends(require_dashboard_auth)])
    def patch_brain_config(
        payload: BrainConfigPatchRequest,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Toggle strategies, adjust confluence/veto thresholds, or reset the circuit breaker."""
        _ensure_bot_state(db, broker)
        updates = payload.model_dump(exclude_none=True)
        return bot_runner.update_brain_config(db, updates)

    @fastapi_app.post("/api/strategy-lab/walk-forward", dependencies=[Depends(require_dashboard_auth)])
    async def run_walk_forward_validation(payload: WalkForwardRequest) -> dict[str, Any]:
        """Run a zero-lookahead Walk-Forward out-of-sample validation with real exchange friction."""
        from app.services.walk_forward import WalkForwardEngine

        symbol = payload.symbol.strip().upper()
        inferred_market = payload.market or ("forex" if "_" in symbol else "bybit")

        def _execute_walk_forward() -> dict[str, Any]:
            fetcher = HistoricalDataFetcher()
            try:
                df, source = fetcher.fetch(
                    symbol=symbol,
                    interval="1h",
                    days=payload.days,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"Upstream market data unavailable for {symbol}: {exc}",
                ) from exc
            finally:
                fetcher.close()

            if df is None or len(df) < 60:
                raise HTTPException(
                    status_code=503,
                    detail=f"Insufficient historical H1 candles for {symbol} walk-forward validation.",
                )

            wf_engine = WalkForwardEngine()
            try:
                result = wf_engine.run(
                    candles_h1=df,
                    symbol=symbol,
                    market=inferred_market,
                    starting_capital=payload.starting_capital,
                    strategies=payload.strategies,
                    folds=payload.folds,
                    leverage=payload.leverage,
                    profile=payload.profile,
                    realism_enabled=payload.realism_enabled,
                    min_confluence_score=payload.min_confluence_score,
                    max_fee_drag_pct=payload.max_fee_drag_pct,
                    data_source=source,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return result.to_dict()

        return await asyncio.to_thread(_execute_walk_forward)

    # ------------------------------------------------------------------
    # MT5-Style Manual Close & Close All Position Endpoints
    # ------------------------------------------------------------------

    @fastapi_app.post("/api/positions/close-all", dependencies=[Depends(require_dashboard_auth)])
    def close_all_positions(
        market: Optional[Literal["all", "bybit", "forex", "ALL", "BYBIT", "FOREX"]] = Query(default=None),
        payload: Optional[CloseAllPositionsRequest] = Body(default=None),
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Close all OPEN paper positions (optionally filtered by market) at live mark price through the Realism Engine."""
        _ensure_bot_state(db, broker)
        resolved_market = (
            (payload.market if payload and payload.market else market) or "ALL"
        ).strip().upper()
        reason = (
            payload.reason.strip()
            if payload and payload.reason and payload.reason.strip()
            else f"Manual MT5 Close All ({resolved_market})"
        )

        q = db.query(Trade).filter(Trade.status == "OPEN")
        if resolved_market in {"BYBIT", "FOREX"}:
            q = q.filter(Trade.market == resolved_market)
        open_trades = q.order_by(Trade.id.asc()).all()

        closed_trades: list[dict[str, Any]] = []
        for trade in open_trades:
            mark_px = broker._prices.get(trade.symbol)
            if mark_px is None or mark_px <= 0:
                mark_px = trade.current_price if trade.current_price else trade.entry_price
            closed = broker.close_position(
                trade_id=trade.id,
                reason=reason,
                db=db,
                order_exit_type="MARKET",
                requested_exit_price=mark_px,
            )
            closed_trades.append(closed)

        if closed_trades:
            balances = broker.get_balance(db=db)
            db.add(
                EquitySnapshot(
                    total_equity=balances["total_equity"],
                    bybit_equity=balances["bybit_equity"],
                    forex_equity=balances["forex_equity"],
                )
            )
            db.commit()

        return {
            "status": "ok",
            "market": resolved_market,
            "closed_count": len(closed_trades),
            "closed_trades": closed_trades,
            "wallet": build_wallet_summary(db),
        }

    @fastapi_app.post("/api/positions/{position_id}/close", dependencies=[Depends(require_dashboard_auth)])
    def close_single_position(
        position_id: int,
        db: Session = Depends(db_dependency),
    ) -> dict[str, Any]:
        """Close a single OPEN paper position at live mark price through the Realism Engine."""
        _ensure_bot_state(db, broker)
        trade = db.query(Trade).filter(Trade.id == position_id).first()
        if trade is None:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found.")
        if trade.status == "CLOSED":
            raise HTTPException(status_code=409, detail=f"Position {position_id} is already closed.")

        mark_px = broker._prices.get(trade.symbol)
        if mark_px is None or mark_px <= 0:
            mark_px = trade.current_price if trade.current_price else trade.entry_price

        closed = broker.close_position(
            trade_id=trade.id,
            reason="Manual MT5 Close from dashboard",
            db=db,
            order_exit_type="MARKET",
            requested_exit_price=mark_px,
        )
        balances = broker.get_balance(db=db)
        db.add(
            EquitySnapshot(
                total_equity=balances["total_equity"],
                bybit_equity=balances["bybit_equity"],
                forex_equity=balances["forex_equity"],
            )
        )
        db.commit()

        return {
            "status": "ok",
            "closed_trade": closed,
            "wallet": build_wallet_summary(db),
        }

    return fastapi_app


app = create_app()
