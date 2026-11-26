"""FastAPI application entrypoint and API routes for APEX-PULSE."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
from app.models import BotState, EquitySnapshot, StrategyConfig, SystemLog, Trade
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

        if market == "BYBIT":
            selected_equity = balances["bybit_equity"]
            starting_equity = INITIAL_BYBIT_BALANCE
            active_regime = state.active_regime_bybit
        elif market == "FOREX":
            selected_equity = balances["forex_equity"]
            starting_equity = INITIAL_FOREX_BALANCE
            active_regime = state.active_regime_forex
        else:
            selected_equity = balances["total_equity"]
            starting_equity = INITIAL_BYBIT_BALANCE + INITIAL_FOREX_BALANCE
            active_regime = f"{state.active_regime_bybit} / {state.active_regime_forex}"

        daily_pnl_usd = round(selected_equity - starting_equity, 2)
        daily_pnl_pct = round((daily_pnl_usd / starting_equity) * 100.0, 2) if starting_equity else 0.0
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

        equity_curve = []
        for snap in snapshots:
            if market == "BYBIT":
                val = snap.bybit_equity
            elif market == "FOREX":
                val = snap.forex_equity
            else:
                val = snap.total_equity
            equity_curve.append(
                {
                    "timestamp": snap.timestamp.isoformat(),
                    "total_equity": round(snap.total_equity, 2),
                    "bybit_equity": round(snap.bybit_equity, 2),
                    "forex_equity": round(snap.forex_equity, 2),
                    "value": round(val, 2),
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

    return fastapi_app


app = create_app()
