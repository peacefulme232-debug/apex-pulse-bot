"""FastAPI application entrypoint and API routes for APEX-PULSE."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker

from app.brokers.paper_sim import INITIAL_BYBIT_BALANCE, INITIAL_FOREX_BALANCE, PaperSimBroker
from app.config import settings
from app.database import SessionLocal, create_db_engine, get_db, init_db
from app.engine.runner import BotRunner
from app.models import BotState, EquitySnapshot, StrategyConfig, SystemLog, Trade

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class StrategyToggleRequest(BaseModel):
    """Payload for toggling a strategy's enabled state."""

    name: str
    enabled: Optional[bool] = None


class KillSwitchRequest(BaseModel):
    """Optional payload for activating the emergency kill switch."""

    reason: Optional[str] = None


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

    def _get_scoped_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    db_dependency = _get_scoped_db if database_url else get_db

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        init_db(target_engine=target_engine)
        await bot_runner.start()
        fastapi_app.state.bot_runner = bot_runner
        fastapi_app.state.broker = broker
        try:
            yield
        finally:
            await bot_runner.stop()

    fastapi_app = FastAPI(
        title="APEX-PULSE",
        description="24/7 Cloud-Ready Multi-Market Algorithmic Trading Bot (Phase 1)",
        version="0.1.0",
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
        return {
            "bybit": bot_runner.bybit_router.get_status(),
            "forex": {
                "status": "PAPER_FALLBACK",
                "mode": "paper",
                "provider": "OANDA (Phase 4)",
                "symbols": ["EUR_USD", "XAU_USD"],
                "last_sync_message": "Paper simulation active (Phase 4 ready)",
            },
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
        """Return service health, active trading mode, and kill switch status."""
        state = _ensure_bot_state(db, broker)
        return {
            "status": "ok",
            "mode": settings.TRADING_MODE,
            "kill_switch": bool(state.kill_switch_active),
        }

    @fastapi_app.get("/api/brokers/status")
    def get_brokers_status() -> dict[str, Any]:
        """Return live connection health for Bybit and Forex brokers."""
        return _build_brokers_status()

    @fastapi_app.get("/api/dashboard")
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

    @fastapi_app.post("/api/strategy/toggle")
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

    @fastapi_app.post("/api/kill-switch")
    def activate_kill_switch(
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
        return {
            "status": "ok",
            "kill_switch_active": True,
            "is_running": False,
            "closed_positions": closed_count,
            "reason": reason,
        }

    @fastapi_app.post("/api/bot/resume")
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

    return fastapi_app


app = create_app()
