"""Async background BotRunner managed via FastAPI lifespan."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from app.brokers.bybit_broker import BybitBrokerError
from app.brokers.bybit_router import BybitRouter
from app.brokers.paper_sim import (
    INITIAL_BYBIT_BALANCE,
    INITIAL_FOREX_BALANCE,
    SYMBOL_SPECS,
    PaperSimBroker,
)
from app.config import settings
from app.database import SessionLocal
from app.engine.confluence import ConfluenceFilter
from app.engine.risk_manager import RiskManager
from app.engine.strategies import StrategyDispatcher
from app.models import BotState, StrategyConfig, SystemLog, Trade


class BotRunner:
    """Isolated async heartbeat loop for quant evaluation, risk management, and execution."""

    def __init__(
        self,
        broker: Optional[PaperSimBroker] = None,
        session_factory: Callable[[], Session] = SessionLocal,
        interval_seconds: Optional[int] = None,
        dispatcher: Optional[StrategyDispatcher] = None,
        risk_manager: Optional[RiskManager] = None,
        bybit_router: Optional[BybitRouter] = None,
        confluence_filter: Optional[ConfluenceFilter] = None,
    ) -> None:
        self._session_factory = session_factory
        self.broker = broker or PaperSimBroker(session_factory=session_factory)
        self.bybit_router = bybit_router or BybitRouter(
            paper_broker=self.broker,
            session_factory=session_factory,
        )
        self.interval_seconds = interval_seconds or settings.LOOP_INTERVAL_SECONDS
        self.dispatcher = dispatcher or StrategyDispatcher()
        self.risk_manager = risk_manager or RiskManager(
            max_risk_per_trade_pct=settings.MAX_RISK_PER_TRADE_PCT,
            max_daily_loss_pct=settings.MAX_DAILY_LOSS_PCT,
        )
        self.confluence_filter = confluence_filter or ConfluenceFilter()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._initial_risks: dict[int, float] = {}
        self._tick_count: int = 0
        self._last_loss_tick: dict[str, int] = {}
        self.last_confluence: dict[str, object] = {
            "symbol": "BTCUSDT",
            "score": 0,
            "htf_bias": "UNAVAILABLE",
            "vwap_aligned": False,
            "volume_ratio": 1.0,
            "passed": False,
            "reasons": ["Awaiting first signal"],
        }

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    async def start(self) -> None:
        """Seed initial telemetry and launch the background heartbeat task."""
        db = self._session_factory()
        try:
            self.broker.seed_demo_data(db)
        finally:
            db.close()

        self._stop_event.clear()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Signal the loop to stop and await clean task termination."""
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run_loop(self) -> None:
        """Main background loop with isolated exception handling."""
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - defensive guard
                self._log_error(f"Heartbeat loop error: {exc}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue

    def _engage_kill_switch(self, db: Session, state: BotState, reason: str) -> None:
        """Persist Kill Switch state, close all open positions, and emit CRITICAL log."""
        state.kill_switch_active = True
        state.is_running = False
        state.kill_switch_reason = reason
        for pos in self.broker.get_open_positions(db=db):
            self.broker.close_position(pos["id"], "Daily Loss Limit Kill Switch", db=db)
        db.add(
            SystemLog(
                level="CRITICAL",
                source="RiskGuard",
                message=reason,
            )
        )

    def _apply_trailing_stops(self, db: Session) -> None:
        """Apply RiskManager trailing SL logic to all currently open trades."""
        open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
        for trade in open_trades:
            initial_risk = self._initial_risks.setdefault(
                trade.id,
                abs(float(trade.entry_price) - float(trade.stop_loss)),
            )
            new_sl = self.risk_manager.compute_trailing_sl(
                side=trade.side,
                entry_price=float(trade.entry_price),
                current_price=float(trade.current_price),
                current_sl=float(trade.stop_loss),
                initial_risk=initial_risk,
            )
            decimals = int(SYMBOL_SPECS.get(trade.symbol, {"decimals": 2})["decimals"])
            rounded_sl = round(new_sl, decimals)
            if rounded_sl != round(float(trade.stop_loss), decimals):
                trade.stop_loss = rounded_sl
                if trade.market == "BYBIT" and self.bybit_router.is_demo_active():
                    try:
                        self.bybit_router.get_broker().update_stop_loss(trade.symbol, rounded_sl)
                    except BybitBrokerError as exc:
                        db.add(SystemLog(level="WARN", source="BybitBroker", message=f"Trailing SL sync skipped: {exc}"))
                db.add(
                    SystemLog(
                        level="INFO",
                        source="RiskManager",
                        message=(
                            f"Trailing SL locked for {trade.symbol} #{trade.id} ({trade.side}): "
                            f"SL moved to {rounded_sl}."
                        ),
                    )
                )

    def run_once(self) -> None:
        """Execute a single heartbeat cycle synchronously."""
        db = self._session_factory()
        try:
            state = db.query(BotState).first()
            if state is None:
                self.broker.seed_demo_data(db)
                state = db.query(BotState).first()

            if state is None:
                return

            state.last_heartbeat = self._utcnow()

            if state.kill_switch_active or not state.is_running:
                db.commit()
                return

            self._tick_count += 1
            open_ids_before = {
                row[0] for row in db.query(Trade.id).filter(Trade.status == "OPEN").all()
            }

            balances = self.broker.tick_market(db)

            # Track losing trades closed during this tick for 3-candle anti-revenge cooldown
            if open_ids_before:
                newly_closed_losses = (
                    db.query(Trade)
                    .filter(
                        Trade.id.in_(open_ids_before),
                        Trade.status == "CLOSED",
                        Trade.pnl_usd < 0.0,
                    )
                    .all()
                )
                for closed_trade in newly_closed_losses:
                    self._last_loss_tick[closed_trade.symbol] = self._tick_count

            # Tighten trailing stops on surviving open trades
            self._apply_trailing_stops(db)

            starting_equity = INITIAL_BYBIT_BALANCE + INITIAL_FOREX_BALANCE
            daily_pnl_pct = ((balances["total_equity"] - starting_equity) / starting_equity) * 100.0

            if daily_pnl_pct <= -abs(settings.MAX_DAILY_LOSS_PCT):
                reason = (
                    f"Auto-triggered: daily loss {abs(daily_pnl_pct):.2f}% >= "
                    f"{settings.MAX_DAILY_LOSS_PCT:.2f}% limit."
                )
                self._engage_kill_switch(db, state, reason)
                db.commit()
                return

            enabled_strategies = {
                s.name
                for s in db.query(StrategyConfig).filter(StrategyConfig.enabled.is_(True)).all()
            }

            for symbol, spec in SYMBOL_SPECS.items():
                htf_df = None
                if spec["market"] == "BYBIT" and self.bybit_router.is_demo_active():
                    try:
                        bybit_broker = self.bybit_router.get_broker()
                        df = bybit_broker.get_ohlcv(symbol, interval=settings.BYBIT_PRIMARY_INTERVAL, limit=250)
                        htf_df = bybit_broker.get_ohlcv(symbol, interval=settings.BYBIT_HTF_INTERVAL, limit=100)
                    except BybitBrokerError as exc:
                        db.add(SystemLog(level="WARN", source="BybitBroker", message=f"{symbol} feed fallback: {exc}"))
                        df = self.broker.get_ohlcv(symbol, interval="15m", limit=250)
                else:
                    df = self.broker.get_ohlcv(symbol, interval="15m", limit=250)

                regime_result, signal = self.dispatcher.evaluate(df, enabled_strategies)

                if symbol == "BTCUSDT":
                    state.active_regime_bybit = regime_result.regime.value
                elif symbol == "EUR_USD":
                    state.active_regime_forex = regime_result.regime.value

                if signal.action not in {"BUY", "SELL"}:
                    continue

                conf_result = self.confluence_filter.evaluate(signal, df, htf_df=htf_df)
                self.last_confluence = {
                    "symbol": symbol,
                    "score": conf_result.score,
                    "htf_bias": conf_result.htf_bias,
                    "vwap_aligned": conf_result.vwap_aligned,
                    "volume_ratio": conf_result.volume_ratio,
                    "passed": conf_result.passed,
                    "reasons": conf_result.reasons,
                }
                db.add(
                    SystemLog(
                        level="INFO",
                        source="ConfluenceFilter",
                        message=(
                            f"{symbol} {signal.action} Confluence {conf_result.score}/100 "
                            f"({'PASSED' if conf_result.passed else 'BLOCKED'}): "
                            f"{' | '.join(conf_result.reasons)}"
                        ),
                    )
                )
                if not conf_result.passed:
                    continue

                open_positions = self.broker.get_open_positions(db=db)
                candles_since_loss = (
                    self._tick_count - self._last_loss_tick[symbol]
                    if symbol in self._last_loss_tick
                    else None
                )
                decision = self.risk_manager.approve_trade(
                    side=signal.action,
                    symbol=symbol,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    equity=balances["total_equity"],
                    daily_pnl_pct=daily_pnl_pct,
                    open_positions=open_positions,
                    candles_since_last_loss=candles_since_loss,
                )

                if decision.trigger_kill_switch:
                    self._engage_kill_switch(
                        db,
                        state,
                        decision.rejection_reason or "RiskManager triggered Kill Switch.",
                    )
                    db.commit()
                    return

                if decision.approved:
                    decimals = int(spec["decimals"])
                    order_side = "LONG" if signal.action == "BUY" else "SHORT"
                    if spec["market"] == "BYBIT" and self.bybit_router.is_demo_active():
                        try:
                            self.bybit_router.get_broker().place_order(
                                symbol=symbol,
                                side=order_side,
                                qty=round(decision.qty, 6),
                                stop_loss=round(signal.stop_loss, decimals),
                                take_profit=round(signal.take_profit, decimals),
                                strategy_name=signal.strategy_name,
                                reason=f"{signal.reason} [Confluence {conf_result.score}/100]",
                                entry_price_ref=signal.entry_price,
                            )
                        except BybitBrokerError as exc:
                            db.add(SystemLog(level="ERROR", source="BybitBroker", message=f"Order rejected: {exc}"))
                            continue

                    created = self.broker.place_order(
                        symbol=symbol,
                        side=order_side,
                        qty=round(decision.qty, 6),
                        stop_loss=round(signal.stop_loss, decimals),
                        take_profit=round(signal.take_profit, decimals),
                        strategy_name=signal.strategy_name,
                        reason=f"{signal.reason} [Confluence {conf_result.score}/100]",
                        db=db,
                    )
                    self._initial_risks[created["id"]] = abs(
                        float(created["entry_price"]) - float(created["stop_loss"])
                    )

            db.commit()
        finally:
            db.close()

    def _log_error(self, message: str) -> None:
        """Persist an error log without raising."""
        db = self._session_factory()
        try:
            db.add(SystemLog(level="ERROR", source="BotRunner", message=message))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
