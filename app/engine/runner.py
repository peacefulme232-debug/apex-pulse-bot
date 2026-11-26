"""Async background BotRunner managed via FastAPI lifespan."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from app.brokers.bybit_broker import BybitBrokerError
from app.brokers.bybit_router import BybitRouter
from app.brokers.forex_deriv import DerivBrokerError
from app.brokers.forex_oanda import OandaBrokerError
from app.brokers.forex_router import ForexRouter
from app.brokers.paper_sim import (
    INITIAL_BYBIT_BALANCE,
    INITIAL_FOREX_BALANCE,
    SYMBOL_SPECS,
    PaperSimBroker,
)
from app.config import settings
from app.database import SessionLocal
from app.engine.confluence import ConfluenceFilter
from app.engine.recovery import reconcile_startup_state
from app.engine.risk_manager import RiskManager
from app.engine.session_guard import SessionGuard
from app.engine.strategies import StrategyDispatcher
from app.models import BotState, StrategyConfig, SystemLog, Trade
from app.services.notifier import TelegramNotifier

logger = logging.getLogger("apex_pulse.runner")


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
        forex_router: Optional[ForexRouter] = None,
        confluence_filter: Optional[ConfluenceFilter] = None,
        session_guard: Optional[SessionGuard] = None,
        notifier: Optional[TelegramNotifier] = None,
    ) -> None:
        self._session_factory = session_factory
        self.broker = broker or PaperSimBroker(session_factory=session_factory)
        self.bybit_router = bybit_router or BybitRouter(
            paper_broker=self.broker,
            session_factory=session_factory,
        )
        self.forex_router = forex_router or ForexRouter(
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
        self.session_guard = session_guard or SessionGuard()
        self.notifier = notifier or TelegramNotifier()
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._tick_lock = asyncio.Lock()
        self._pending_notifications: list[tuple[str, dict[str, Any]]] = []
        self._initial_risks: dict[int, float] = {}
        self._tick_count: int = 0
        self._last_loss_tick: dict[str, int] = {}
        self.last_recovery_report: Optional[dict[str, Any]] = None
        self.last_confluence: dict[str, object] = {
            "symbol": "BTCUSDT",
            "score": 0,
            "htf_bias": "UNAVAILABLE",
            "vwap_aligned": False,
            "volume_ratio": 1.0,
            "passed": False,
            "reasons": ["Awaiting first signal"],
        }
        self.last_forex_session: dict[str, object] = self.session_guard.evaluate("EUR_USD").to_dict()

    def get_forex_session_status(self, symbol: str = "EUR_USD") -> dict[str, object]:
        """Evaluate and return current Forex session/news guard status."""
        status = self.session_guard.evaluate(symbol).to_dict()
        self.last_forex_session = status
        return status

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    async def start(self) -> None:
        """Seed initial telemetry, reconcile startup state, and optionally launch background loop."""
        db = self._session_factory()
        try:
            self.broker.seed_demo_data(db)
            self.last_recovery_report = reconcile_startup_state(
                db=db,
                bybit_broker=self.bybit_router,
                forex_broker=self.forex_router,
            )
        finally:
            db.close()

        self._stop_event.clear()
        if settings.ENABLE_BACKGROUND_LOOP:
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._run_loop())
        else:
            logger.info(
                "ENABLE_BACKGROUND_LOOP=false: background heartbeat disabled (cron-driven mode)."
            )

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

    async def _dispatch_pending_notifications(self) -> None:
        """Send queued Telegram notifications asynchronously without blocking or raising."""
        if not self._pending_notifications:
            return
        queued = list(self._pending_notifications)
        self._pending_notifications.clear()
        for method_name, kwargs in queued:
            handler = getattr(self.notifier, method_name, None)
            if handler is None:
                continue
            try:
                await handler(**kwargs)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Notification %s failed (non-fatal): %s", method_name, exc)

    def _read_tick_summary(self) -> tuple[bool, dict[str, str]]:
        """Read current kill_switch flag and active regimes for cron/status responses."""
        db = self._session_factory()
        try:
            state = db.query(BotState).first()
            if state is None:
                return False, {"bybit": "UNKNOWN", "forex": "UNKNOWN"}
            return (
                bool(state.kill_switch_active),
                {
                    "bybit": str(state.active_regime_bybit),
                    "forex": str(state.active_regime_forex),
                },
            )
        finally:
            db.close()

    async def run_locked_tick(self) -> dict[str, Any]:
        """Run one mutually-excluded trading tick off the event loop and dispatch alerts."""
        async with self._tick_lock:
            started = time.perf_counter()
            await asyncio.to_thread(self.run_once)
            await self._dispatch_pending_notifications()
            kill_switch, regimes = await asyncio.to_thread(self._read_tick_summary)
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "status": "ok",
                "kill_switch": kill_switch,
                "duration_ms": duration_ms,
                "regimes": regimes,
            }

    async def _run_loop(self) -> None:
        """Main background loop with isolated exception handling and graceful cancellation."""
        try:
            while not self._stop_event.is_set():
                try:
                    await self.run_locked_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._log_error(f"Heartbeat loop error: {exc}")

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            return

    def _engage_kill_switch(self, db: Session, state: BotState, reason: str) -> None:
        """Persist Kill Switch state, close all open positions, and emit CRITICAL log."""
        state.kill_switch_active = True
        state.is_running = False
        state.kill_switch_reason = reason
        open_positions = self.broker.get_open_positions(db=db)
        for pos in open_positions:
            self.broker.close_position(pos["id"], "Daily Loss Limit Kill Switch", db=db)
        db.add(
            SystemLog(
                level="CRITICAL",
                source="RiskGuard",
                message=reason,
            )
        )
        self._pending_notifications.append(
            (
                "notify_kill_switch",
                {
                    "reason": reason,
                    "closed_positions": len(open_positions),
                },
            )
        )

    def _apply_partial_scale_outs(self, db: Session) -> None:
        """Apply +1.0R partial take-profit scale-out to open positions when enabled."""
        if not settings.PARTIAL_TP_ENABLED:
            return

        open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
        for trade in open_trades:
            if trade.partial_tp_taken:
                continue
            initial_risk = self._initial_risks.setdefault(
                trade.id,
                abs(float(trade.entry_price) - float(trade.stop_loss)),
            )
            decision = self.risk_manager.evaluate_partial_scale_out(
                side=trade.side,
                entry_price=float(trade.entry_price),
                current_price=float(trade.current_price),
                current_sl=float(trade.stop_loss),
                initial_risk=initial_risk,
                current_qty=float(trade.qty),
                partial_already_taken=bool(trade.partial_tp_taken),
            )
            if not decision.should_scale_out:
                continue

            decimals = int(SYMBOL_SPECS.get(trade.symbol, {"decimals": 2})["decimals"])
            rounded_sl = round(decision.new_stop_loss, decimals)
            direction = 1.0 if trade.side.upper() == "LONG" else -1.0
            partial_pnl = round(
                (float(trade.current_price) - float(trade.entry_price)) * decision.close_qty * direction,
                2,
            )
            remaining_pnl, _ = self.broker._calc_pnl(
                trade.side,
                float(trade.entry_price),
                float(trade.current_price),
                decision.remaining_qty,
            )

            if trade.initial_qty is None:
                trade.initial_qty = float(trade.qty)
            trade.qty = round(decision.remaining_qty, 6)
            trade.partial_tp_taken = True
            trade.realized_partial_pnl_usd = round(float(trade.realized_partial_pnl_usd or 0.0) + partial_pnl, 2)
            trade.stop_loss = rounded_sl
            trade.pnl_usd = round(trade.realized_partial_pnl_usd + remaining_pnl, 2)

            if trade.market == "BYBIT" and self.bybit_router.is_demo_active():
                try:
                    self.bybit_router.get_broker().update_stop_loss(trade.symbol, rounded_sl)
                except BybitBrokerError as exc:
                    db.add(SystemLog(level="WARN", source="BybitBroker", message=f"Partial scale-out SL sync skipped: {exc}"))

            db.add(
                SystemLog(
                    level="INFO",
                    source="RiskManager",
                    message=(
                        f"+1.0R Partial Scale-Out on {trade.symbol} #{trade.id} ({trade.side}): "
                        f"locked ${partial_pnl:+.2f} on 50% qty, runner SL moved to {rounded_sl}."
                    ),
                )
            )
            self._pending_notifications.append(
                (
                    "notify_partial_scale_out",
                    {
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "trade_id": trade.id,
                        "partial_pnl_usd": partial_pnl,
                        "new_stop_loss": rounded_sl,
                        "remaining_qty": trade.qty,
                    },
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

            # Track trades closed during this tick (alerts + 3-candle anti-revenge cooldown on losses)
            if open_ids_before:
                newly_closed = (
                    db.query(Trade)
                    .filter(
                        Trade.id.in_(open_ids_before),
                        Trade.status == "CLOSED",
                    )
                    .all()
                )
                for closed_trade in newly_closed:
                    if float(closed_trade.pnl_usd) < 0.0:
                        self._last_loss_tick[closed_trade.symbol] = self._tick_count
                    self._pending_notifications.append(
                        (
                            "notify_trade_closed",
                            {
                                "symbol": closed_trade.symbol,
                                "side": closed_trade.side,
                                "market": closed_trade.market,
                                "exit_price": float(
                                    closed_trade.exit_price
                                    if closed_trade.exit_price is not None
                                    else closed_trade.current_price
                                ),
                                "pnl_usd": float(closed_trade.pnl_usd),
                                "pnl_pct": float(closed_trade.pnl_pct),
                                "strategy_name": closed_trade.strategy_name,
                                "reason": closed_trade.entry_reason or "",
                            },
                        )
                    )

            # Lock +1.0R partial scale-out profits, then tighten trailing stops on surviving open trades
            self._apply_partial_scale_outs(db)
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
                elif spec["market"] == "FOREX" and self.forex_router.is_cloud_active():
                    try:
                        fx_broker = self.forex_router.get_broker()
                        df = fx_broker.get_ohlcv(symbol, interval=settings.FOREX_PRIMARY_INTERVAL, limit=250)
                        htf_df = fx_broker.get_ohlcv(symbol, interval=settings.FOREX_HTF_INTERVAL, limit=100)
                    except (OandaBrokerError, DerivBrokerError) as exc:
                        db.add(SystemLog(level="WARN", source="ForexBroker", message=f"{symbol} feed fallback: {exc}"))
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

                if spec["market"] == "FOREX":
                    sess = self.session_guard.evaluate(symbol)
                    self.last_forex_session = sess.to_dict()
                    if not sess.can_trade:
                        db.add(
                            SystemLog(
                                level="INFO",
                                source="SessionGuard",
                                message=f"{symbol} {signal.action} paused ({sess.session_name}): {sess.reason}",
                            )
                        )
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
                    elif spec["market"] == "FOREX" and self.forex_router.is_cloud_active():
                        try:
                            self.forex_router.get_broker().place_order(
                                symbol=symbol,
                                side=order_side,
                                qty=round(decision.qty, 6),
                                stop_loss=round(signal.stop_loss, decimals),
                                take_profit=round(signal.take_profit, decimals),
                                strategy_name=signal.strategy_name,
                                reason=f"{signal.reason} [Confluence {conf_result.score}/100]",
                            )
                        except (OandaBrokerError, DerivBrokerError) as exc:
                            db.add(SystemLog(level="ERROR", source="ForexBroker", message=f"Order rejected: {exc}"))
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
                    self._pending_notifications.append(
                        (
                            "notify_trade_opened",
                            {
                                "symbol": symbol,
                                "side": order_side,
                                "market": spec["market"],
                                "qty": round(decision.qty, 6),
                                "entry_price": float(created["entry_price"]),
                                "stop_loss": round(signal.stop_loss, decimals),
                                "take_profit": round(signal.take_profit, decimals),
                                "strategy_name": signal.strategy_name,
                                "reason": f"{signal.reason} [Confluence {conf_result.score}/100]",
                            },
                        )
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
