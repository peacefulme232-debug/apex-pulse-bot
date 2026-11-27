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
from app.models import ArmedRetestSetup, BotState, StrategyConfig, SystemLog, Trade, WalletAccount
from app.services.confluence_evaluator import BrainConfig, ConfluenceEvaluator
from app.services.notifier import TelegramNotifier
from app.services.regime_router import RegimeRouter
from app.services.strategies import (
    BreakoutRetestStrategy,
    CandidateSignal,
    EmaPullbackStrategy,
    RsiDivergenceSniperStrategy,
)
from app.services.trade_manager import TradeManager

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
        brain_config: Optional[BrainConfig] = None,
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
        self.brain_config = brain_config or BrainConfig()
        self.regime_router = RegimeRouter()
        self.ema_pullback = EmaPullbackStrategy()
        self.breakout_retest = BreakoutRetestStrategy()
        self.rsi_divergence_sniper = RsiDivergenceSniperStrategy()
        self.confluence_evaluator = ConfluenceEvaluator(
            config=self.brain_config,
            divergence_detector=self.rsi_divergence_sniper,
        )
        self.trade_manager = TradeManager(config=self.brain_config)
        self.brain_telemetry: dict[str, dict[str, Any]] = {}
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

            # Credit realized partial profit to wallet active_balance so close_position delta stays exact
            from app.services.realism_engine import apply_negative_balance_protection, calculate_net_pnl, q6, to_decimal

            wallet = self.broker.get_wallet_account(db, trade.market)
            wallet.active_balance = apply_negative_balance_protection(wallet.active_balance, partial_pnl)
            trade.gross_pnl = q6(to_decimal(trade.realized_partial_pnl_usd) + to_decimal(remaining_pnl))
            trade.net_pnl = calculate_net_pnl(
                trade.gross_pnl,
                trade.entry_fee,
                trade.exit_fee,
                trade.funding_or_swap_fee,
            )

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

    def _fetch_brain_candles(
        self,
        db: Session,
        symbol: str,
        market: str,
    ) -> tuple[Any, Any, Optional[Any]]:
        """Fetch closed H1, H4, and optional D1 OHLCV candles for Strategy Brain evaluation."""
        h1_df: Any = None
        h4_df: Any = None
        d1_df: Any = None
        if market == "BYBIT" and self.bybit_router.is_demo_active():
            try:
                bybit_broker = self.bybit_router.get_broker()
                h1_df = bybit_broker.get_ohlcv(symbol, interval="60", limit=250)
                h4_df = bybit_broker.get_ohlcv(symbol, interval="240", limit=250)
            except BybitBrokerError as exc:
                db.add(SystemLog(level="WARN", source="BybitBroker", message=f"{symbol} feed fallback: {exc}"))
        elif market == "FOREX" and self.forex_router.is_cloud_active():
            try:
                fx_broker = self.forex_router.get_broker()
                h1_df = fx_broker.get_ohlcv(symbol, interval="H1", limit=250)
                h4_df = fx_broker.get_ohlcv(symbol, interval="H4", limit=250)
            except (OandaBrokerError, DerivBrokerError) as exc:
                db.add(SystemLog(level="WARN", source="ForexBroker", message=f"{symbol} feed fallback: {exc}"))

        if h1_df is None:
            h1_df = self.broker.get_ohlcv(symbol, interval="1h", limit=250)
        if h4_df is None:
            h4_df = self.broker.get_ohlcv(symbol, interval="4h", limit=250)
        return h1_df, h4_df, d1_df

    def _evaluate_brain_strategies(
        self,
        *,
        db: Session,
        symbol: str,
        market_lower: str,
        regime_decision: Any,
        h1_candles: Any,
        h4_candles: Any,
        d1_candles: Optional[Any],
    ) -> Optional[CandidateSignal]:
        """Run allowed strategy for the classified regime (and advance any armed retest setup)."""
        candidate: Optional[CandidateSignal] = None

        # Always advance an already-armed Breakout+Retest setup so it decrements/triggers/invalidates
        if self.brain_config.is_strategy_enabled("BREAKOUT_RETEST"):
            existing_armed = self.breakout_retest.get_armed_setup(
                db, symbol=symbol, market=market_lower
            )
            if existing_armed is not None:
                candidate = self.breakout_retest.evaluate(
                    db=db,
                    symbol=symbol,
                    market=market_lower,
                    candles=h1_candles,
                )
                if candidate is not None:
                    return candidate

        for strategy_name in regime_decision.allowed_strategies:
            if not self.brain_config.is_strategy_enabled(strategy_name):
                continue
            if strategy_name == "EMA_50_PULLBACK":
                candidate = self.ema_pullback.evaluate(
                    symbol=symbol,
                    market=market_lower,
                    regime=regime_decision,
                    h1_candles=h1_candles,
                    h4_candles=h4_candles,
                )
            elif strategy_name == "BREAKOUT_RETEST":
                candidate = self.breakout_retest.evaluate(
                    db=db,
                    symbol=symbol,
                    market=market_lower,
                    candles=h1_candles,
                )
            elif strategy_name == "RSI_DIVERGENCE_SNIPER":
                htf_source = d1_candles if d1_candles is not None else h4_candles
                candidate = self.rsi_divergence_sniper.evaluate_sniper_entry(
                    candles_htf=htf_source,
                    candles_exec=h1_candles,
                    symbol=symbol,
                    market=market_lower,
                )
            if candidate is not None:
                return candidate

        return None

    def get_brain_telemetry(self, db: Session) -> dict[str, Any]:
        """Build live Strategy Brain telemetry for GET /api/brain/telemetry."""
        from app.services.realism_engine import to_decimal
        from app.services.wallet_service import compute_daily_pnl_isolated, lock_wallet_account

        now = self._utcnow()
        armed_rows = (
            db.query(ArmedRetestSetup)
            .filter(ArmedRetestSetup.status == "ARMED")
            .order_by(ArmedRetestSetup.id.desc())
            .all()
        )
        armed_list = [
            {
                "id": row.id,
                "market": row.market,
                "symbol": row.symbol,
                "side": row.side,
                "zone_low": float(row.zone_low),
                "zone_high": float(row.zone_high),
                "breakout_candle_range": float(row.breakout_candle_range),
                "breakout_atr": float(row.breakout_atr),
                "bars_remaining": int(row.bars_remaining),
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in armed_rows
        ]

        circuit_breaker: dict[str, Any] = {}
        for mkt in ("bybit", "forex"):
            w = lock_wallet_account(db, mkt)
            until = w.circuit_breaker_until
            if until is not None and until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            is_locked = bool(until is not None and now < until)
            _d_usd, d_pct = compute_daily_pnl_isolated(
                current_active_equity=to_decimal(w.active_balance),
                equity_baseline_day=to_decimal(w.equity_baseline_day),
                fallback_capital=to_decimal(w.active_balance),
            )
            circuit_breaker[mkt] = {
                "consecutive_losses": int(w.consecutive_losses or 0),
                "max_consecutive_losses": self.brain_config.max_consecutive_losses,
                "circuit_breaker_until": until.isoformat() if until else None,
                "is_locked": is_locked,
                "daily_pnl_pct": float(d_pct),
                "micro_shield_active": bool(
                    self.brain_config.micro_shield_enabled
                    and float(w.active_balance) <= self.brain_config.micro_shield_threshold_usd
                ),
            }

        active_divergences = [
            {
                "symbol": sym,
                **snap["active_divergence"],
            }
            for sym, snap in self.brain_telemetry.items()
            if snap.get("active_divergence") is not None
        ]

        return {
            "config": self.brain_config.to_dict(),
            "symbols": dict(self.brain_telemetry),
            "armed_retest_setups": armed_list,
            "active_divergences": active_divergences,
            "circuit_breaker": circuit_breaker,
        }

    def update_brain_config(self, db: Session, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply PATCH /api/brain/config updates and optional circuit-breaker reset."""
        from app.services.wallet_service import lock_wallet_account

        reset_breaker = bool(updates.pop("reset_circuit_breaker", False))

        for field_name, value in updates.items():
            if value is not None and hasattr(self.brain_config, field_name):
                setattr(self.brain_config, field_name, value)

        # Keep matching StrategyConfig rows in sync if present
        strategy_flag_map = {
            "EMA_50_PULLBACK": self.brain_config.ema_50_pullback_enabled,
            "BREAKOUT_RETEST": self.brain_config.breakout_retest_enabled,
            "RSI_DIVERGENCE_SNIPER": self.brain_config.rsi_divergence_sniper_enabled,
        }
        for strat_name, enabled_flag in strategy_flag_map.items():
            row = db.query(StrategyConfig).filter(StrategyConfig.name == strat_name).first()
            if row is not None:
                row.enabled = bool(enabled_flag)

        if reset_breaker:
            for mkt in ("bybit", "forex"):
                w = lock_wallet_account(db, mkt)
                w.consecutive_losses = 0
                w.circuit_breaker_until = None
            db.add(
                SystemLog(
                    level="INFO",
                    source="BrainConfig",
                    message="Circuit breaker manually reset via /api/brain/config.",
                )
            )

        db.commit()
        return self.get_brain_telemetry(db)

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

            # Track trades closed during this tick (TradeManager consecutive losses, circuit breaker & auto-sweep)
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
                    self.trade_manager.record_trade_close(db, closed_trade)
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

            # Smart Trade Manager: +1.0R Breakeven + Fees Shield on surviving open trades
            symbol_decimals = {sym: int(spec["decimals"]) for sym, spec in SYMBOL_SPECS.items()}
            be_updates = self.trade_manager.manage_open_trades(
                db,
                initial_risks=self._initial_risks,
                symbol_decimals=symbol_decimals,
            )
            if be_updates and self.bybit_router.is_demo_active():
                for upd in be_updates:
                    if SYMBOL_SPECS.get(upd.symbol, {}).get("market") == "BYBIT":
                        try:
                            self.bybit_router.get_broker().update_stop_loss(
                                upd.symbol, upd.new_stop_loss
                            )
                        except BybitBrokerError as exc:
                            db.add(
                                SystemLog(
                                    level="WARN",
                                    source="BybitBroker",
                                    message=f"Breakeven SL sync skipped: {exc}",
                                )
                            )

            # Lock +1.0R partial scale-out profits, then tighten trailing stops on surviving open trades
            self._apply_partial_scale_outs(db)
            self._apply_trailing_stops(db)

            from app.services.realism_engine import q6, to_decimal
            from app.services.wallet_service import compute_daily_pnl_isolated

            bybit_wallet = self.broker.get_wallet_account(db, "bybit")
            forex_wallet = self.broker.get_wallet_account(db, "forex")
            total_baseline = q6(
                to_decimal(bybit_wallet.equity_baseline_day)
                + to_decimal(forex_wallet.equity_baseline_day)
            )
            total_active = q6(
                to_decimal(bybit_wallet.active_balance) + to_decimal(forex_wallet.active_balance)
            )
            _daily_usd, daily_pnl_pct_dec = compute_daily_pnl_isolated(
                current_active_equity=to_decimal(balances["total_equity"]),
                equity_baseline_day=total_baseline,
                fallback_capital=total_active,
            )
            daily_pnl_pct = float(daily_pnl_pct_dec)

            if daily_pnl_pct <= -abs(settings.MAX_DAILY_LOSS_PCT):
                reason = (
                    f"Auto-triggered: daily loss {abs(daily_pnl_pct):.2f}% >= "
                    f"{settings.MAX_DAILY_LOSS_PCT:.2f}% limit."
                )
                self._engage_kill_switch(db, state, reason)
                db.commit()
                return

            for symbol, spec in SYMBOL_SPECS.items():
                market_upper = str(spec["market"]).upper()
                market_lower = market_upper.lower()
                h1_df, h4_df, d1_df = self._fetch_brain_candles(db, symbol, market_upper)

                # 1. RegimeRouter (zero-repaint mutually exclusive regime classification)
                regime_decision = self.regime_router.classify(
                    symbol=symbol,
                    h1_candles=h1_df,
                    h4_candles=h4_df,
                    d1_candles=d1_df,
                )

                if symbol == "BTCUSDT":
                    state.active_regime_bybit = regime_decision.state.value
                elif symbol == "EUR_USD":
                    state.active_regime_forex = regime_decision.state.value

                # Active H4/D1 RSI divergence snapshot for telemetry
                active_div = (
                    self.rsi_divergence_sniper.detect_active_divergence(d1_df)
                    if d1_df is not None
                    else None
                )
                if active_div is None:
                    active_div = self.rsi_divergence_sniper.detect_active_divergence(h4_df)

                # 2. Allowed Strategy evaluation
                candidate = self._evaluate_brain_strategies(
                    db=db,
                    symbol=symbol,
                    market_lower=market_lower,
                    regime_decision=regime_decision,
                    h1_candles=h1_df,
                    h4_candles=h4_df,
                    d1_candles=d1_df,
                )

                symbol_telemetry: dict[str, Any] = {
                    "symbol": symbol,
                    "market": market_lower,
                    "regime": regime_decision.state.value,
                    "bias": regime_decision.bias,
                    "allowed_strategies": list(regime_decision.allowed_strategies),
                    "veto_longs": regime_decision.veto_longs,
                    "veto_shorts": regime_decision.veto_shorts,
                    "regime_reason": regime_decision.reason,
                    "active_divergence": active_div.to_dict() if active_div is not None else None,
                    "confluence": None,
                }

                if candidate is None:
                    self.brain_telemetry[symbol] = symbol_telemetry
                    continue

                if market_upper == "FOREX":
                    sess = self.session_guard.evaluate(symbol)
                    self.last_forex_session = sess.to_dict()
                    if not sess.can_trade:
                        db.add(
                            SystemLog(
                                level="INFO",
                                source="SessionGuard",
                                message=f"{symbol} {candidate.side} paused ({sess.session_name}): {sess.reason}",
                            )
                        )
                        self.brain_telemetry[symbol] = symbol_telemetry
                        continue

                # 3. 0-100 Confluence Evaluator + 6 Hard Veto Gates
                eval_result = self.confluence_evaluator.evaluate(
                    candidate,
                    db=db,
                    h1_candles=h1_df,
                    h4_candles=h4_df,
                    d1_candles=d1_df,
                    regime_decision=regime_decision,
                    active_divergence=active_div,
                )
                symbol_telemetry["confluence"] = eval_result.to_dict()
                self.brain_telemetry[symbol] = symbol_telemetry
                self.last_confluence = {
                    "symbol": symbol,
                    "score": eval_result.total_score,
                    "htf_bias": regime_decision.bias,
                    "vwap_aligned": eval_result.breakdown.level_precision >= 25,
                    "volume_ratio": eval_result.telemetry.get("volume_ratio") or 1.0,
                    "passed": eval_result.approved,
                    "reasons": [eval_result.details],
                    "breakdown": eval_result.breakdown.to_dict(),
                    "vetoes": list(eval_result.vetoes),
                }
                db.add(
                    SystemLog(
                        level="INFO" if eval_result.approved else "WARN",
                        source="ConfluenceEvaluator",
                        message=(
                            f"{symbol} {candidate.strategy} {candidate.side} "
                            f"{eval_result.total_score}/100 "
                            f"({'APPROVED' if eval_result.approved else 'VETOED'}): "
                            f"{eval_result.details}"
                        ),
                    )
                )
                if not eval_result.approved:
                    continue

                # 4. Realism Sizing & Execution
                market_wallet = self.broker.get_wallet_account(db, market_upper)
                decimals = int(spec["decimals"])
                order_side = candidate.side
                final_qty = 0.0

                if market_wallet.realism_enabled:
                    sizing = self.broker.size_and_validate_order(
                        db=db,
                        symbol=symbol,
                        side=order_side,
                        entry_price=candidate.entry_price,
                        stop_loss=candidate.sl_price,
                        max_risk_per_trade_pct=self.risk_manager.max_risk_per_trade_pct,
                        order_entry_type=candidate.order_entry_type,
                        strategy=candidate.strategy,
                        confluence_score=eval_result.total_score,
                    )
                    if not sizing.approved:
                        db.add(
                            SystemLog(
                                level="WARN",
                                source="RealismEngine",
                                message=(
                                    f"Order rejected for {symbol} ({order_side}) "
                                    f"[{sizing.reason_code}]: {sizing.details}"
                                ),
                            )
                        )
                        continue
                    final_qty = float(sizing.qty)
                else:
                    open_positions = self.broker.get_open_positions(db=db)
                    decision = self.risk_manager.approve_trade(
                        side="BUY" if order_side == "LONG" else "SELL",
                        symbol=symbol,
                        entry_price=candidate.entry_price,
                        stop_loss=candidate.sl_price,
                        take_profit=candidate.tp_price,
                        equity=float(market_wallet.active_balance),
                        daily_pnl_pct=daily_pnl_pct,
                        open_positions=open_positions,
                    )
                    if not decision.approved:
                        continue
                    final_qty = round(decision.qty, 6)

                if market_upper == "BYBIT" and self.bybit_router.is_demo_active():
                    try:
                        self.bybit_router.get_broker().place_order(
                            symbol=symbol,
                            side=order_side,
                            qty=final_qty,
                            stop_loss=round(candidate.sl_price, decimals),
                            take_profit=round(candidate.tp_price, decimals),
                            strategy_name=candidate.strategy,
                            reason=f"{candidate.reason} [Confluence {eval_result.total_score}/100]",
                            entry_price_ref=candidate.entry_price,
                        )
                    except BybitBrokerError as exc:
                        db.add(SystemLog(level="ERROR", source="BybitBroker", message=f"Order rejected: {exc}"))
                        continue
                elif market_upper == "FOREX" and self.forex_router.is_cloud_active():
                    try:
                        self.forex_router.get_broker().place_order(
                            symbol=symbol,
                            side=order_side,
                            qty=final_qty,
                            stop_loss=round(candidate.sl_price, decimals),
                            take_profit=round(candidate.tp_price, decimals),
                            strategy_name=candidate.strategy,
                            reason=f"{candidate.reason} [Confluence {eval_result.total_score}/100]",
                        )
                    except (OandaBrokerError, DerivBrokerError) as exc:
                        db.add(SystemLog(level="ERROR", source="ForexBroker", message=f"Order rejected: {exc}"))
                        continue

                created = self.broker.place_order(
                    symbol=symbol,
                    side=order_side,
                    qty=final_qty,
                    stop_loss=round(candidate.sl_price, decimals),
                    take_profit=round(candidate.tp_price, decimals),
                    strategy_name=candidate.strategy,
                    reason=f"{candidate.reason} [Confluence {eval_result.total_score}/100]",
                    db=db,
                    order_entry_type=candidate.order_entry_type,
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
                            "market": market_upper,
                            "qty": final_qty,
                            "entry_price": float(created["entry_price"]),
                            "stop_loss": round(candidate.sl_price, decimals),
                            "take_profit": round(candidate.tp_price, decimals),
                            "strategy_name": candidate.strategy,
                            "reason": f"{candidate.reason} [Confluence {eval_result.total_score}/100]",
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
