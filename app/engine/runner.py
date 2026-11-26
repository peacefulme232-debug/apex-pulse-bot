"""Async background BotRunner managed via FastAPI lifespan."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from app.brokers.paper_sim import INITIAL_BYBIT_BALANCE, INITIAL_FOREX_BALANCE, PaperSimBroker
from app.config import settings
from app.database import SessionLocal
from app.models import BotState, SystemLog


class BotRunner:
    """Isolated async heartbeat loop for market simulation and risk monitoring."""

    def __init__(
        self,
        broker: Optional[PaperSimBroker] = None,
        session_factory: Callable[[], Session] = SessionLocal,
        interval_seconds: Optional[int] = None,
    ) -> None:
        self._session_factory = session_factory
        self.broker = broker or PaperSimBroker(session_factory=session_factory)
        self.interval_seconds = interval_seconds or settings.LOOP_INTERVAL_SECONDS
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

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

            balances = self.broker.tick_market(db)
            starting_equity = INITIAL_BYBIT_BALANCE + INITIAL_FOREX_BALANCE
            daily_loss_pct = max(
                0.0,
                ((starting_equity - balances["total_equity"]) / starting_equity) * 100.0,
            )

            if daily_loss_pct >= settings.MAX_DAILY_LOSS_PCT:
                state.kill_switch_active = True
                state.is_running = False
                state.kill_switch_reason = (
                    f"Auto-triggered: daily loss {daily_loss_pct:.2f}% >= {settings.MAX_DAILY_LOSS_PCT:.2f}% limit."
                )
                for pos in self.broker.get_open_positions(db=db):
                    self.broker.close_position(pos["id"], "Daily Loss Limit Kill Switch", db=db)
                db.add(
                    SystemLog(
                        level="CRITICAL",
                        source="RiskGuard",
                        message=state.kill_switch_reason,
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
