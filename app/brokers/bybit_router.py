"""BybitRouter: safely selects BybitBroker (demo) or falls back to PaperSimBroker."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from sqlalchemy.orm import Session

from app.brokers.bybit_broker import BybitBroker
from app.brokers.paper_sim import PaperSimBroker
from app.config import settings
from app.database import SessionLocal


class BybitRouter:
    """Routes Bybit execution to BybitBroker in demo mode or PaperSimBroker fallback."""

    def __init__(
        self,
        paper_broker: Optional[PaperSimBroker] = None,
        session_factory: Callable[[], Session] = SessionLocal,
        bybit_mode: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        bybit_session: Optional[Any] = None,
    ) -> None:
        self.paper_broker = paper_broker or PaperSimBroker(session_factory=session_factory)
        self._mode = (settings.BYBIT_MODE if bybit_mode is None else bybit_mode).strip().lower()
        self._api_key = (settings.BYBIT_API_KEY if api_key is None else api_key).strip()
        self._api_secret = (settings.BYBIT_API_SECRET if api_secret is None else api_secret).strip()
        self._bybit_session = bybit_session
        self.bybit_broker: Optional[BybitBroker] = None
        self.status: str = "PAPER_FALLBACK"
        self.last_sync_message: str = "Using PaperSimBroker (BYBIT_MODE=paper or keys blank)"

        self.active_broker: Union[BybitBroker, PaperSimBroker] = self._resolve()

    def _resolve(self) -> Union[BybitBroker, PaperSimBroker]:
        if self._mode == "demo" and self._api_key and self._api_secret:
            try:
                broker = BybitBroker(
                    api_key=self._api_key,
                    api_secret=self._api_secret,
                    mode="demo",
                    session=self._bybit_session,
                )
                self.bybit_broker = broker
                self.status = broker.status
                self.last_sync_message = broker.last_sync_message
                return broker
            except Exception as exc:
                self.status = "ERROR"
                self.last_sync_message = f"Bybit init failed, using Paper fallback: {exc}"
                return self.paper_broker

        self.status = "PAPER_FALLBACK"
        if self._mode == "demo" and (not self._api_key or not self._api_secret):
            self.last_sync_message = "Bybit keys blank; safely using PaperSimBroker"
        else:
            self.last_sync_message = "BYBIT_MODE=paper; using PaperSimBroker"
        return self.paper_broker

    def get_broker(self) -> Union[BybitBroker, PaperSimBroker]:
        """Return the active broker instance."""
        return self.active_broker

    def is_demo_active(self) -> bool:
        """Return True when live Bybit Demo REST broker is active."""
        return self.bybit_broker is not None and self.active_broker is self.bybit_broker

    def get_status(self) -> dict[str, Any]:
        """Return structured Bybit broker health telemetry."""
        current_status = self.bybit_broker.status if self.bybit_broker is not None else self.status
        current_msg = (
            self.bybit_broker.last_sync_message if self.bybit_broker is not None else self.last_sync_message
        )
        symbols = [s.strip() for s in settings.BYBIT_SYMBOLS.split(",") if s.strip()]
        return {
            "status": current_status,
            "mode": "demo" if self.is_demo_active() else "paper",
            "category": settings.BYBIT_CATEGORY,
            "symbols": symbols,
            "leverage_cap": settings.BYBIT_LEVERAGE,
            "last_sync_message": current_msg,
        }
