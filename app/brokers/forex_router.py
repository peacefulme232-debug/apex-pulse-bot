"""ForexRouter: safely selects OandaForexBroker, DerivForexBroker, or PaperSimBroker fallback."""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from sqlalchemy.orm import Session

from app.brokers.forex_deriv import DerivForexBroker
from app.brokers.forex_oanda import OandaForexBroker
from app.brokers.paper_sim import PaperSimBroker
from app.config import settings
from app.database import SessionLocal

ForexBrokerType = Union[OandaForexBroker, DerivForexBroker, PaperSimBroker]


class ForexRouter:
    """Routes Forex execution to OANDA Practice, Deriv Demo, or PaperSimBroker fallback."""

    def __init__(
        self,
        paper_broker: Optional[PaperSimBroker] = None,
        session_factory: Callable[[], Session] = SessionLocal,
        provider: Optional[str] = None,
        oanda_api_key: Optional[str] = None,
        oanda_account_id: Optional[str] = None,
        oanda_base_url: Optional[str] = None,
        deriv_api_token: Optional[str] = None,
        deriv_app_id: Optional[str] = None,
        deriv_account_id: Optional[str] = None,
        deriv_auth_mode: Optional[str] = None,
    ) -> None:
        self.paper_broker = paper_broker or PaperSimBroker(session_factory=session_factory)
        self._provider = (settings.FOREX_PROVIDER if provider is None else provider).strip().lower()
        self._oanda_api_key = (settings.OANDA_API_KEY if oanda_api_key is None else oanda_api_key).strip()
        self._oanda_account_id = (
            settings.OANDA_ACCOUNT_ID if oanda_account_id is None else oanda_account_id
        ).strip()
        self._oanda_base_url = (
            settings.OANDA_BASE_URL if oanda_base_url is None else oanda_base_url
        ).strip()
        self._deriv_api_token = (
            settings.DERIV_API_TOKEN if deriv_api_token is None else deriv_api_token
        ).strip()
        self._deriv_app_id = (settings.DERIV_APP_ID if deriv_app_id is None else deriv_app_id).strip()
        self._deriv_account_id = (
            settings.DERIV_ACCOUNT_ID if deriv_account_id is None else deriv_account_id
        ).strip()
        self._deriv_auth_mode = (
            settings.DERIV_AUTH_MODE if deriv_auth_mode is None else deriv_auth_mode
        ).strip().lower()

        self.live_broker: Optional[Union[OandaForexBroker, DerivForexBroker]] = None
        self.active_provider: str = "paper"
        self.status: str = "PAPER_FALLBACK"
        self.last_sync_message: str = "FOREX_PROVIDER=paper; using PaperSimBroker"

        self.active_broker: ForexBrokerType = self._resolve()

    def _resolve(self) -> ForexBrokerType:
        if self._provider == "oanda":
            if self._oanda_api_key and self._oanda_account_id:
                try:
                    broker = OandaForexBroker(
                        api_key=self._oanda_api_key,
                        account_id=self._oanda_account_id,
                        base_url=self._oanda_base_url,
                    )
                    self.live_broker = broker
                    self.active_provider = "oanda"
                    self.status = broker.status
                    self.last_sync_message = broker.last_sync_message
                    return broker
                except Exception as exc:
                    self.status = "PAPER_FALLBACK"
                    self.active_provider = "paper"
                    self.last_sync_message = f"OANDA init failed, using Paper fallback: {exc}"
                    return self.paper_broker

            self.status = "PAPER_FALLBACK"
            self.active_provider = "paper"
            self.last_sync_message = "OANDA keys blank; safely using PaperSimBroker"
            return self.paper_broker

        if self._provider == "deriv":
            if self._deriv_api_token:
                try:
                    broker = DerivForexBroker(
                        api_token=self._deriv_api_token,
                        app_id=self._deriv_app_id,
                        account_id=self._deriv_account_id,
                        auth_mode=self._deriv_auth_mode,
                    )
                    self.live_broker = broker
                    self.active_provider = "deriv"
                    self.status = broker.status
                    self.last_sync_message = broker.last_sync_message
                    return broker
                except Exception as exc:
                    self.status = "PAPER_FALLBACK"
                    self.active_provider = "paper"
                    self.last_sync_message = f"Deriv init failed, using Paper fallback: {exc}"
                    return self.paper_broker

            self.status = "PAPER_FALLBACK"
            self.active_provider = "paper"
            self.last_sync_message = "Deriv token blank; safely using PaperSimBroker"
            return self.paper_broker

        self.status = "PAPER_FALLBACK"
        self.active_provider = "paper"
        self.last_sync_message = "FOREX_PROVIDER=paper; using PaperSimBroker"
        return self.paper_broker

    def get_broker(self) -> ForexBrokerType:
        """Return the active Forex broker instance."""
        return self.active_broker

    def is_cloud_active(self) -> bool:
        """Return True when a cloud Demo/Practice Forex adapter (OANDA or Deriv) is active."""
        return self.live_broker is not None and self.active_broker is self.live_broker

    def get_status(self) -> dict[str, Any]:
        """Return structured Forex broker health telemetry."""
        current_status = self.live_broker.status if self.live_broker is not None else self.status
        current_msg = (
            self.live_broker.last_sync_message
            if self.live_broker is not None
            else self.last_sync_message
        )
        symbols = [s.strip() for s in settings.FOREX_SYMBOLS.split(",") if s.strip()]
        mode_label = "demo" if self.is_cloud_active() else "paper"
        provider_label = {
            "oanda": "OANDA Practice",
            "deriv": "Deriv Demo",
            "paper": "Paper Simulation",
        }.get(self.active_provider, self.active_provider)

        return {
            "status": current_status,
            "mode": mode_label,
            "provider": provider_label,
            "provider_key": self.active_provider,
            "symbols": symbols,
            "killzone_only": settings.FOREX_KILLZONE_ONLY,
            "news_blackout_minutes": settings.NEWS_BLACKOUT_MINUTES,
            "last_sync_message": current_msg,
        }
