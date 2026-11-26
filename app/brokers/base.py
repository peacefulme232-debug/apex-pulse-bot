"""Abstract BaseBroker interface for multi-market execution."""

from abc import ABC, abstractmethod
from typing import Any


class BaseBroker(ABC):
    """Abstract interface that all market brokers (Paper, Bybit, Forex) must implement."""

    @abstractmethod
    def get_balance(self) -> dict[str, float]:
        """Return current balances per market and total equity."""

    @abstractmethod
    def get_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 50) -> list[dict[str, Any]]:
        """Return recent OHLCV candles for a given symbol and timeframe."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str,
        reason: str,
    ) -> dict[str, Any]:
        """Open a new position and return the created trade representation."""

    @abstractmethod
    def close_position(self, trade_id: int, reason: str) -> dict[str, Any]:
        """Close an open position by trade ID and return the closed trade representation."""

    @abstractmethod
    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return all currently open positions."""
