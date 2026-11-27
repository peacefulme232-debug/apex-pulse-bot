"""Shared signal contract for APEX-PULSE strategy modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class CandidateSignal:
    """Unified trade candidate emitted by any strategy before confluence scoring."""

    symbol: str
    market: str
    strategy: str
    side: Literal["LONG", "SHORT"]
    entry_price: float
    sl_price: float
    tp_price: float
    order_entry_type: str = "MARKET"
    order_exit_type: str = "TAKE_PROFIT"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_distance(self) -> float:
        """Absolute distance between entry and stop loss."""
        return abs(self.entry_price - self.sl_price)

    @property
    def reward_distance(self) -> float:
        """Absolute distance between entry and take profit."""
        return abs(self.tp_price - self.entry_price)

    @property
    def rr_ratio(self) -> float:
        """Reward-to-risk ratio (R multiple)."""
        risk = self.risk_distance
        if risk <= 1e-12:
            return 0.0
        return self.reward_distance / risk

    def to_dict(self) -> dict[str, Any]:
        """Serialize signal to a plain dictionary."""
        return {
            "symbol": self.symbol,
            "market": self.market,
            "strategy": self.strategy,
            "side": self.side,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "order_entry_type": self.order_entry_type,
            "order_exit_type": self.order_exit_type,
            "rr_ratio": round(self.rr_ratio, 4),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }
