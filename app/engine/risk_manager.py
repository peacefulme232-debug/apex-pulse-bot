"""Non-negotiable RiskManager survival layer: drawdown kill switch, sizing, and trailing SL."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class RiskDecision:
    """Typed approval or rejection output from RiskManager."""

    approved: bool
    qty: float
    rejection_reason: str | None
    risk_usd: float
    rr_ratio: float
    trigger_kill_switch: bool = False


def _is_finite_positive(value: float | None) -> bool:
    return (
        value is not None
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _normalize_side(side: str) -> str | None:
    cleaned = (side or "").strip().upper()
    if cleaned in {"BUY", "LONG"}:
        return "LONG"
    if cleaned in {"SELL", "SHORT"}:
        return "SHORT"
    return None


class RiskManager:
    """Enforces drawdown circuit breaker, SL/TP sanity, exposure caps, cooldowns, and sizing."""

    def __init__(
        self,
        max_risk_per_trade_pct: float = 1.0,
        max_daily_loss_pct: float = 3.0,
        max_open_trades: int = 3,
        max_open_per_symbol: int = 1,
        min_rr_ratio: float = 1.5,
        max_notional_pct: float = 25.0,
        loss_cooldown_candles: int = 3,
    ) -> None:
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_trades = max_open_trades
        self.max_open_per_symbol = max_open_per_symbol
        self.min_rr_ratio = min_rr_ratio
        self.max_notional_pct = max_notional_pct
        self.loss_cooldown_candles = loss_cooldown_candles

    def approve_trade(
        self,
        side: str,
        symbol: str,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        equity: float,
        daily_pnl_pct: float,
        open_positions: Sequence[dict[str, Any]],
        candles_since_last_loss: int | None = None,
    ) -> RiskDecision:
        """Validate a proposed trade against all survival rules and compute safe position size."""
        # 1. Daily Drawdown Circuit Breaker FIRST
        if not isinstance(daily_pnl_pct, (int, float)) or not math.isfinite(float(daily_pnl_pct)):
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason="Invalid daily_pnl_pct telemetry.",
                risk_usd=0.0,
                rr_ratio=0.0,
                trigger_kill_switch=False,
            )

        if float(daily_pnl_pct) <= -abs(self.max_daily_loss_pct):
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason=(
                    f"CRITICAL: Daily drawdown {daily_pnl_pct:.2f}% breached "
                    f"-{abs(self.max_daily_loss_pct):.2f}% circuit breaker. Engaging Kill Switch."
                ),
                risk_usd=0.0,
                rr_ratio=0.0,
                trigger_kill_switch=True,
            )

        # 2. Equity & Side Validation
        if not _is_finite_positive(equity):
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason="Account equity must be positive and finite.",
                risk_usd=0.0,
                rr_ratio=0.0,
            )

        norm_side = _normalize_side(side)
        if norm_side is None:
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason=f"Unsupported trade side '{side}'.",
                risk_usd=0.0,
                rr_ratio=0.0,
            )

        # 3. SL/TP Presence, Sanity, Direction, and R:R Validation
        if not _is_finite_positive(entry_price):
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason="Missing or non-positive entry_price.",
                risk_usd=0.0,
                rr_ratio=0.0,
            )

        if not _is_finite_positive(stop_loss):
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason="Missing, zero, or NaN stop_loss.",
                risk_usd=0.0,
                rr_ratio=0.0,
            )

        if not _is_finite_positive(take_profit):
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason="Missing, zero, or NaN take_profit.",
                risk_usd=0.0,
                rr_ratio=0.0,
            )

        entry = float(entry_price)
        sl = float(stop_loss)  # type: ignore[arg-type]
        tp = float(take_profit)  # type: ignore[arg-type]

        if norm_side == "LONG":
            if sl >= entry:
                return RiskDecision(
                    approved=False,
                    qty=0.0,
                    rejection_reason=f"Invalid LONG stop_loss ({sl}) >= entry_price ({entry}).",
                    risk_usd=0.0,
                    rr_ratio=0.0,
                )
            if tp <= entry:
                return RiskDecision(
                    approved=False,
                    qty=0.0,
                    rejection_reason=f"Invalid LONG take_profit ({tp}) <= entry_price ({entry}).",
                    risk_usd=0.0,
                    rr_ratio=0.0,
                )
        else:
            if sl <= entry:
                return RiskDecision(
                    approved=False,
                    qty=0.0,
                    rejection_reason=f"Invalid SHORT stop_loss ({sl}) <= entry_price ({entry}).",
                    risk_usd=0.0,
                    rr_ratio=0.0,
                )
            if tp >= entry:
                return RiskDecision(
                    approved=False,
                    qty=0.0,
                    rejection_reason=f"Invalid SHORT take_profit ({tp}) >= entry_price ({entry}).",
                    risk_usd=0.0,
                    rr_ratio=0.0,
                )

        stop_distance = abs(entry - sl)
        if stop_distance <= 1e-12:
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason="Stop distance too small (abs(entry - sl) <= 1e-12).",
                risk_usd=0.0,
                rr_ratio=0.0,
            )

        reward_distance = abs(tp - entry)
        rr_ratio = reward_distance / stop_distance
        if rr_ratio < self.min_rr_ratio - 1e-9:
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason=(
                    f"Reward-to-risk ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio:.2f}."
                ),
                risk_usd=0.0,
                rr_ratio=round(rr_ratio, 4),
            )

        # 4. Exposure & Anti-Revenge Guards
        if len(open_positions) >= self.max_open_trades:
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason=(
                    f"Max concurrent open trades reached ({len(open_positions)}/{self.max_open_trades})."
                ),
                risk_usd=0.0,
                rr_ratio=round(rr_ratio, 4),
            )

        symbol_open_count = sum(1 for pos in open_positions if pos.get("symbol") == symbol)
        if symbol_open_count >= self.max_open_per_symbol:
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason=f"Duplicate symbol exposure blocked: {symbol} already has an open trade.",
                risk_usd=0.0,
                rr_ratio=round(rr_ratio, 4),
            )

        if candles_since_last_loss is not None and candles_since_last_loss < self.loss_cooldown_candles:
            return RiskDecision(
                approved=False,
                qty=0.0,
                rejection_reason=(
                    f"Anti-revenge cooldown active on {symbol}: {candles_since_last_loss}/"
                    f"{self.loss_cooldown_candles} candles elapsed since last loss."
                ),
                risk_usd=0.0,
                rr_ratio=round(rr_ratio, 4),
            )

        # 5. Position Sizing with 25% Notional Cap
        target_risk_usd = float(equity) * (self.max_risk_per_trade_pct / 100.0)
        raw_qty = target_risk_usd / stop_distance

        max_notional_usd = float(equity) * (self.max_notional_pct / 100.0)
        max_qty = max_notional_usd / entry

        qty = min(raw_qty, max_qty)
        effective_risk_usd = qty * stop_distance

        return RiskDecision(
            approved=True,
            qty=qty,
            rejection_reason=None,
            risk_usd=round(effective_risk_usd, 4),
            rr_ratio=round(rr_ratio, 4),
            trigger_kill_switch=False,
        )

    @staticmethod
    def compute_trailing_sl(
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        initial_risk: float,
    ) -> float:
        """Shift SL to breakeven +0.1R once open profit reaches >= +1.5R; never move backward."""
        if (
            not _is_finite_positive(entry_price)
            or not _is_finite_positive(current_price)
            or not _is_finite_positive(current_sl)
            or not _is_finite_positive(initial_risk)
            or float(initial_risk) <= 1e-12
        ):
            return current_sl

        norm_side = _normalize_side(side)
        if norm_side is None:
            return current_sl

        entry = float(entry_price)
        price = float(current_price)
        sl = float(current_sl)
        r = float(initial_risk)

        if norm_side == "LONG":
            open_profit = price - entry
            if open_profit >= 1.5 * r:
                target_sl = entry + 0.1 * r
                return max(sl, target_sl)
            return sl

        # SHORT
        open_profit = entry - price
        if open_profit >= 1.5 * r:
            target_sl = entry - 0.1 * r
            return min(sl, target_sl)
        return sl
