"""Strategy 3: H4/D1 RSI(14) Divergence Sniper + Veto.

Exposes two clean methods:
1. `detect_active_divergence(candles_htf)`:
   Detects confirmed regular bullish or bearish RSI(14) divergence using
   strictly lagging confirmed pivots (left=3, right=3, separated by 5-45 bars).
   - Bullish: P2.price < P1.price - 0.25 * ATR, P2.rsi > P1.rsi + 2.5, P1.rsi <= 35.
   - Bearish: P2.price > P1.price + 0.25 * ATR, P2.rsi < P1.rsi - 2.5, P1.rsi >= 65.
2. `evaluate_sniper_entry(candles_htf, candles_exec)`:
   No Falling Knives! Emits a CandidateSignal ONLY when regular divergence is
   active AND the closed execution candle confirms reversal (engulfing or strong
   rejection pin bar + RSI turning upward past 35 for longs, or downward past 65
   for shorts).
   - SL: Beyond P2.price +/- 0.45 * ATR.
   - TP: Limit Take-Profit at >= 2.5R (order_exit_type="TAKE_PROFIT").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.services.indicators import (
    atr_wilder,
    candle_metrics,
    confirmed_pivots,
    filter_closed_candles,
    rsi_wilder,
)
from app.services.strategies.base import CandidateSignal


@dataclass(frozen=True)
class DivergencePivot:
    """Lagging confirmed price pivot paired with its closed-candle RSI(14) reading."""

    index: int
    confirmed_at_index: int
    kind: Literal["HIGH", "LOW"]
    price: float
    rsi: float


@dataclass(frozen=True)
class ActiveDivergence:
    """Active confirmed regular RSI(14) divergence detected on HTF candles."""

    kind: Literal["BULLISH", "BEARISH"]
    p1: DivergencePivot
    p2: DivergencePivot
    strength: float
    atr: float
    bars_between: int
    confirmed_at_index: int

    @property
    def direction(self) -> Literal["BULLISH", "BEARISH"]:
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "p1_index": self.p1.index,
            "p1_price": self.p1.price,
            "p1_rsi": self.p1.rsi,
            "p2_index": self.p2.index,
            "p2_price": self.p2.price,
            "p2_rsi": self.p2.rsi,
            "strength": self.strength,
            "atr": self.atr,
            "bars_between": self.bars_between,
            "confirmed_at_index": self.confirmed_at_index,
        }


class RsiDivergenceSniperStrategy:
    """Zero-repaint H4/D1 RSI(14) Divergence detector and execution sniper."""

    STRATEGY_NAME = "RSI_DIVERGENCE_SNIPER"

    def __init__(
        self,
        *,
        rsi_period: int = 14,
        atr_period: int = 14,
        pivot_left: int = 3,
        pivot_right: int = 3,
        min_pivot_separation: int = 5,
        max_pivot_separation: int = 45,
        price_diff_atr: float = 0.25,
        rsi_diff_min: float = 2.5,
        bullish_p1_rsi_max: float = 35.0,
        bearish_p1_rsi_min: float = 65.0,
        recency_bars: int = 12,
        exec_long_rsi_threshold: float = 35.0,
        exec_short_rsi_threshold: float = 65.0,
        sl_buffer_atr: float = 0.45,
        rr_multiple: float = 2.5,
        order_entry_type: str = "MARKET",
        order_exit_type: str = "TAKE_PROFIT",
    ) -> None:
        if rr_multiple < 2.5:
            raise ValueError(f"rr_multiple must be >= 2.5, got {rr_multiple}")

        self.rsi_period = int(rsi_period)
        self.atr_period = int(atr_period)
        self.pivot_left = int(pivot_left)
        self.pivot_right = int(pivot_right)
        self.min_pivot_separation = int(min_pivot_separation)
        self.max_pivot_separation = int(max_pivot_separation)
        self.price_diff_atr = float(price_diff_atr)
        self.rsi_diff_min = float(rsi_diff_min)
        self.bullish_p1_rsi_max = float(bullish_p1_rsi_max)
        self.bearish_p1_rsi_min = float(bearish_p1_rsi_min)
        self.recency_bars = int(recency_bars)
        self.exec_long_rsi_threshold = float(exec_long_rsi_threshold)
        self.exec_short_rsi_threshold = float(exec_short_rsi_threshold)
        self.sl_buffer_atr = float(sl_buffer_atr)
        self.rr_multiple = float(rr_multiple)
        self.order_entry_type = order_entry_type
        self.order_exit_type = order_exit_type

    def detect_active_divergence(
        self,
        candles_htf: Any,
    ) -> Optional[ActiveDivergence]:
        """Detect an active confirmed regular RSI(14) divergence on closed HTF candles.

        Uses strictly lagging confirmed pivots (left=3, right=3, separated by 5-45 bars).
        """
        htf = filter_closed_candles(candles_htf)
        n = len(htf)
        min_bars = max(self.rsi_period + 2, self.atr_period + 1, self.pivot_left + self.pivot_right + 1)
        if n < min_bars:
            return None

        highs = [c.high for c in htf]
        lows = [c.low for c in htf]
        closes = [c.close for c in htf]

        rsi_vals = rsi_wilder(closes, period=self.rsi_period)
        atr_vals = atr_wilder(highs, lows, closes, period=self.atr_period)
        pivots = confirmed_pivots(highs, lows, left=self.pivot_left, right=self.pivot_right)
        if not pivots:
            return None

        last_idx = n - 1
        candidates: list[ActiveDivergence] = []

        # --- Check Bullish Regular Divergence (confirmed swing LOWs) ---
        low_pivots = [p for p in pivots if p.kind == "LOW"]
        if len(low_pivots) >= 2:
            p2 = low_pivots[-1]
            r2 = rsi_vals[p2.index]
            atr_at_p2 = atr_vals[p2.index]
            if (
                r2 is not None
                and atr_at_p2 is not None
                and atr_at_p2 > 1e-12
                and (last_idx - p2.confirmed_at_index) <= self.recency_bars
            ):
                for p1 in reversed(low_pivots[:-1]):
                    sep = p2.index - p1.index
                    if sep < self.min_pivot_separation:
                        continue
                    if sep > self.max_pivot_separation:
                        break
                    r1 = rsi_vals[p1.index]
                    if r1 is None:
                        continue
                    if (
                        p2.price < p1.price - self.price_diff_atr * atr_at_p2
                        and r2 > r1 + self.rsi_diff_min
                        and r1 <= self.bullish_p1_rsi_max
                    ):
                        rsi_delta = r2 - r1
                        price_delta_atr = (p1.price - p2.price) / atr_at_p2
                        strength = min(100.0, round(rsi_delta * 3.0 + price_delta_atr * 15.0, 2))
                        candidates.append(
                            ActiveDivergence(
                                kind="BULLISH",
                                p1=DivergencePivot(
                                    index=p1.index,
                                    confirmed_at_index=p1.confirmed_at_index,
                                    kind="LOW",
                                    price=p1.price,
                                    rsi=r1,
                                ),
                                p2=DivergencePivot(
                                    index=p2.index,
                                    confirmed_at_index=p2.confirmed_at_index,
                                    kind="LOW",
                                    price=p2.price,
                                    rsi=r2,
                                ),
                                strength=strength,
                                atr=atr_at_p2,
                                bars_between=sep,
                                confirmed_at_index=p2.confirmed_at_index,
                            )
                        )
                        break

        # --- Check Bearish Regular Divergence (confirmed swing HIGHs) ---
        high_pivots = [p for p in pivots if p.kind == "HIGH"]
        if len(high_pivots) >= 2:
            p2 = high_pivots[-1]
            r2 = rsi_vals[p2.index]
            atr_at_p2 = atr_vals[p2.index]
            if (
                r2 is not None
                and atr_at_p2 is not None
                and atr_at_p2 > 1e-12
                and (last_idx - p2.confirmed_at_index) <= self.recency_bars
            ):
                for p1 in reversed(high_pivots[:-1]):
                    sep = p2.index - p1.index
                    if sep < self.min_pivot_separation:
                        continue
                    if sep > self.max_pivot_separation:
                        break
                    r1 = rsi_vals[p1.index]
                    if r1 is None:
                        continue
                    if (
                        p2.price > p1.price + self.price_diff_atr * atr_at_p2
                        and r2 < r1 - self.rsi_diff_min
                        and r1 >= self.bearish_p1_rsi_min
                    ):
                        rsi_delta = r1 - r2
                        price_delta_atr = (p2.price - p1.price) / atr_at_p2
                        strength = min(100.0, round(rsi_delta * 3.0 + price_delta_atr * 15.0, 2))
                        candidates.append(
                            ActiveDivergence(
                                kind="BEARISH",
                                p1=DivergencePivot(
                                    index=p1.index,
                                    confirmed_at_index=p1.confirmed_at_index,
                                    kind="HIGH",
                                    price=p1.price,
                                    rsi=r1,
                                ),
                                p2=DivergencePivot(
                                    index=p2.index,
                                    confirmed_at_index=p2.confirmed_at_index,
                                    kind="HIGH",
                                    price=p2.price,
                                    rsi=r2,
                                ),
                                strength=strength,
                                atr=atr_at_p2,
                                bars_between=sep,
                                confirmed_at_index=p2.confirmed_at_index,
                            )
                        )
                        break

        if not candidates:
            return None

        # Return the most recently confirmed divergence
        candidates.sort(key=lambda d: (d.confirmed_at_index, d.p2.index, d.strength))
        return candidates[-1]

    def evaluate_sniper_entry(
        self,
        candles_htf: Any,
        candles_exec: Any,
        *,
        symbol: str = "UNKNOWN",
        market: str = "bybit",
    ) -> Optional[CandidateSignal]:
        """Emit a CandidateSignal only when HTF divergence is active AND exec candle confirms reversal."""
        div = self.detect_active_divergence(candles_htf)
        if div is None:
            return None

        exec_closed = filter_closed_candles(candles_exec)
        if len(exec_closed) < self.rsi_period + 2:
            return None

        exec_closes = [c.close for c in exec_closed]
        exec_rsi_seq = rsi_wilder(exec_closes, period=self.rsi_period)
        cur_rsi = exec_rsi_seq[-1]
        prev_rsi = exec_rsi_seq[-2]
        if cur_rsi is None or prev_rsi is None:
            return None

        last_exec = exec_closed[-1]
        prev_exec = exec_closed[-2]
        metrics = candle_metrics(last_exec, prev_exec)

        if div.kind == "BULLISH":
            # No falling knives: require bullish reversal candle + RSI turning upward past 35
            has_reversal_candle = metrics.is_bullish_engulfing or metrics.is_bullish_rejection
            rsi_turning_up = cur_rsi > prev_rsi and cur_rsi > self.exec_long_rsi_threshold
            if not (has_reversal_candle and rsi_turning_up):
                return None

            entry_price = last_exec.close
            sl_price = div.p2.price - self.sl_buffer_atr * div.atr
            risk = entry_price - sl_price
            if risk <= 1e-12:
                return None

            tp_price = entry_price + self.rr_multiple * risk
            pattern = (
                "bullish_engulfing"
                if metrics.is_bullish_engulfing
                else "bullish_rejection"
            )
            return CandidateSignal(
                symbol=symbol,
                market=market,
                strategy=self.STRATEGY_NAME,
                side="LONG",
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                order_entry_type=self.order_entry_type,
                order_exit_type=self.order_exit_type,
                reason=(
                    f"RSI(14) Bullish Divergence Sniper LONG ({pattern}): "
                    f"P1={div.p1.price:.4f} (RSI {div.p1.rsi:.1f}) -> "
                    f"P2={div.p2.price:.4f} (RSI {div.p2.rsi:.1f}), "
                    f"exec RSI {prev_rsi:.1f}->{cur_rsi:.1f}, RR={self.rr_multiple:.2f}R."
                ),
                metadata={
                    "divergence": div.to_dict(),
                    "pattern": pattern,
                    "exec_rsi_prev": prev_rsi,
                    "exec_rsi_curr": cur_rsi,
                    "rr_multiple": self.rr_multiple,
                },
            )

        # --- Exact mirror for BEARISH divergence (SHORT) ---
        has_reversal_candle = metrics.is_bearish_engulfing or metrics.is_bearish_rejection
        rsi_turning_down = cur_rsi < prev_rsi and cur_rsi < self.exec_short_rsi_threshold
        if not (has_reversal_candle and rsi_turning_down):
            return None

        entry_price = last_exec.close
        sl_price = div.p2.price + self.sl_buffer_atr * div.atr
        risk = sl_price - entry_price
        if risk <= 1e-12:
            return None

        tp_price = entry_price - self.rr_multiple * risk
        pattern = (
            "bearish_engulfing"
            if metrics.is_bearish_engulfing
            else "bearish_rejection"
        )
        return CandidateSignal(
            symbol=symbol,
            market=market,
            strategy=self.STRATEGY_NAME,
            side="SHORT",
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            order_entry_type=self.order_entry_type,
            order_exit_type=self.order_exit_type,
            reason=(
                f"RSI(14) Bearish Divergence Sniper SHORT ({pattern}): "
                f"P1={div.p1.price:.4f} (RSI {div.p1.rsi:.1f}) -> "
                f"P2={div.p2.price:.4f} (RSI {div.p2.rsi:.1f}), "
                f"exec RSI {prev_rsi:.1f}->{cur_rsi:.1f}, RR={self.rr_multiple:.2f}R."
            ),
            metadata={
                "divergence": div.to_dict(),
                "pattern": pattern,
                "exec_rsi_prev": prev_rsi,
                "exec_rsi_curr": cur_rsi,
                "rr_multiple": self.rr_multiple,
            },
        )
