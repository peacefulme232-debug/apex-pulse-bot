"""Strategy 1: 50 EMA Pullback (Daily Driver).

Runs strictly in TRENDING_BULL or TRENDING_BEAR regimes on closed H1/H4 candles.
Waits for a controlled pullback into the H1 50 EMA value zone, confirms rejection
or engulfing closing back across the 50 EMA, verifies healthy RSI(14) momentum
(never chasing overbought/oversold extremes), places SL with a minimum 0.8x ATR
stop distance, and targets a Maker Limit Take-Profit >= 2.2R.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from app.services.indicators import (
    adx_wilder,
    atr_wilder,
    candle_metrics,
    ema,
    ema_slope,
    filter_closed_candles,
    rsi_wilder,
)
from app.services.regime_router import RegimeDecision, RegimeState
from app.services.strategies.base import CandidateSignal


def _normalize_regime(regime: Union[str, RegimeState, RegimeDecision, None]) -> str:
    """Normalize a RegimeDecision, RegimeState, or string into an uppercase regime label."""
    if regime is None:
        return ""
    if isinstance(regime, RegimeDecision):
        return regime.state.value
    if isinstance(regime, RegimeState):
        return regime.value
    return str(regime).strip().upper()


class EmaPullbackStrategy:
    """Zero-repaint 50 EMA Pullback strategy operating on closed H1 and H4 candles."""

    STRATEGY_NAME = "EMA_50_PULLBACK"

    def __init__(
        self,
        *,
        min_adx: float = 22.0,
        ema_zone_upper_atr: float = 0.35,
        ema_zone_lower_atr: float = 0.75,
        long_rsi_min: float = 38.0,
        long_rsi_max: float = 58.0,
        short_rsi_min: float = 42.0,
        short_rsi_max: float = 62.0,
        sl_buffer_atr: float = 0.35,
        min_sl_atr: float = 0.8,
        rr_multiple: float = 2.2,
        pullback_lookback: int = 5,
        slope_lookback: int = 5,
        order_entry_type: str = "MARKET",
        order_exit_type: str = "TAKE_PROFIT",
    ) -> None:
        if rr_multiple < 2.2:
            raise ValueError(f"rr_multiple must be >= 2.2, got {rr_multiple}")
        if min_sl_atr < 0.8:
            raise ValueError(f"min_sl_atr must be >= 0.8, got {min_sl_atr}")

        self.min_adx = float(min_adx)
        self.ema_zone_upper_atr = float(ema_zone_upper_atr)
        self.ema_zone_lower_atr = float(ema_zone_lower_atr)
        self.long_rsi_min = float(long_rsi_min)
        self.long_rsi_max = float(long_rsi_max)
        self.short_rsi_min = float(short_rsi_min)
        self.short_rsi_max = float(short_rsi_max)
        self.sl_buffer_atr = float(sl_buffer_atr)
        self.min_sl_atr = float(min_sl_atr)
        self.rr_multiple = float(rr_multiple)
        self.pullback_lookback = int(pullback_lookback)
        self.slope_lookback = int(slope_lookback)
        self.order_entry_type = order_entry_type
        self.order_exit_type = order_exit_type

    def evaluate(
        self,
        symbol: str,
        market: str,
        regime: Union[str, RegimeState, RegimeDecision, None],
        h1_candles: Any,
        h4_candles: Any,
    ) -> Optional[CandidateSignal]:
        """Evaluate closed H1/H4 candles for a 50 EMA pullback rejection setup."""
        regime_label = _normalize_regime(regime)
        if regime_label not in (RegimeState.TRENDING_BULL.value, RegimeState.TRENDING_BEAR.value):
            return None

        h1 = filter_closed_candles(h1_candles)
        h4 = filter_closed_candles(h4_candles)
        if len(h1) < 200 or len(h4) < 200:
            return None

        h1_highs = [c.high for c in h1]
        h1_lows = [c.low for c in h1]
        h1_closes = [c.close for c in h1]
        h4_closes = [c.close for c in h4]

        h4_ema200 = ema(h4_closes, 200)[-1]
        h1_ema50_seq = ema(h1_closes, 50)
        h1_ema200_seq = ema(h1_closes, 200)
        h1_ema50_slope_seq = ema_slope(h1_ema50_seq, lookback=self.slope_lookback)
        h1_rsi_seq = rsi_wilder(h1_closes, period=14)
        h1_adx_seq, _, _ = adx_wilder(h1_highs, h1_lows, h1_closes, period=14)
        h1_atr_seq = atr_wilder(h1_highs, h1_lows, h1_closes, period=14)

        h4_close = h4_closes[-1]
        h1_ema50 = h1_ema50_seq[-1]
        h1_ema200 = h1_ema200_seq[-1]
        h1_ema50_slope = h1_ema50_slope_seq[-1]
        h1_rsi = h1_rsi_seq[-1]
        h1_adx = h1_adx_seq[-1]
        h1_atr = h1_atr_seq[-1]

        if (
            h4_ema200 is None
            or h1_ema50 is None
            or h1_ema200 is None
            or h1_ema50_slope is None
            or h1_rsi is None
            or h1_adx is None
            or h1_atr is None
            or h1_atr <= 1e-12
        ):
            return None

        candle = h1[-1]
        prev_candle = h1[-2]
        metrics = candle_metrics(candle, prev_candle)
        lookback_window = h1[-max(1, self.pullback_lookback) :]

        if regime_label == RegimeState.TRENDING_BULL.value:
            # 1. Trend alignment
            if not (
                h4_close > h4_ema200
                and h1_ema50 > h1_ema200
                and h1_ema50_slope > 0.0
                and h1_adx >= self.min_adx
            ):
                return None

            # 2. Price pulls back into 50 EMA zone
            if not (
                candle.low <= h1_ema50 + self.ema_zone_upper_atr * h1_atr
                and candle.close >= h1_ema50 - self.ema_zone_lower_atr * h1_atr
            ):
                return None

            # 3. Rejection confirmation closing above EMA 50
            if not (
                (metrics.is_bullish_rejection or metrics.is_bullish_engulfing)
                and candle.close > h1_ema50
            ):
                return None

            # 4. Healthy RSI momentum (never chasing overbought extremes)
            if not (self.long_rsi_min <= h1_rsi <= self.long_rsi_max):
                return None

            # 5. Stop Loss with minimum 0.8 * ATR stop distance
            pullback_swing_low = min(c.low for c in lookback_window)
            entry_price = candle.close
            raw_sl = min(pullback_swing_low, candle.low) - self.sl_buffer_atr * h1_atr
            sl_distance = max(entry_price - raw_sl, self.min_sl_atr * h1_atr)
            sl_price = entry_price - sl_distance

            # 6. Limit Take-Profit at >= 2.2R
            tp_price = entry_price + self.rr_multiple * sl_distance
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
                    f"50 EMA pullback LONG ({pattern}): H4 close > EMA200, H1 EMA50 > EMA200 "
                    f"(slope={h1_ema50_slope:.5f}), ADX={h1_adx:.1f}, RSI={h1_rsi:.1f}, "
                    f"RR={self.rr_multiple:.2f}R."
                ),
                metadata={
                    "regime": regime_label,
                    "pattern": pattern,
                    "h4_close": h4_close,
                    "h4_ema200": h4_ema200,
                    "h1_ema50": h1_ema50,
                    "h1_ema200": h1_ema200,
                    "h1_ema50_slope": h1_ema50_slope,
                    "h1_adx": h1_adx,
                    "h1_rsi": h1_rsi,
                    "h1_atr": h1_atr,
                    "pullback_swing_low": pullback_swing_low,
                    "sl_distance": sl_distance,
                    "sl_distance_atr": sl_distance / h1_atr,
                    "rr_multiple": self.rr_multiple,
                },
            )

        # --- Exact mirror for TRENDING_BEAR (SHORT) ---
        # 1. Trend alignment
        if not (
            h4_close < h4_ema200
            and h1_ema50 < h1_ema200
            and h1_ema50_slope < 0.0
            and h1_adx >= self.min_adx
        ):
            return None

        # 2. Price pulls back into 50 EMA zone
        if not (
            candle.high >= h1_ema50 - self.ema_zone_upper_atr * h1_atr
            and candle.close <= h1_ema50 + self.ema_zone_lower_atr * h1_atr
        ):
            return None

        # 3. Rejection confirmation closing below EMA 50
        if not (
            (metrics.is_bearish_rejection or metrics.is_bearish_engulfing)
            and candle.close < h1_ema50
        ):
            return None

        # 4. Healthy RSI momentum (never chasing oversold extremes)
        if not (self.short_rsi_min <= h1_rsi <= self.short_rsi_max):
            return None

        # 5. Stop Loss with minimum 0.8 * ATR stop distance
        pullback_swing_high = max(c.high for c in lookback_window)
        entry_price = candle.close
        raw_sl = max(pullback_swing_high, candle.high) + self.sl_buffer_atr * h1_atr
        sl_distance = max(raw_sl - entry_price, self.min_sl_atr * h1_atr)
        sl_price = entry_price + sl_distance

        # 6. Limit Take-Profit at >= 2.2R
        tp_price = entry_price - self.rr_multiple * sl_distance
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
                f"50 EMA pullback SHORT ({pattern}): H4 close < EMA200, H1 EMA50 < EMA200 "
                f"(slope={h1_ema50_slope:.5f}), ADX={h1_adx:.1f}, RSI={h1_rsi:.1f}, "
                f"RR={self.rr_multiple:.2f}R."
            ),
            metadata={
                "regime": regime_label,
                "pattern": pattern,
                "h4_close": h4_close,
                "h4_ema200": h4_ema200,
                "h1_ema50": h1_ema50,
                "h1_ema200": h1_ema200,
                "h1_ema50_slope": h1_ema50_slope,
                "h1_adx": h1_adx,
                "h1_rsi": h1_rsi,
                "h1_atr": h1_atr,
                "pullback_swing_high": pullback_swing_high,
                "sl_distance": sl_distance,
                "sl_distance_atr": sl_distance / h1_atr,
                "rr_multiple": self.rr_multiple,
            },
        )
