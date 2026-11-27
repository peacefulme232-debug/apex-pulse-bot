"""Mutually exclusive zero-repaint Regime Router operating on closed H1/H4/D1 candles.

Priority Order (strictly mutually exclusive):
1. DEAD_CHOP (Capital Preservation — checked FIRST):
   H1 ADX(14) < 18 OR H1 ATR percentile < 15 OR abnormal spike candle (range > 3.2 * ATR).
   -> allowed_strategies = [] (Cash mode).
2. REVERSAL_EXHAUSTION:
   Confirmed H4 or Daily regular RSI(14) divergence active at a swing extreme.
   -> allowed_strategies = ["RSI_DIVERGENCE_SNIPER"] + veto_longs / veto_shorts.
3. BREAKOUT_EXPANSION:
   Latest closed H1/H4 candle broke and closed beyond a confirmed key S/R zone
   (close > zone_high + 0.25 * ATR or close < zone_low - 0.25 * ATR) with strong
   momentum (body_ratio >= 0.55 and range >= 1.15 * ATR).
   -> allowed_strategies = ["BREAKOUT_RETEST"].
4. TRENDING_BULL:
   H4 Close > H4 EMA 200, H1 EMA 50 > H1 EMA 200, H1 EMA 50 slope > 0,
   H1 ADX >= 22, +DI > -DI.
   -> allowed_strategies = ["EMA_50_PULLBACK"], bias = "LONG".
5. TRENDING_BEAR:
   H4 Close < H4 EMA 200, H1 EMA 50 < H1 EMA 200, H1 EMA 50 slope < 0,
   H1 ADX >= 22, -DI > +DI.
   -> allowed_strategies = ["EMA_50_PULLBACK"], bias = "SHORT".
6. Ambiguous Fallback (e.g. 18 <= ADX < 22 with no breakout/divergence):
   DEAD_CHOP -> allowed_strategies = [] (when in doubt, stay in cash).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Sequence

from app.services.indicators import (
    Candle,
    CandleLike,
    SRZone,
    adx_wilder,
    atr_percentile,
    atr_wilder,
    candle_metrics,
    confirmed_pivots,
    ema,
    ema_slope,
    filter_closed_candles,
    key_sr_zones,
    rsi_wilder,
)


class RegimeState(str, Enum):
    """Mutually exclusive market regime classifications."""

    DEAD_CHOP = "DEAD_CHOP"
    REVERSAL_EXHAUSTION = "REVERSAL_EXHAUSTION"
    BREAKOUT_EXPANSION = "BREAKOUT_EXPANSION"
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"


@dataclass(frozen=True)
class DivergenceSignal:
    """Confirmed regular RSI(14) divergence at a lagging swing extreme."""

    kind: Literal["BULLISH", "BEARISH"]
    timeframe: str
    pivot_1_index: int
    pivot_2_index: int
    confirmed_at_index: int
    price_1: float
    price_2: float
    rsi_1: float
    rsi_2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "timeframe": self.timeframe,
            "pivot_1_index": self.pivot_1_index,
            "pivot_2_index": self.pivot_2_index,
            "confirmed_at_index": self.confirmed_at_index,
            "price_1": self.price_1,
            "price_2": self.price_2,
            "rsi_1": self.rsi_1,
            "rsi_2": self.rsi_2,
        }


@dataclass(frozen=True)
class BreakoutSignal:
    """Confirmed momentum breakout beyond a clustered S/R zone on the latest closed candle."""

    direction: Literal["LONG", "SHORT"]
    timeframe: str
    zone: SRZone
    close: float
    atr: float
    body_ratio: float
    candle_range: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "timeframe": self.timeframe,
            "zone_low": self.zone.zone_low,
            "zone_high": self.zone.zone_high,
            "touches": self.zone.touches,
            "last_touch_index": self.zone.last_touch_index,
            "close": self.close,
            "atr": self.atr,
            "body_ratio": self.body_ratio,
            "candle_range": self.candle_range,
        }


@dataclass(frozen=True)
class RegimeDecision:
    """Mutually exclusive regime classification with strategy routing and telemetry."""

    symbol: str
    state: RegimeState
    allowed_strategies: list[str]
    bias: Literal["LONG", "SHORT", "NEUTRAL"]
    veto_longs: bool = False
    veto_shorts: bool = False
    reason: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)


def detect_regular_rsi_divergence(
    candles: Sequence[Candle],
    timeframe: str,
    *,
    rsi_period: int = 14,
    left: int = 3,
    right: int = 3,
    recency_bars: int = 6,
    overbought_extreme: float = 60.0,
    oversold_extreme: float = 40.0,
) -> Optional[DivergenceSignal]:
    """Detect a recently confirmed regular RSI(14) divergence at a swing extreme.

    Zero-repaint: uses ONLY confirmed lagging pivots (`confirmed_at_index <= last_idx`)
    and Wilder RSI computed strictly on closed candles.
    """
    n = len(candles)
    if n < max(rsi_period + 2, left + right + 1):
        return None

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]

    rsi_vals = rsi_wilder(closes, period=rsi_period)
    pivots = confirmed_pivots(highs, lows, left=left, right=right)
    if not pivots:
        return None

    last_idx = n - 1
    candidates: list[DivergenceSignal] = []

    high_pivots = [p for p in pivots if p.kind == "HIGH"]
    if len(high_pivots) >= 2:
        p1, p2 = high_pivots[-2], high_pivots[-1]
        r1, r2 = rsi_vals[p1.index], rsi_vals[p2.index]
        if (
            r1 is not None
            and r2 is not None
            and (last_idx - p2.confirmed_at_index) <= recency_bars
            and p2.price > p1.price
            and r2 < r1
            and r1 >= overbought_extreme
        ):
            candidates.append(
                DivergenceSignal(
                    kind="BEARISH",
                    timeframe=timeframe,
                    pivot_1_index=p1.index,
                    pivot_2_index=p2.index,
                    confirmed_at_index=p2.confirmed_at_index,
                    price_1=p1.price,
                    price_2=p2.price,
                    rsi_1=r1,
                    rsi_2=r2,
                )
            )

    low_pivots = [p for p in pivots if p.kind == "LOW"]
    if len(low_pivots) >= 2:
        p1, p2 = low_pivots[-2], low_pivots[-1]
        r1, r2 = rsi_vals[p1.index], rsi_vals[p2.index]
        if (
            r1 is not None
            and r2 is not None
            and (last_idx - p2.confirmed_at_index) <= recency_bars
            and p2.price < p1.price
            and r2 > r1
            and r1 <= oversold_extreme
        ):
            candidates.append(
                DivergenceSignal(
                    kind="BULLISH",
                    timeframe=timeframe,
                    pivot_1_index=p1.index,
                    pivot_2_index=p2.index,
                    confirmed_at_index=p2.confirmed_at_index,
                    price_1=p1.price,
                    price_2=p2.price,
                    rsi_1=r1,
                    rsi_2=r2,
                )
            )

    if not candidates:
        return None

    # Return the most recently confirmed divergence.
    candidates.sort(key=lambda d: (d.confirmed_at_index, d.pivot_2_index))
    return candidates[-1]


def detect_breakout_expansion(
    candles: Sequence[Candle],
    timeframe: str,
    *,
    atr_period: int = 14,
    left: int = 3,
    right: int = 3,
    merge_tolerance_atr: float = 0.5,
    min_touches: int = 2,
) -> tuple[Optional[BreakoutSignal], list[SRZone]]:
    """Check if the latest closed candle broke beyond a confirmed key S/R zone with momentum."""
    n = len(candles)
    if n < max(atr_period + 1, left + right + 2):
        return None, []

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]

    atr_vals = atr_wilder(highs, lows, closes, period=atr_period)
    cur_atr = atr_vals[-1]
    if cur_atr is None or cur_atr <= 1e-12:
        return None, []

    pivots = confirmed_pivots(highs, lows, left=left, right=right)
    zones = key_sr_zones(
        pivots,
        cur_atr,
        merge_tolerance_atr=merge_tolerance_atr,
        min_touches=min_touches,
    )
    if not zones:
        return None, zones

    last_candle = candles[-1]
    prev_candle = candles[-2]
    metrics = candle_metrics(last_candle, prev_candle)
    candle_range = last_candle.high - last_candle.low

    if metrics.body_ratio < 0.55 or candle_range < 1.15 * cur_atr:
        return None, zones

    # Check upside breakout (strong bullish candle closing > zone_high + 0.25 * ATR)
    if last_candle.close > last_candle.open:
        for zone in reversed(zones):
            threshold = zone.zone_high + 0.25 * cur_atr
            if last_candle.close > threshold and prev_candle.close <= threshold:
                return (
                    BreakoutSignal(
                        direction="LONG",
                        timeframe=timeframe,
                        zone=zone,
                        close=last_candle.close,
                        atr=cur_atr,
                        body_ratio=metrics.body_ratio,
                        candle_range=candle_range,
                    ),
                    zones,
                )

    # Check downside breakout (strong bearish candle closing < zone_low - 0.25 * ATR)
    if last_candle.close < last_candle.open:
        for zone in zones:
            threshold = zone.zone_low - 0.25 * cur_atr
            if last_candle.close < threshold and prev_candle.close >= threshold:
                return (
                    BreakoutSignal(
                        direction="SHORT",
                        timeframe=timeframe,
                        zone=zone,
                        close=last_candle.close,
                        atr=cur_atr,
                        body_ratio=metrics.body_ratio,
                        candle_range=candle_range,
                    ),
                    zones,
                )

    return None, zones


def classify_regime(
    symbol: str,
    h1_candles: Any,
    h4_candles: Any,
    d1_candles: Optional[Any] = None,
    *,
    pivot_left: int = 3,
    pivot_right: int = 3,
    divergence_recency_bars: int = 6,
) -> RegimeDecision:
    """Classify a symbol into ONE mutually exclusive `RegimeState` using strictly closed candles."""
    h1 = filter_closed_candles(h1_candles)
    h4 = filter_closed_candles(h4_candles)
    d1 = filter_closed_candles(d1_candles) if d1_candles is not None else []

    base_telemetry: dict[str, Any] = {
        "h1_closed_count": len(h1),
        "h4_closed_count": len(h4),
        "d1_closed_count": len(d1),
        "h1_close": h1[-1].close if h1 else None,
        "h4_close": h4[-1].close if h4 else None,
    }

    if not h1 or not h4:
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.DEAD_CHOP,
            allowed_strategies=[],
            bias="NEUTRAL",
            reason="Insufficient closed H1/H4 candle history; staying in cash.",
            telemetry=base_telemetry,
        )

    # --- Compute H1 closed-candle indicators ---
    h1_highs = [c.high for c in h1]
    h1_lows = [c.low for c in h1]
    h1_closes = [c.close for c in h1]

    h1_ema20_seq = ema(h1_closes, 20)
    h1_ema50_seq = ema(h1_closes, 50)
    h1_ema200_seq = ema(h1_closes, 200)
    h1_ema50_slope_seq = ema_slope(h1_ema50_seq, lookback=5)

    h1_rsi_seq = rsi_wilder(h1_closes, period=14)
    h1_adx_seq, h1_pdi_seq, h1_mdi_seq = adx_wilder(h1_highs, h1_lows, h1_closes, period=14)
    h1_atr_seq = atr_wilder(h1_highs, h1_lows, h1_closes, period=14)
    h1_atr_pct_seq = atr_percentile(h1_atr_seq, window=100)

    h1_close = h1_closes[-1]
    h1_ema20 = h1_ema20_seq[-1]
    h1_ema50 = h1_ema50_seq[-1]
    h1_ema200 = h1_ema200_seq[-1]
    h1_ema50_slope = h1_ema50_slope_seq[-1]
    h1_rsi = h1_rsi_seq[-1]
    h1_adx = h1_adx_seq[-1]
    h1_plus_di = h1_pdi_seq[-1]
    h1_minus_di = h1_mdi_seq[-1]
    h1_atr = h1_atr_seq[-1]
    h1_atr_pct = h1_atr_pct_seq[-1]
    h1_last_range = h1[-1].high - h1[-1].low
    h1_range_atr_ratio = (
        (h1_last_range / h1_atr) if (h1_atr is not None and h1_atr > 1e-12) else None
    )

    # --- Compute H4 closed-candle indicators ---
    h4_highs = [c.high for c in h4]
    h4_lows = [c.low for c in h4]
    h4_closes = [c.close for c in h4]

    h4_close = h4_closes[-1]
    h4_ema200 = ema(h4_closes, 200)[-1]
    h4_rsi = rsi_wilder(h4_closes, period=14)[-1]
    h4_atr = atr_wilder(h4_highs, h4_lows, h4_closes, period=14)[-1]

    telemetry: dict[str, Any] = {
        **base_telemetry,
        "h1_close": h1_close,
        "h1_ema20": h1_ema20,
        "h1_ema50": h1_ema50,
        "h1_ema200": h1_ema200,
        "h1_ema50_slope": h1_ema50_slope,
        "h1_rsi14": h1_rsi,
        "h1_adx14": h1_adx,
        "h1_plus_di": h1_plus_di,
        "h1_minus_di": h1_minus_di,
        "h1_atr14": h1_atr,
        "h1_atr_percentile": h1_atr_pct,
        "h1_last_range": h1_last_range,
        "h1_range_atr_ratio": h1_range_atr_ratio,
        "h4_close": h4_close,
        "h4_ema200": h4_ema200,
        "h4_rsi14": h4_rsi,
        "h4_atr14": h4_atr,
        "divergence": None,
        "breakout_zone": None,
        "h1_sr_zones_count": 0,
        "h4_sr_zones_count": 0,
    }

    # -----------------------------------------------------------------------
    # 1. DEAD_CHOP (Checked FIRST — Capital Preservation)
    # -----------------------------------------------------------------------
    if h1_adx is None or h1_atr is None or h1_atr_pct is None:
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.DEAD_CHOP,
            allowed_strategies=[],
            bias="NEUTRAL",
            reason="H1 ADX/ATR indicators not warmed up; staying in cash.",
            telemetry=telemetry,
        )

    if h1_adx < 18.0:
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.DEAD_CHOP,
            allowed_strategies=[],
            bias="NEUTRAL",
            reason=f"Dead chop filter: H1 ADX(14) {h1_adx:.2f} < 18.0; cash mode.",
            telemetry=telemetry,
        )

    if h1_atr_pct < 15.0:
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.DEAD_CHOP,
            allowed_strategies=[],
            bias="NEUTRAL",
            reason=f"Dead volatility filter: H1 ATR percentile {h1_atr_pct:.2f} < 15.0; cash mode.",
            telemetry=telemetry,
        )

    if h1_atr > 1e-12 and h1_last_range > 3.2 * h1_atr:
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.DEAD_CHOP,
            allowed_strategies=[],
            bias="NEUTRAL",
            reason=(
                f"Abnormal spike candle: H1 range {h1_last_range:.6f} > "
                f"3.2x ATR ({3.2 * h1_atr:.6f}); cash mode."
            ),
            telemetry=telemetry,
        )

    # -----------------------------------------------------------------------
    # 2. REVERSAL_EXHAUSTION (H4 or Daily regular RSI 14 divergence)
    # -----------------------------------------------------------------------
    divergence = None
    if d1:
        divergence = detect_regular_rsi_divergence(
            d1,
            "D1",
            left=pivot_left,
            right=pivot_right,
            recency_bars=divergence_recency_bars,
        )
    if divergence is None:
        divergence = detect_regular_rsi_divergence(
            h4,
            "H4",
            left=pivot_left,
            right=pivot_right,
            recency_bars=divergence_recency_bars,
        )

    if divergence is not None:
        telemetry["divergence"] = divergence.to_dict()
        if divergence.kind == "BEARISH":
            return RegimeDecision(
                symbol=symbol,
                state=RegimeState.REVERSAL_EXHAUSTION,
                allowed_strategies=["RSI_DIVERGENCE_SNIPER"],
                bias="SHORT",
                veto_longs=True,
                veto_shorts=False,
                reason=(
                    f"Confirmed {divergence.timeframe} regular bearish RSI(14) divergence "
                    f"(price {divergence.price_1:.4f}->{divergence.price_2:.4f}, "
                    f"RSI {divergence.rsi_1:.1f}->{divergence.rsi_2:.1f}); vetoing longs."
                ),
                telemetry=telemetry,
            )
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.REVERSAL_EXHAUSTION,
            allowed_strategies=["RSI_DIVERGENCE_SNIPER"],
            bias="LONG",
            veto_longs=False,
            veto_shorts=True,
            reason=(
                f"Confirmed {divergence.timeframe} regular bullish RSI(14) divergence "
                f"(price {divergence.price_1:.4f}->{divergence.price_2:.4f}, "
                f"RSI {divergence.rsi_1:.1f}->{divergence.rsi_2:.1f}); vetoing shorts."
            ),
            telemetry=telemetry,
        )

    # -----------------------------------------------------------------------
    # 3. BREAKOUT_EXPANSION (H4 or H1 confirmed S/R zone momentum breakout)
    # -----------------------------------------------------------------------
    h4_breakout, h4_zones = detect_breakout_expansion(
        h4,
        "H4",
        left=pivot_left,
        right=pivot_right,
    )
    h1_breakout, h1_zones = detect_breakout_expansion(
        h1,
        "H1",
        left=pivot_left,
        right=pivot_right,
    )
    telemetry["h4_sr_zones_count"] = len(h4_zones)
    telemetry["h1_sr_zones_count"] = len(h1_zones)

    breakout = h4_breakout or h1_breakout
    if breakout is not None:
        telemetry["breakout_zone"] = breakout.to_dict()
        return RegimeDecision(
            symbol=symbol,
            state=RegimeState.BREAKOUT_EXPANSION,
            allowed_strategies=["BREAKOUT_RETEST"],
            bias=breakout.direction,
            veto_longs=False,
            veto_shorts=False,
            reason=(
                f"Confirmed {breakout.timeframe} {breakout.direction} breakout beyond S/R zone "
                f"[{breakout.zone.zone_low:.4f}, {breakout.zone.zone_high:.4f}] "
                f"(body_ratio={breakout.body_ratio:.2f}, range={breakout.candle_range:.4f})."
            ),
            telemetry=telemetry,
        )

    # -----------------------------------------------------------------------
    # 4. TRENDING_BULL
    # -----------------------------------------------------------------------
    if (
        h4_ema200 is not None
        and h1_ema50 is not None
        and h1_ema200 is not None
        and h1_ema50_slope is not None
        and h1_plus_di is not None
        and h1_minus_di is not None
    ):
        if (
            h4_close > h4_ema200
            and h1_ema50 > h1_ema200
            and h1_ema50_slope > 0.0
            and h1_adx >= 22.0
            and h1_plus_di > h1_minus_di
        ):
            return RegimeDecision(
                symbol=symbol,
                state=RegimeState.TRENDING_BULL,
                allowed_strategies=["EMA_50_PULLBACK"],
                bias="LONG",
                veto_longs=False,
                veto_shorts=False,
                reason=(
                    f"Aligned bullish trend: H4 close > EMA200, H1 EMA50 > EMA200 "
                    f"(slope={h1_ema50_slope:.5f}), ADX={h1_adx:.2f} >= 22, "
                    f"+DI={h1_plus_di:.2f} > -DI={h1_minus_di:.2f}."
                ),
                telemetry=telemetry,
            )

        # -------------------------------------------------------------------
        # 5. TRENDING_BEAR
        # -------------------------------------------------------------------
        if (
            h4_close < h4_ema200
            and h1_ema50 < h1_ema200
            and h1_ema50_slope < 0.0
            and h1_adx >= 22.0
            and h1_minus_di > h1_plus_di
        ):
            return RegimeDecision(
                symbol=symbol,
                state=RegimeState.TRENDING_BEAR,
                allowed_strategies=["EMA_50_PULLBACK"],
                bias="SHORT",
                veto_longs=False,
                veto_shorts=False,
                reason=(
                    f"Aligned bearish trend: H4 close < EMA200, H1 EMA50 < EMA200 "
                    f"(slope={h1_ema50_slope:.5f}), ADX={h1_adx:.2f} >= 22, "
                    f"-DI={h1_minus_di:.2f} > +DI={h1_plus_di:.2f}."
                ),
                telemetry=telemetry,
            )

    # -----------------------------------------------------------------------
    # 6. Ambiguous Fallback -> DEAD_CHOP (When in doubt, stay in cash!)
    # -----------------------------------------------------------------------
    return RegimeDecision(
        symbol=symbol,
        state=RegimeState.DEAD_CHOP,
        allowed_strategies=[],
        bias="NEUTRAL",
        veto_longs=False,
        veto_shorts=False,
        reason=(
            f"Ambiguous regime (H1 ADX={h1_adx:.2f}, no confirmed divergence/breakout "
            "or unaligned HTF/LTF trend); defaulting to DEAD_CHOP cash mode."
        ),
        telemetry=telemetry,
    )


class RegimeRouter:
    """Stateless wrapper around `classify_regime` for service-layer dependency injection."""

    def __init__(
        self,
        *,
        pivot_left: int = 3,
        pivot_right: int = 3,
        divergence_recency_bars: int = 6,
    ) -> None:
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.divergence_recency_bars = divergence_recency_bars

    def classify(
        self,
        symbol: str,
        h1_candles: Any,
        h4_candles: Any,
        d1_candles: Optional[Any] = None,
    ) -> RegimeDecision:
        return classify_regime(
            symbol=symbol,
            h1_candles=h1_candles,
            h4_candles=h4_candles,
            d1_candles=d1_candles,
            pivot_left=self.pivot_left,
            pivot_right=self.pivot_right,
            divergence_recency_bars=self.divergence_recency_bars,
        )
