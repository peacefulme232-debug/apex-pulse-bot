"""Market Regime Traffic Cop evaluating strictly on the last closed candle (iloc[-2])."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import pandas as pd

from app.engine.indicators import MIN_CANDLES, add_all_indicators

REQUIRED_INDICATOR_COLUMNS = (
    "ema21",
    "ema50",
    "atr",
    "atr_pct",
    "adx",
    "bbw_percentile",
)


class MarketRegime(str, Enum):
    """Canonical market regime classifications."""

    TRENDING = "TRENDING"
    RANGING = "RANGING"
    BREAKOUT_SQUEEZE = "BREAKOUT_SQUEEZE"
    CHAOTIC_NO_TRADE = "CHAOTIC_NO_TRADE"


@dataclass(frozen=True)
class RegimeResult:
    """Structured regime evaluation output for strategy routing and dashboard display."""

    regime: MarketRegime
    confidence: float
    adx: float
    atr_pct: float
    bbw_percentile: float
    reason: str


def _is_valid_number(value: float) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


class RegimeDetector:
    """Classifies market structure strictly on the last completed closed candle (iloc[-2])."""

    def __init__(self, min_candles: int = MIN_CANDLES) -> None:
        self.min_candles = min_candles

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """Evaluate regime strictly on closed candle iloc[-2] (never iloc[-1])."""
        if df is None or len(df) < self.min_candles:
            count = 0 if df is None else len(df)
            return RegimeResult(
                regime=MarketRegime.CHAOTIC_NO_TRADE,
                confidence=1.0,
                adx=0.0,
                atr_pct=0.0,
                bbw_percentile=0.0,
                reason=f"Insufficient candle history ({count}/{self.min_candles} bars); trading halted.",
            )

        if all(col in df.columns for col in REQUIRED_INDICATOR_COLUMNS):
            enriched = df
        else:
            enriched = add_all_indicators(df)

        # Strictly exclude the live forming candle (iloc[-1])
        closed_history = enriched.iloc[:-1]
        candle = closed_history.iloc[-1]  # identical to enriched.iloc[-2]

        close = float(candle["close"])
        high = float(candle["high"])
        low = float(candle["low"])
        atr_val = float(candle["atr"])
        atr_pct = float(candle["atr_pct"])
        adx_val = float(candle["adx"])
        ema21 = float(candle["ema21"])
        ema50 = float(candle["ema50"])
        bbw_pct = float(candle["bbw_percentile"])

        if not all(
            _is_valid_number(v)
            for v in (close, high, low, atr_val, atr_pct, adx_val, ema21, ema50, bbw_pct)
        ) or close <= 0.0:
            return RegimeResult(
                regime=MarketRegime.CHAOTIC_NO_TRADE,
                confidence=1.0,
                adx=0.0,
                atr_pct=0.0,
                bbw_percentile=0.0,
                reason="Invalid or NaN indicator values on closed candle; blocking new entries.",
            )

        # 1. CHAOTIC_NO_TRADE check: ATR% > 95th percentile of last 100 closed candles OR candle range > 3.2 * ATR
        recent_atr_pct = closed_history["atr_pct"].dropna().tail(100)
        p95_atr_pct = float(np.percentile(recent_atr_pct, 95)) if len(recent_atr_pct) > 0 else float("inf")
        candle_range = abs(high - low)

        if atr_pct > p95_atr_pct:
            return RegimeResult(
                regime=MarketRegime.CHAOTIC_NO_TRADE,
                confidence=0.95,
                adx=round(adx_val, 2),
                atr_pct=round(atr_pct, 4),
                bbw_percentile=round(bbw_pct, 2),
                reason=(
                    f"Volatility spike detected: ATR% ({atr_pct:.3f}%) exceeds 95th percentile "
                    f"({p95_atr_pct:.3f}%) of last 100 closed candles."
                ),
            )

        if atr_val > 1e-12 and candle_range > 3.2 * atr_val:
            return RegimeResult(
                regime=MarketRegime.CHAOTIC_NO_TRADE,
                confidence=0.95,
                adx=round(adx_val, 2),
                atr_pct=round(atr_pct, 4),
                bbw_percentile=round(bbw_pct, 2),
                reason=(
                    f"Extreme shock candle detected: range ({candle_range:.5f}) > 3.2x ATR "
                    f"({3.2 * atr_val:.5f}); blocking entries."
                ),
            )

        # 2. BREAKOUT_SQUEEZE check
        if bbw_pct <= 20.0 and adx_val < 25.0:
            confidence = min(0.95, max(0.60, (25.0 - bbw_pct) / 25.0))
            return RegimeResult(
                regime=MarketRegime.BREAKOUT_SQUEEZE,
                confidence=round(confidence, 2),
                adx=round(adx_val, 2),
                atr_pct=round(atr_pct, 4),
                bbw_percentile=round(bbw_pct, 2),
                reason=(
                    f"Volatility compression squeeze: BBW percentile {bbw_pct:.1f}% <= 20% "
                    f"with ADX {adx_val:.1f} < 25."
                ),
            )

        # 3. TRENDING check
        ema_sep = abs(ema21 - ema50) / close
        if adx_val >= 25.0 and ema_sep >= 0.0015:
            confidence = min(0.95, max(0.65, adx_val / 50.0))
            return RegimeResult(
                regime=MarketRegime.TRENDING,
                confidence=round(confidence, 2),
                adx=round(adx_val, 2),
                atr_pct=round(atr_pct, 4),
                bbw_percentile=round(bbw_pct, 2),
                reason=(
                    f"Sustained directional trend: ADX {adx_val:.1f} >= 25 and EMA21/50 "
                    f"separation {ema_sep * 100:.2f}% >= 0.15%."
                ),
            )

        # 4. RANGING check
        if adx_val < 22.0:
            confidence = min(0.90, max(0.60, (25.0 - adx_val) / 25.0))
            return RegimeResult(
                regime=MarketRegime.RANGING,
                confidence=round(confidence, 2),
                adx=round(adx_val, 2),
                atr_pct=round(atr_pct, 4),
                bbw_percentile=round(bbw_pct, 2),
                reason=(
                    f"Mean-reverting range conditions: ADX {adx_val:.1f} < 22 under normal volatility "
                    f"(ATR% {atr_pct:.3f}%)."
                ),
            )

        # 5. Borderline fallback
        return RegimeResult(
            regime=MarketRegime.RANGING,
            confidence=0.45,
            adx=round(adx_val, 2),
            atr_pct=round(atr_pct, 4),
            bbw_percentile=round(bbw_pct, 2),
            reason=(
                f"Borderline regime (ADX {adx_val:.1f}, EMA sep {ema_sep * 100:.2f}%, "
                f"BBW pct {bbw_pct:.1f}%); defaulting conservatively to RANGING."
            ),
        )
