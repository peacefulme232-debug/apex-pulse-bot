"""Institutional Sniper Filter: Multi-Timeframe (4H) + Rolling VWAP + Volume Surge."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.engine.indicators import ema
from app.engine.strategies import TradeSignal


@dataclass(frozen=True)
class ConfluenceResult:
    """Result of the Institutional Confluence Filter evaluation."""

    passed: bool
    score: int
    htf_bias: str  # "BULLISH" | "BEARISH" | "NEUTRAL" | "UNAVAILABLE"
    vwap_aligned: bool
    volume_ratio: float
    reasons: list[str] = field(default_factory=list)


class ConfluenceFilter:
    """High-precision MTF + VWAP + Volume filter evaluated strictly on closed candle iloc[-2]."""

    def __init__(self, min_score: Optional[int] = None) -> None:
        self.min_score = settings.CONFLUENCE_MIN_SCORE if min_score is None else int(min_score)

    @staticmethod
    def compute_rolling_vwap(df: pd.DataFrame, window: int = 24) -> pd.Series:
        """Compute rolling VWAP = sum(typical_price * volume) / sum(volume)."""
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float).clip(lower=0.0)

        typical_price = (high + low + close) / 3.0
        pv = typical_price * volume

        sum_pv = pv.rolling(window=window, min_periods=1).sum()
        sum_vol = volume.rolling(window=window, min_periods=1).sum()
        safe_vol = sum_vol.where(sum_vol > 1e-12, np.nan)
        vwap = (sum_pv / safe_vol).fillna(typical_price)
        return vwap

    @staticmethod
    def compute_volume_ratio(df: pd.DataFrame, window: int = 20) -> float:
        """Compute closed candle volume / avg_20_volume strictly on iloc[-2]."""
        if df is None or len(df) < 3:
            return 1.0
        vol = df["volume"].astype(float)
        closed_vol = float(vol.iloc[-2])
        # Use up to `window` closed candles ending at iloc[-2] (inclusive of iloc[-2] or prior 20)
        start_idx = max(0, len(vol) - 1 - window)
        avg_vol = float(vol.iloc[start_idx:-1].mean())
        if not math.isfinite(avg_vol) or avg_vol <= 1e-12:
            return 1.0 if closed_vol <= 1e-12 else 1.5
        if not math.isfinite(closed_vol):
            return 0.0
        return round(closed_vol / avg_vol, 4)

    @staticmethod
    def _evaluate_htf_bias(htf_df: Optional[pd.DataFrame]) -> str:
        """Determine 4H macro bias strictly on closed candle iloc[-2] when >= 55 bars exist."""
        if htf_df is None or len(htf_df) < 55:
            return "UNAVAILABLE"

        htf_close = htf_df["close"].astype(float)
        ema21 = ema(htf_close, 21)
        ema50 = ema(htf_close, 50)

        c = float(htf_close.iloc[-2])
        e21 = float(ema21.iloc[-2])
        e50 = float(ema50.iloc[-2])

        if not (math.isfinite(c) and math.isfinite(e21) and math.isfinite(e50)):
            return "NEUTRAL"

        if e21 > e50 and c > e50:
            return "BULLISH"
        if e21 < e50 and c < e50:
            return "BEARISH"
        return "NEUTRAL"

    def evaluate(
        self,
        signal: TradeSignal,
        ltf_df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame] = None,
    ) -> ConfluenceResult:
        """Score a TradeSignal 0-100 across 4 independent institutional checks on iloc[-2]."""
        if signal.action == "HOLD":
            return ConfluenceResult(
                passed=True,
                score=0,
                htf_bias=self._evaluate_htf_bias(htf_df),
                vwap_aligned=False,
                volume_ratio=self.compute_volume_ratio(ltf_df) if ltf_df is not None and len(ltf_df) >= 3 else 0.0,
                reasons=["HOLD signal passed through untouched"],
            )

        if ltf_df is None or len(ltf_df) < 3:
            return ConfluenceResult(
                passed=False,
                score=0,
                htf_bias="UNAVAILABLE",
                vwap_aligned=False,
                volume_ratio=0.0,
                reasons=["Insufficient LTF candles for confluence check"],
            )

        score = 0
        reasons: list[str] = []
        is_buy = signal.action == "BUY"

        # 1. Higher-Timeframe (4H) Macro Alignment (+35 pts, or -20 pts counter-trend penalty)
        htf_bias = self._evaluate_htf_bias(htf_df)
        if htf_bias == "BULLISH":
            if is_buy:
                score += 35
                reasons.append("4H Bullish Aligned")
            else:
                score -= 20
                reasons.append("Counter-HTF warning (4H Bullish)")
        elif htf_bias == "BEARISH":
            if not is_buy:
                score += 35
                reasons.append("4H Bearish Aligned")
            else:
                score -= 20
                reasons.append("Counter-HTF warning (4H Bearish)")
        elif htf_bias == "NEUTRAL":
            score += 15
            reasons.append("4H Neutral")
        else:
            # UNAVAILABLE (e.g. paper mode without HTF feed): neutral partial credit so paper loop isn't starved
            score += 20
            reasons.append("4H N/A (LTF mode)")

        # 2. VWAP Institutional Alignment (+25 pts)
        vwap_series = self.compute_rolling_vwap(ltf_df, window=24)
        closed_close = float(ltf_df["close"].astype(float).iloc[-2])
        closed_vwap = float(vwap_series.iloc[-2])

        vwap_aligned = False
        if math.isfinite(closed_close) and math.isfinite(closed_vwap) and closed_vwap > 0:
            is_mean_rev = signal.strategy_name == "mean_reversion"
            band_pct = abs(closed_close - closed_vwap) / closed_vwap
            if is_buy and (closed_close >= closed_vwap or (is_mean_rev and band_pct <= 0.003)):
                vwap_aligned = True
                score += 25
                reasons.append("Above VWAP" if closed_close >= closed_vwap else "VWAP Pullback Band")
            elif not is_buy and (closed_close <= closed_vwap or (is_mean_rev and band_pct <= 0.003)):
                vwap_aligned = True
                score += 25
                reasons.append("Below VWAP" if closed_close <= closed_vwap else "VWAP Pullback Band")
            else:
                reasons.append("VWAP Misaligned")
        else:
            reasons.append("VWAP Invalid")

        # 3. Volume Confirmation (+20 pts)
        volume_ratio = self.compute_volume_ratio(ltf_df, window=20)
        if volume_ratio >= 1.15:
            score += 20
            reasons.append(f"Vol {volume_ratio:.2f}x")
        else:
            reasons.append(f"Low Vol {volume_ratio:.2f}x")

        # 4. Signal Confidence & R:R Quality (+20 pts)
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        rr = (reward / risk) if risk > 1e-12 else 0.0
        conf = max(0.0, min(1.0, float(signal.confidence)))

        quality_pts = 0
        if rr >= 2.0:
            quality_pts += 10
        elif rr >= 1.5:
            quality_pts += 5
        quality_pts += int(round(conf * 10))
        score += min(20, quality_pts)

        # Clamp score to [0, 100]
        final_score = max(0, min(100, int(score)))
        passed = final_score >= self.min_score
        reasons.append(f"Score {final_score}/100")

        return ConfluenceResult(
            passed=passed,
            score=final_score,
            htf_bias=htf_bias,
            vwap_aligned=vwap_aligned,
            volume_ratio=round(volume_ratio, 2),
            reasons=reasons,
        )
