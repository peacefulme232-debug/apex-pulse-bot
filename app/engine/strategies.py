"""Four pluggable regime-matched strategies and StrategyDispatcher."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
import pandas as pd

from app.engine.indicators import add_all_indicators
from app.engine.regime import MarketRegime, RegimeDetector, RegimeResult

REQUIRED_STRATEGY_COLUMNS = (
    "ema9",
    "ema21",
    "ema50",
    "ema200",
    "rsi",
    "atr",
    "atr_pct",
    "adx",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bbw_percentile",
    "dc_high",
    "dc_low",
)


@dataclass(frozen=True)
class TradeSignal:
    """Standardized signal emitted by a strategy on the last closed candle."""

    action: Literal["BUY", "SELL", "HOLD"]
    strategy_name: str
    regime: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str


def _valid(*values: float) -> bool:
    return all(isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)) for v in values)


def _hold(strategy_name: str, regime: MarketRegime, reason: str, entry_price: float = 0.0) -> TradeSignal:
    return TradeSignal(
        action="HOLD",
        strategy_name=strategy_name,
        regime=regime.value,
        confidence=0.0,
        entry_price=entry_price,
        stop_loss=0.0,
        take_profit=0.0,
        reason=reason,
    )


class BaseStrategy(ABC):
    """Abstract base class for regime-specific trading strategies."""

    name: str
    target_regime: MarketRegime

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, regime_result: RegimeResult) -> TradeSignal:
        """Evaluate strictly on the last closed candle (df.iloc[-2])."""


class TrendRiderStrategy(BaseStrategy):
    """Momentum trend-following strategy for TRENDING regimes."""

    name = "trend_rider"
    target_regime = MarketRegime.TRENDING

    def evaluate(self, df: pd.DataFrame, regime_result: RegimeResult) -> TradeSignal:
        candle = df.iloc[-2]
        close = float(candle["close"])
        ema9 = float(candle["ema9"])
        ema21 = float(candle["ema21"])
        ema50 = float(candle["ema50"])
        ema200 = float(candle["ema200"])
        rsi_val = float(candle["rsi"])
        atr_val = float(candle["atr"])

        if not _valid(close, ema9, ema21, ema50, ema200, rsi_val, atr_val) or atr_val <= 1e-12 or close <= 0.0:
            return _hold(self.name, regime_result.regime, "Invalid indicator or zero ATR on closed candle.", close)

        if ema50 > ema200 and ema9 > ema21 and close > ema21 and 45.0 <= rsi_val <= 68.0:
            sl = close - 1.5 * atr_val
            tp = close + 3.75 * atr_val
            return TradeSignal(
                action="BUY",
                strategy_name=self.name,
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                reason=(
                    f"Trend Rider BUY: EMA50>EMA200 & EMA9>EMA21, close {close:.5f} > EMA21, "
                    f"RSI {rsi_val:.1f} in [45, 68]."
                ),
            )

        if ema50 < ema200 and ema9 < ema21 and close < ema21 and 32.0 <= rsi_val <= 55.0:
            sl = close + 1.5 * atr_val
            tp = close - 3.75 * atr_val
            return TradeSignal(
                action="SELL",
                strategy_name=self.name,
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                reason=(
                    f"Trend Rider SELL: EMA50<EMA200 & EMA9<EMA21, close {close:.5f} < EMA21, "
                    f"RSI {rsi_val:.1f} in [32, 55]."
                ),
            )

        return _hold(
            self.name,
            regime_result.regime,
            f"No aligned trend pullback setup (RSI={rsi_val:.1f}).",
            close,
        )


class MeanReversionStrategy(BaseStrategy):
    """Bollinger Band + RSI rejection fade strategy for RANGING regimes."""

    name = "mean_reversion"
    target_regime = MarketRegime.RANGING

    def evaluate(self, df: pd.DataFrame, regime_result: RegimeResult) -> TradeSignal:
        candle = df.iloc[-2]
        open_p = float(candle["open"])
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        bb_upper = float(candle["bb_upper"])
        bb_mid = float(candle["bb_mid"])
        bb_lower = float(candle["bb_lower"])
        rsi_val = float(candle["rsi"])
        atr_val = float(candle["atr"])

        if (
            not _valid(open_p, high, low, close, bb_upper, bb_mid, bb_lower, rsi_val, atr_val)
            or atr_val <= 1e-12
            or close <= 0.0
        ):
            return _hold(self.name, regime_result.regime, "Invalid indicator or zero ATR on closed candle.", close)

        body = abs(close - open_p)
        lower_wick = min(open_p, close) - low
        upper_wick = high - max(open_p, close)
        risk = 1.2 * atr_val

        bullish_rejection = (close > open_p) or (lower_wick > body)
        bearish_rejection = (close < open_p) or (upper_wick > body)

        if low <= bb_lower and rsi_val < 32.0 and bullish_rejection:
            sl = close - risk
            tp = max(bb_mid, close + 1.8 * risk)
            return TradeSignal(
                action="BUY",
                strategy_name=self.name,
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                reason=(
                    f"Mean Reversion BUY: low {low:.5f} <= BB lower {bb_lower:.5f}, "
                    f"RSI {rsi_val:.1f} < 32 with bullish rejection."
                ),
            )

        if high >= bb_upper and rsi_val > 68.0 and bearish_rejection:
            sl = close + risk
            tp = min(bb_mid, close - 1.8 * risk)
            return TradeSignal(
                action="SELL",
                strategy_name=self.name,
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                reason=(
                    f"Mean Reversion SELL: high {high:.5f} >= BB upper {bb_upper:.5f}, "
                    f"RSI {rsi_val:.1f} > 68 with bearish rejection."
                ),
            )

        return _hold(
            self.name,
            regime_result.regime,
            f"No Bollinger/RSI extreme rejection setup (RSI={rsi_val:.1f}).",
            close,
        )


class VolatilityBreakoutStrategy(BaseStrategy):
    """Donchian Channel + ATR expansion breakout strategy for BREAKOUT_SQUEEZE regimes."""

    name = "volatility_breakout"
    target_regime = MarketRegime.BREAKOUT_SQUEEZE

    def evaluate(self, df: pd.DataFrame, regime_result: RegimeResult) -> TradeSignal:
        candle = df.iloc[-2]
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        dc_high = float(candle["dc_high"])
        dc_low = float(candle["dc_low"])
        atr_val = float(candle["atr"])

        if not _valid(high, low, close, dc_high, dc_low, atr_val) or atr_val <= 1e-12 or close <= 0.0:
            return _hold(self.name, regime_result.regime, "Invalid indicator or zero ATR on closed candle.", close)

        candle_range = abs(high - low)

        if close > dc_high and candle_range >= 1.1 * atr_val:
            sl = close - 1.5 * atr_val
            tp = close + 3.5 * atr_val
            return TradeSignal(
                action="BUY",
                strategy_name=self.name,
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                reason=(
                    f"Volatility Breakout BUY: close {close:.5f} > prior 20-bar high {dc_high:.5f} "
                    f"with range {candle_range:.5f} >= 1.1x ATR."
                ),
            )

        if close < dc_low and candle_range >= 1.1 * atr_val:
            sl = close + 1.5 * atr_val
            tp = close - 3.5 * atr_val
            return TradeSignal(
                action="SELL",
                strategy_name=self.name,
                regime=regime_result.regime.value,
                confidence=regime_result.confidence,
                entry_price=close,
                stop_loss=sl,
                take_profit=tp,
                reason=(
                    f"Volatility Breakout SELL: close {close:.5f} < prior 20-bar low {dc_low:.5f} "
                    f"with range {candle_range:.5f} >= 1.1x ATR."
                ),
            )

        return _hold(
            self.name,
            regime_result.regime,
            "Squeeze active; awaiting confirmed Donchian expansion breakout.",
            close,
        )


class SafetyStandbyStrategy(BaseStrategy):
    """Capital preservation standby strategy for CHAOTIC_NO_TRADE regimes."""

    name = "safety_standby"
    target_regime = MarketRegime.CHAOTIC_NO_TRADE

    def evaluate(self, df: pd.DataFrame, regime_result: RegimeResult) -> TradeSignal:
        close = float(df.iloc[-2]["close"]) if df is not None and len(df) >= 2 else 0.0
        if not _valid(close):
            close = 0.0
        return _hold(
            self.name,
            regime_result.regime,
            f"Capital preservation active: {regime_result.reason}",
            close,
        )


class StrategyDispatcher:
    """Routes market data through RegimeDetector to the single matching strategy."""

    def __init__(self, regime_detector: RegimeDetector | None = None) -> None:
        self.regime_detector = regime_detector or RegimeDetector()
        self._strategies: dict[MarketRegime, BaseStrategy] = {
            MarketRegime.TRENDING: TrendRiderStrategy(),
            MarketRegime.RANGING: MeanReversionStrategy(),
            MarketRegime.BREAKOUT_SQUEEZE: VolatilityBreakoutStrategy(),
            MarketRegime.CHAOTIC_NO_TRADE: SafetyStandbyStrategy(),
        }

    def evaluate(
        self,
        df: pd.DataFrame,
        enabled_strategy_names: set[str],
    ) -> tuple[RegimeResult, TradeSignal]:
        """Run RegimeDetector, select matching strategy, enforce DB toggles, return result."""
        if df is not None and len(df) > 0 and all(col in df.columns for col in REQUIRED_STRATEGY_COLUMNS):
            enriched = df
        else:
            enriched = add_all_indicators(df)

        regime_result = self.regime_detector.detect(enriched)
        strategy = self._strategies[regime_result.regime]

        # CHAOTIC_NO_TRADE always blocks new entries via SafetyStandbyStrategy
        if regime_result.regime == MarketRegime.CHAOTIC_NO_TRADE:
            return regime_result, strategy.evaluate(enriched, regime_result)

        if strategy.name not in enabled_strategy_names:
            close = float(enriched.iloc[-2]["close"]) if len(enriched) >= 2 else 0.0
            if not _valid(close):
                close = 0.0
            return regime_result, _hold(
                strategy.name,
                regime_result.regime,
                "Strategy disabled by user",
                close,
            )

        signal = strategy.evaluate(enriched, regime_result)
        return regime_result, signal
