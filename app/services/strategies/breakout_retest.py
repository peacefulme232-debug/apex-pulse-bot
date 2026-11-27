"""Strategy 2: H1/H4 Breakout + Retest (Anti-FOMO Persistent State Machine).

Persists armed breakout zones in the `armed_retest_setups` database table so
container restarts never forget pending setups.

2-Stage Logic (Long — exact mirror for Short):
- Stage A (Breakout -> ARM ONLY, NEVER ENTER):
  Closed candle breaks cleanly above a confirmed key S/R zone
  (close > zone_high + 0.25 * ATR, body_ratio >= 0.55, range >= 1.15 * ATR).
  Persists an ARMED setup (bars_remaining = 10) and returns None (zero FOMO).
- Stage B (Low-Momentum Retest + Rejection -> ENTER):
  On subsequent closed candles, decrements bars_remaining (expires if <= 0).
  - Fakeout Cancellation: close < zone_low - 0.30 * ATR -> INVALIDATED, returns None.
  - Valid Retest Entry: low <= zone_high + 0.30 * ATR and close >= zone_low,
    retest_range < breakout_candle_range, and bullish rejection/continuation close
    (close > open and close > zone_high) -> TRIGGERED, emits CandidateSignal.
  - SL: min(candle.low, zone_low) - 0.35 * ATR (Short: max(candle.high, zone_high) + 0.35 * ATR).
  - TP: Limit Take-Profit at >= 2.4R (order_exit_type="TAKE_PROFIT").
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

from app.models import ArmedRetestSetup
from app.services.indicators import (
    SRZone,
    atr_wilder,
    candle_metrics,
    confirmed_pivots,
    filter_closed_candles,
    key_sr_zones,
)
from app.services.strategies.base import CandidateSignal


def _to_dec(value: float) -> Decimal:
    return Decimal(f"{float(value):.8f}")


class BreakoutRetestStrategy:
    """Stateful anti-FOMO Breakout + Retest strategy backed by `armed_retest_setups`."""

    STRATEGY_NAME = "BREAKOUT_RETEST"

    def __init__(
        self,
        *,
        arm_bars: int = 10,
        breakout_close_atr: float = 0.25,
        breakout_min_body_ratio: float = 0.55,
        breakout_min_range_atr: float = 1.15,
        fakeout_atr: float = 0.30,
        retest_zone_atr: float = 0.30,
        sl_buffer_atr: float = 0.35,
        rr_multiple: float = 2.4,
        atr_period: int = 14,
        pivot_left: int = 3,
        pivot_right: int = 3,
        merge_tolerance_atr: float = 0.5,
        min_touches: int = 2,
        order_entry_type: str = "MARKET",
        order_exit_type: str = "TAKE_PROFIT",
    ) -> None:
        if rr_multiple < 2.4:
            raise ValueError(f"rr_multiple must be >= 2.4, got {rr_multiple}")
        if arm_bars < 1:
            raise ValueError(f"arm_bars must be >= 1, got {arm_bars}")

        self.arm_bars = int(arm_bars)
        self.breakout_close_atr = float(breakout_close_atr)
        self.breakout_min_body_ratio = float(breakout_min_body_ratio)
        self.breakout_min_range_atr = float(breakout_min_range_atr)
        self.fakeout_atr = float(fakeout_atr)
        self.retest_zone_atr = float(retest_zone_atr)
        self.sl_buffer_atr = float(sl_buffer_atr)
        self.rr_multiple = float(rr_multiple)
        self.atr_period = int(atr_period)
        self.pivot_left = int(pivot_left)
        self.pivot_right = int(pivot_right)
        self.merge_tolerance_atr = float(merge_tolerance_atr)
        self.min_touches = int(min_touches)
        self.order_entry_type = order_entry_type
        self.order_exit_type = order_exit_type

    def get_armed_setup(
        self,
        db: Session,
        symbol: str,
        market: str,
    ) -> Optional[ArmedRetestSetup]:
        """Fetch the active ARMED setup for `(symbol, market)`, if any."""
        return (
            db.query(ArmedRetestSetup)
            .filter(
                ArmedRetestSetup.symbol == symbol,
                ArmedRetestSetup.market == market,
                ArmedRetestSetup.status == "ARMED",
            )
            .order_by(ArmedRetestSetup.id.desc())
            .first()
        )

    def evaluate(
        self,
        db: Session,
        symbol: str,
        market: str,
        candles: Any,
        *,
        zones: Optional[Sequence[SRZone]] = None,
    ) -> Optional[CandidateSignal]:
        """Advance the Breakout + Retest state machine by one closed candle."""
        closed = filter_closed_candles(candles)
        if len(closed) < max(self.atr_period + 1, 2):
            return None

        highs = [c.high for c in closed]
        lows = [c.low for c in closed]
        closes = [c.close for c in closed]

        atr_seq = atr_wilder(highs, lows, closes, period=self.atr_period)
        cur_atr = atr_seq[-1]
        if cur_atr is None or cur_atr <= 1e-12:
            return None

        last_candle = closed[-1]
        prev_candle = closed[-2]

        armed = self.get_armed_setup(db, symbol=symbol, market=market)
        if armed is not None:
            return self._evaluate_stage_b(
                db=db,
                setup=armed,
                candle=last_candle,
                atr=cur_atr,
            )

        # No armed setup exists -> Stage A: detect clean breakout and ARM ONLY
        resolved_zones: Sequence[SRZone]
        if zones is not None:
            resolved_zones = zones
        else:
            if len(closed) < self.pivot_left + self.pivot_right + 2:
                return None
            pivots = confirmed_pivots(
                highs,
                lows,
                left=self.pivot_left,
                right=self.pivot_right,
            )
            resolved_zones = key_sr_zones(
                pivots,
                cur_atr,
                merge_tolerance_atr=self.merge_tolerance_atr,
                min_touches=self.min_touches,
            )

        if not resolved_zones:
            return None

        self._evaluate_stage_a(
            db=db,
            symbol=symbol,
            market=market,
            last_candle=last_candle,
            prev_candle=prev_candle,
            atr=cur_atr,
            zones=resolved_zones,
        )
        # Stage A NEVER enters immediately (zero FOMO chasing)
        return None

    def _evaluate_stage_a(
        self,
        *,
        db: Session,
        symbol: str,
        market: str,
        last_candle: Any,
        prev_candle: Any,
        atr: float,
        zones: Sequence[SRZone],
    ) -> Optional[ArmedRetestSetup]:
        """Stage A: detect a strong momentum breakout beyond a confirmed S/R zone and arm it."""
        metrics = candle_metrics(last_candle, prev_candle)
        candle_range = last_candle.high - last_candle.low

        if (
            metrics.body_ratio < self.breakout_min_body_ratio
            or candle_range < self.breakout_min_range_atr * atr
        ):
            return None

        # Upside breakout (LONG)
        if last_candle.close > last_candle.open:
            for zone in reversed(zones):
                threshold = zone.zone_high + self.breakout_close_atr * atr
                if last_candle.close > threshold and prev_candle.close <= threshold:
                    setup = ArmedRetestSetup(
                        market=market,
                        symbol=symbol,
                        side="LONG",
                        zone_low=_to_dec(zone.zone_low),
                        zone_high=_to_dec(zone.zone_high),
                        breakout_candle_range=_to_dec(candle_range),
                        breakout_atr=_to_dec(atr),
                        bars_remaining=self.arm_bars,
                        status="ARMED",
                    )
                    db.add(setup)
                    db.commit()
                    db.refresh(setup)
                    return setup

        # Downside breakout (SHORT)
        if last_candle.close < last_candle.open:
            for zone in zones:
                threshold = zone.zone_low - self.breakout_close_atr * atr
                if last_candle.close < threshold and prev_candle.close >= threshold:
                    setup = ArmedRetestSetup(
                        market=market,
                        symbol=symbol,
                        side="SHORT",
                        zone_low=_to_dec(zone.zone_low),
                        zone_high=_to_dec(zone.zone_high),
                        breakout_candle_range=_to_dec(candle_range),
                        breakout_atr=_to_dec(atr),
                        bars_remaining=self.arm_bars,
                        status="ARMED",
                    )
                    db.add(setup)
                    db.commit()
                    db.refresh(setup)
                    return setup

        return None

    def _evaluate_stage_b(
        self,
        *,
        db: Session,
        setup: ArmedRetestSetup,
        candle: Any,
        atr: float,
    ) -> Optional[CandidateSignal]:
        """Stage B: evaluate an armed setup for fakeout invalidation, valid retest, or expiry."""
        if setup.bars_remaining <= 0:
            setup.status = "EXPIRED"
            db.commit()
            return None

        setup.bars_remaining -= 1

        zone_low = float(setup.zone_low)
        zone_high = float(setup.zone_high)
        breakout_range = float(setup.breakout_candle_range)
        retest_range = candle.high - candle.low
        side = str(setup.side).upper()

        if side == "LONG":
            # Fakeout cancellation: closes back inside the old range
            if candle.close < zone_low - self.fakeout_atr * atr:
                setup.status = "INVALIDATED"
                db.commit()
                return None

            tests_zone = (
                candle.low <= zone_high + self.retest_zone_atr * atr
                and candle.close >= zone_low
            )
            lower_momentum = retest_range < breakout_range
            bullish_confirmation = candle.close > candle.open and candle.close > zone_high

            if tests_zone and lower_momentum and bullish_confirmation:
                entry_price = candle.close
                sl_price = min(candle.low, zone_low) - self.sl_buffer_atr * atr
                risk = entry_price - sl_price
                if risk <= 1e-12:
                    if setup.bars_remaining <= 0:
                        setup.status = "EXPIRED"
                    db.commit()
                    return None

                tp_price = entry_price + self.rr_multiple * risk
                setup.status = "TRIGGERED"
                db.commit()

                return CandidateSignal(
                    symbol=setup.symbol,
                    market=setup.market,
                    strategy=self.STRATEGY_NAME,
                    side="LONG",
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    order_entry_type=self.order_entry_type,
                    order_exit_type=self.order_exit_type,
                    reason=(
                        f"Breakout+Retest LONG triggered on zone [{zone_low:.4f}, {zone_high:.4f}]: "
                        f"retest_range={retest_range:.4f} < breakout_range={breakout_range:.4f}, "
                        f"RR={self.rr_multiple:.2f}R."
                    ),
                    metadata={
                        "armed_setup_id": setup.id,
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "breakout_candle_range": breakout_range,
                        "retest_candle_range": retest_range,
                        "breakout_atr": float(setup.breakout_atr),
                        "current_atr": atr,
                        "bars_remaining_at_trigger": setup.bars_remaining,
                        "rr_multiple": self.rr_multiple,
                    },
                )
        else:
            # Exact mirror for SHORT
            if candle.close > zone_high + self.fakeout_atr * atr:
                setup.status = "INVALIDATED"
                db.commit()
                return None

            tests_zone = (
                candle.high >= zone_low - self.retest_zone_atr * atr
                and candle.close <= zone_high
            )
            lower_momentum = retest_range < breakout_range
            bearish_confirmation = candle.close < candle.open and candle.close < zone_low

            if tests_zone and lower_momentum and bearish_confirmation:
                entry_price = candle.close
                sl_price = max(candle.high, zone_high) + self.sl_buffer_atr * atr
                risk = sl_price - entry_price
                if risk <= 1e-12:
                    if setup.bars_remaining <= 0:
                        setup.status = "EXPIRED"
                    db.commit()
                    return None

                tp_price = entry_price - self.rr_multiple * risk
                setup.status = "TRIGGERED"
                db.commit()

                return CandidateSignal(
                    symbol=setup.symbol,
                    market=setup.market,
                    strategy=self.STRATEGY_NAME,
                    side="SHORT",
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    order_entry_type=self.order_entry_type,
                    order_exit_type=self.order_exit_type,
                    reason=(
                        f"Breakout+Retest SHORT triggered on zone [{zone_low:.4f}, {zone_high:.4f}]: "
                        f"retest_range={retest_range:.4f} < breakout_range={breakout_range:.4f}, "
                        f"RR={self.rr_multiple:.2f}R."
                    ),
                    metadata={
                        "armed_setup_id": setup.id,
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "breakout_candle_range": breakout_range,
                        "retest_candle_range": retest_range,
                        "breakout_atr": float(setup.breakout_atr),
                        "current_atr": atr,
                        "bars_remaining_at_trigger": setup.bars_remaining,
                        "rr_multiple": self.rr_multiple,
                    },
                )

        if setup.bars_remaining <= 0:
            setup.status = "EXPIRED"
        db.commit()
        return None
