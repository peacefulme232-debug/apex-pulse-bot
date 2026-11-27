"""Comprehensive tests for the 3 synced strategies:
1. EmaPullbackStrategy (50 EMA Pullback — Daily Driver)
2. BreakoutRetestStrategy (H1/H4 Breakout + Retest — Anti-FOMO State Machine)
3. RsiDivergenceSniperStrategy (H4/D1 RSI 14 Divergence — Sniper + Veto)
"""

from __future__ import annotations

import math
from typing import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import ArmedRetestSetup
from app.services.indicators import (
    Candle,
    SRZone,
    adx_wilder,
    atr_wilder,
    candle_metrics,
    ema,
    rsi_wilder,
)
from app.services.regime_router import RegimeState
from app.services.strategies import (
    BreakoutRetestStrategy,
    CandidateSignal,
    EmaPullbackStrategy,
    RsiDivergenceSniperStrategy,
)


def _c(
    o: float,
    h: float,
    l: float,
    c: float,
    v: float = 100.0,
    is_closed: bool = True,
) -> Candle:
    return Candle(
        open=o,
        high=max(h, o, c),
        low=min(l, o, c),
        close=c,
        volume=v,
        is_closed=is_closed,
    )


def _trend_candles(
    n: int = 230,
    start: float = 100.0,
    step: float = 0.5,
    wick: float = 0.4,
) -> list[Candle]:
    candles: list[Candle] = []
    price = start
    for _ in range(n):
        o = price
        c = price + step
        h = max(o, c) + wick
        l = min(o, c) - wick
        candles.append(_c(o, h, l, c))
        price = c
    return candles


@pytest.fixture()
def db_session() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


# ===========================================================================
# 1. STRATEGY 1: 50 EMA PULLBACK
# ===========================================================================


def _build_bullish_pullback_series() -> tuple[list[Candle], list[Candle]]:
    """Construct closed H4 and H1 candles with a textbook 50 EMA pullback rejection."""
    h4 = _trend_candles(n=220, start=100.0, step=0.6, wick=0.4)

    # Strong H1 uptrend (220 bars) so EMA50 > EMA200, slope > 0, ADX >= 22
    h1 = _trend_candles(n=220, start=100.0, step=0.55, wick=0.45)

    # Controlled multi-bar pullback toward H1 EMA 50 to cool RSI into [38, 58]
    # while keeping EMA50 > EMA200, slope > 0, and ADX >= 22.
    ema50_now = ema([c.close for c in h1], 50)[-1]
    assert ema50_now is not None
    price = h1[-1].close
    # Realistic stair-step pullback toward EMA 50 (keeps RSI in ~44-52 healthy zone)
    target_near_ema = ema50_now + 1.0
    total_drop = price - target_near_ema
    weights = [-0.24, 0.08, -0.24, 0.08, -0.24, 0.08, -0.22, 0.08, -0.22, -0.16]
    # Normalize weights so net movement equals -total_drop
    net = -sum(weights)
    for w in weights:
        o = price
        c = price + total_drop * (w / net)
        h1.append(_c(o, max(o, c) + 0.35, min(o, c) - 0.35, c))
        price = c

    # Recompute exact EMA50 and ATR right before final rejection candle
    cur_ema50 = ema([c.close for c in h1], 50)[-1]
    cur_atr = atr_wilder([c.high for c in h1], [c.low for c in h1], [c.close for c in h1], 14)[-1]
    assert cur_ema50 is not None and cur_atr is not None

    # Final closed candle: wicks down into 50 EMA zone and prints a bullish rejection pin bar
    # closing above EMA 50.
    o = cur_ema50 + 0.10 * cur_atr
    c = cur_ema50 + 0.35 * cur_atr
    l = cur_ema50 - 0.55 * cur_atr  # lower wick >= 1.5 * body, low <= ema50 + 0.35*ATR
    h = c + 0.08 * cur_atr
    h1.append(_c(o, h, l, c))

    return h1, h4


def _build_bearish_pullback_series() -> tuple[list[Candle], list[Candle]]:
    """Construct closed H4 and H1 candles with a textbook bearish 50 EMA pullback rejection."""
    h4 = _trend_candles(n=220, start=400.0, step=-0.6, wick=0.4)
    h1 = _trend_candles(n=220, start=400.0, step=-0.55, wick=0.45)

    ema50_now = ema([c.close for c in h1], 50)[-1]
    assert ema50_now is not None
    price = h1[-1].close
    target_near_ema = ema50_now - 1.0
    total_rise = target_near_ema - price
    weights = [0.24, -0.08, 0.24, -0.08, 0.24, -0.08, 0.22, -0.08, 0.22, 0.16]
    net = sum(weights)
    for w in weights:
        o = price
        c = price + total_rise * (w / net)
        h1.append(_c(o, max(o, c) + 0.35, min(o, c) - 0.35, c))
        price = c

    cur_ema50 = ema([c.close for c in h1], 50)[-1]
    cur_atr = atr_wilder([c.high for c in h1], [c.low for c in h1], [c.close for c in h1], 14)[-1]
    assert cur_ema50 is not None and cur_atr is not None

    # Final closed candle: wicks up into 50 EMA zone and prints a bearish rejection pin bar
    # closing below EMA 50.
    o = cur_ema50 - 0.10 * cur_atr
    c = cur_ema50 - 0.35 * cur_atr
    h = cur_ema50 + 0.55 * cur_atr
    l = c - 0.08 * cur_atr
    h1.append(_c(o, h, l, c))

    return h1, h4


def test_ema_pullback_fires_on_textbook_bullish_rejection() -> None:
    h1, h4 = _build_bullish_pullback_series()

    # Sanity-check our synthetic series matches textbook conditions
    h1_closes = [c.close for c in h1]
    h1_rsi = rsi_wilder(h1_closes, 14)[-1]
    h1_adx = adx_wilder([c.high for c in h1], [c.low for c in h1], h1_closes, 14)[0][-1]
    h1_atr = atr_wilder([c.high for c in h1], [c.low for c in h1], h1_closes, 14)[-1]
    assert h1_rsi is not None and 38.0 <= h1_rsi <= 58.0
    assert h1_adx is not None and h1_adx >= 22.0
    assert h1_atr is not None

    strat = EmaPullbackStrategy()
    signal = strat.evaluate(
        symbol="BTCUSDT",
        market="bybit",
        regime=RegimeState.TRENDING_BULL,
        h1_candles=h1,
        h4_candles=h4,
    )

    assert isinstance(signal, CandidateSignal)
    assert signal.symbol == "BTCUSDT"
    assert signal.market == "bybit"
    assert signal.strategy == "EMA_50_PULLBACK"
    assert signal.side == "LONG"
    assert signal.order_exit_type == "TAKE_PROFIT"
    assert signal.entry_price == pytest.approx(h1[-1].close)
    assert signal.sl_price < signal.entry_price < signal.tp_price
    assert signal.risk_distance >= 0.8 * h1_atr - 1e-9
    assert signal.rr_ratio >= 2.2 - 1e-6


def test_ema_pullback_fires_on_textbook_bearish_rejection() -> None:
    h1, h4 = _build_bearish_pullback_series()
    strat = EmaPullbackStrategy()

    signal = strat.evaluate(
        symbol="EURUSD",
        market="forex",
        regime="TRENDING_BEAR",
        h1_candles=h1,
        h4_candles=h4,
    )

    assert isinstance(signal, CandidateSignal)
    assert signal.strategy == "EMA_50_PULLBACK"
    assert signal.side == "SHORT"
    assert signal.order_exit_type == "TAKE_PROFIT"
    assert signal.tp_price < signal.entry_price < signal.sl_price
    assert signal.rr_ratio >= 2.2 - 1e-6


def test_ema_pullback_returns_none_when_rsi_overbought() -> None:
    """Rejects overbought RSI (> 65) even if candle wicks down to EMA 50 with a rejection shape."""
    h4 = _trend_candles(n=220, start=100.0, step=0.6, wick=0.4)
    # Pure uninterrupted uptrend -> RSI will be well above 65 (near 90-100)
    h1 = _trend_candles(n=225, start=100.0, step=0.8, wick=0.3)

    cur_ema50 = ema([c.close for c in h1], 50)[-1]
    cur_atr = atr_wilder([c.high for c in h1], [c.low for c in h1], [c.close for c in h1], 14)[-1]
    assert cur_ema50 is not None and cur_atr is not None

    # Huge lower wick touching EMA 50 while closing high so RSI stays > 65
    o = h1[-1].close
    c = o + 0.5
    l = cur_ema50 - 0.1 * cur_atr
    h = c + 0.1
    h1.append(_c(o, h, l, c))

    rsi_val = rsi_wilder([x.close for x in h1], 14)[-1]
    assert rsi_val is not None and rsi_val > 65.0

    strat = EmaPullbackStrategy()
    assert (
        strat.evaluate(
            symbol="BTCUSDT",
            market="bybit",
            regime=RegimeState.TRENDING_BULL,
            h1_candles=h1,
            h4_candles=h4,
        )
        is None
    )


def test_ema_pullback_returns_none_in_dead_chop_regime() -> None:
    h1, h4 = _build_bullish_pullback_series()
    strat = EmaPullbackStrategy()

    assert (
        strat.evaluate(
            symbol="BTCUSDT",
            market="bybit",
            regime=RegimeState.DEAD_CHOP,
            h1_candles=h1,
            h4_candles=h4,
        )
        is None
    )
    assert (
        strat.evaluate(
            symbol="BTCUSDT",
            market="bybit",
            regime="DEAD_CHOP",
            h1_candles=h1,
            h4_candles=h4,
        )
        is None
    )


# ===========================================================================
# 2. STRATEGY 2: BREAKOUT + RETEST (ANTI-FOMO STATE MACHINE)
# ===========================================================================


def _build_range_base_candles(n: int = 40, center: float = 100.0) -> tuple[list[Candle], SRZone]:
    """Build steady range candles with predictable ATR ~ 1.0 and a clear resistance zone."""
    candles: list[Candle] = []
    for i in range(n):
        offset = 0.25 if (i % 2 == 0) else -0.25
        o = center + offset
        c = center - offset
        candles.append(_c(o, center + 0.5, center - 0.5, c))
    zone = SRZone(zone_low=100.2, zone_high=100.8, touches=3, last_touch_index=n - 5)
    return candles, zone


def test_breakout_retest_arms_on_breakout_and_triggers_only_on_low_momentum_retest(
    db_session: Session,
) -> None:
    strat = BreakoutRetestStrategy()
    candles, zone = _build_range_base_candles(40, center=100.0)

    # Stage A: Strong breakout candle (range=2.2 >= 1.15*ATR, body_ratio=2.0/2.2=0.91 >= 0.55,
    # close=102.5 > 100.8 + 0.25*ATR)
    breakout_candle = _c(o=100.5, h=102.6, l=100.4, c=102.5)
    candles.append(breakout_candle)

    sig_on_breakout = strat.evaluate(
        db_session,
        symbol="ETHUSDT",
        market="bybit",
        candles=candles,
        zones=[zone],
    )
    # Anti-FOMO: MUST return None on breakout candle and persist ARMED row
    assert sig_on_breakout is None

    armed = strat.get_armed_setup(db_session, symbol="ETHUSDT", market="bybit")
    assert armed is not None
    assert armed.status == "ARMED"
    assert armed.side == "LONG"
    assert armed.bars_remaining == 10
    assert float(armed.zone_low) == pytest.approx(zone.zone_low)
    assert float(armed.zone_high) == pytest.approx(zone.zone_high)

    # Bar +1: High-momentum retest candle (range=2.5 >= breakout_range 2.2) -> MUST NOT enter
    high_mom_retest = _c(o=100.7, h=103.0, l=100.5, c=101.6)
    candles.append(high_mom_retest)
    sig_high_mom = strat.evaluate(
        db_session,
        symbol="ETHUSDT",
        market="bybit",
        candles=candles,
        zones=[zone],
    )
    assert sig_high_mom is None
    db_session.refresh(armed)
    assert armed.status == "ARMED"
    assert armed.bars_remaining == 9

    # Bar +2: Valid low-momentum retest + bullish rejection/continuation close above zone_high
    # range = 1.2 < breakout_range (2.2), low=100.4 <= zone_high + 0.30*ATR, close=101.5 > open & > zone_high
    retest_candle = _c(o=100.9, h=101.6, l=100.4, c=101.5)
    candles.append(retest_candle)

    sig_entry = strat.evaluate(
        db_session,
        symbol="ETHUSDT",
        market="bybit",
        candles=candles,
        zones=[zone],
    )
    assert isinstance(sig_entry, CandidateSignal)
    assert sig_entry.strategy == "BREAKOUT_RETEST"
    assert sig_entry.side == "LONG"
    assert sig_entry.order_exit_type == "TAKE_PROFIT"
    assert sig_entry.entry_price == pytest.approx(101.5)
    assert sig_entry.rr_ratio >= 2.4 - 1e-6

    cur_atr = atr_wilder(
        [c.high for c in candles],
        [c.low for c in candles],
        [c.close for c in candles],
        14,
    )[-1]
    assert cur_atr is not None
    expected_sl = min(retest_candle.low, zone.zone_low) - 0.35 * cur_atr
    assert sig_entry.sl_price == pytest.approx(expected_sl)

    db_session.refresh(armed)
    assert armed.status == "TRIGGERED"
    assert strat.get_armed_setup(db_session, symbol="ETHUSDT", market="bybit") is None


def test_breakout_retest_cancels_cleanly_on_fakeout_close_inside_range(
    db_session: Session,
) -> None:
    strat = BreakoutRetestStrategy()
    candles, zone = _build_range_base_candles(40, center=100.0)

    # Arm breakout
    candles.append(_c(o=100.5, h=102.6, l=100.4, c=102.5))
    assert (
        strat.evaluate(db_session, symbol="SOLUSDT", market="bybit", candles=candles, zones=[zone])
        is None
    )

    armed = strat.get_armed_setup(db_session, symbol="SOLUSDT", market="bybit")
    assert armed is not None
    setup_id = armed.id

    # Fakeout candle: slams back inside old range (close=99.2 < zone_low 100.2 - 0.30*ATR)
    candles.append(_c(o=102.3, h=102.4, l=99.0, c=99.2))
    sig = strat.evaluate(db_session, symbol="SOLUSDT", market="bybit", candles=candles, zones=[zone])
    assert sig is None

    row = db_session.query(ArmedRetestSetup).filter(ArmedRetestSetup.id == setup_id).one()
    assert row.status == "INVALIDATED"
    assert strat.get_armed_setup(db_session, symbol="SOLUSDT", market="bybit") is None


def test_breakout_retest_expires_after_bars_remaining_exhausted(
    db_session: Session,
) -> None:
    strat = BreakoutRetestStrategy(arm_bars=3)
    candles, zone = _build_range_base_candles(40, center=100.0)

    # Arm breakout
    candles.append(_c(o=100.5, h=102.6, l=100.4, c=102.5))
    assert (
        strat.evaluate(db_session, symbol="XRPUSDT", market="bybit", candles=candles, zones=[zone])
        is None
    )
    armed = strat.get_armed_setup(db_session, symbol="XRPUSDT", market="bybit")
    assert armed is not None
    setup_id = armed.id

    # 3 bars hovering far above the zone without retesting
    for _ in range(3):
        candles.append(_c(o=103.5, h=103.8, l=103.2, c=103.6))
        sig = strat.evaluate(
            db_session, symbol="XRPUSDT", market="bybit", candles=candles, zones=[zone]
        )
        assert sig is None

    row = db_session.query(ArmedRetestSetup).filter(ArmedRetestSetup.id == setup_id).one()
    assert row.status == "EXPIRED"
    assert row.bars_remaining == 0


# ===========================================================================
# 3. STRATEGY 3: RSI(14) DIVERGENCE SNIPER + VETO
# ===========================================================================


def _build_bearish_divergence_htf() -> list[Candle]:
    """Construct closed HTF candles with confirmed regular bearish RSI(14) divergence.

    P1: sharp impulse rally to high peak (RSI >= 65).
    Pullback: cool-off trough.
    P2: slower grind to higher high (P2.price > P1.price + 0.25*ATR) with lower RSI
        (P2.rsi < P1.rsi - 2.5), followed by 3 lower closed confirmation bars (right=3).
    """
    candles: list[Candle] = []
    price = 100.0

    # Warmup flat base (20 bars)
    for i in range(20):
        delta = 0.2 if i % 2 == 0 else -0.2
        candles.append(_c(price, price + 0.6, price - 0.6, price + delta))
        price += delta

    # Impulse rally to P1 (10 strong bullish bars -> RSI shoots well above 70)
    for _ in range(10):
        o = price
        c = price + 2.2
        candles.append(_c(o, c + 0.4, o - 0.2, c))
        price = c
    # P1 peak bar (index 30)
    p1_high = price + 2.6
    candles.append(_c(price, p1_high, price - 0.2, p1_high - 0.2))
    price = p1_high - 0.2

    # Pullback (6 bars down)
    for _ in range(6):
        o = price
        c = price - 1.5
        candles.append(_c(o, o + 0.2, c - 0.3, c))
        price = c

    # Grind up to higher high P2 with alternating mild red/green bars so RSI stays lower than P1
    steps = [1.4, -0.5, 1.5, -0.4, 1.6, -0.5, 1.7, -0.4, 1.8, -0.3, 2.0, -0.4, 2.2]
    for s in steps:
        o = price
        c = price + s
        candles.append(_c(o, max(o, c) + 0.3, min(o, c) - 0.3, c))
        price = c

    # Ensure P2 peak exceeds P1 by > 0.25 * ATR
    p2_high = max(price + 1.5, p1_high + 2.5)
    candles.append(_c(price, p2_high, price - 0.2, p2_high - 0.2))
    price = p2_high - 0.2

    # 3 strictly closed confirmation bars after P2 (right=3)
    for _ in range(3):
        o = price
        c = price - 1.2
        candles.append(_c(o, o + 0.1, c - 0.3, c))
        price = c

    return candles


def _build_bullish_divergence_htf() -> list[Candle]:
    """Construct closed HTF candles with confirmed regular bullish RSI(14) divergence."""
    candles: list[Candle] = []
    price = 200.0

    # Warmup flat base (20 bars)
    for i in range(20):
        delta = 0.2 if i % 2 == 0 else -0.2
        candles.append(_c(price, price + 0.6, price - 0.6, price + delta))
        price += delta

    # Sharp selloff to P1 (10 strong bearish bars -> RSI plunges well below 30)
    for _ in range(10):
        o = price
        c = price - 2.2
        candles.append(_c(o, o + 0.2, c - 0.4, c))
        price = c
    p1_low = price - 2.6
    candles.append(_c(price, price + 0.2, p1_low, p1_low + 0.2))
    price = p1_low + 0.2

    # Bounce (6 bars up)
    for _ in range(6):
        o = price
        c = price + 1.5
        candles.append(_c(o, c + 0.3, o - 0.2, c))
        price = c

    # Grind down to lower low P2 with alternating bars so RSI stays higher than P1
    steps = [-1.4, 0.5, -1.5, 0.4, -1.6, 0.5, -1.7, 0.4, -1.8, 0.3, -2.0, 0.4, -2.2]
    for s in steps:
        o = price
        c = price + s
        candles.append(_c(o, max(o, c) + 0.3, min(o, c) - 0.3, c))
        price = c

    p2_low = min(price - 1.5, p1_low - 2.5)
    candles.append(_c(price, price + 0.2, p2_low, p2_low + 0.2))
    price = p2_low + 0.2

    # 3 strictly closed confirmation bars after P2 (right=3)
    for _ in range(3):
        o = price
        c = price + 1.2
        candles.append(_c(o, c + 0.3, o - 0.1, c))
        price = c

    return candles


def test_rsi_divergence_detects_confirmed_bearish_and_respects_lagging_pivots() -> None:
    htf = _build_bearish_divergence_htf()
    strat = RsiDivergenceSniperStrategy()

    # Before the 3rd confirmation bar closes, P2 pivot is NOT yet confirmed -> MUST return None
    unconfirmed = htf[:-1]
    assert strat.detect_active_divergence(unconfirmed) is None

    # Once the 3rd confirmation bar closes -> confirmed BEARISH divergence
    div = strat.detect_active_divergence(htf)
    assert div is not None
    assert div.kind == "BEARISH"
    assert 5 <= div.bars_between <= 45
    assert div.p2.price > div.p1.price + 0.25 * div.atr
    assert div.p2.rsi < div.p1.rsi - 2.5
    assert div.p1.rsi >= 65.0
    assert div.confirmed_at_index == div.p2.index + 3
    assert div.strength > 0.0


def test_rsi_divergence_detects_confirmed_bullish_and_refuses_falling_knife_until_reversal() -> None:
    htf = _build_bullish_divergence_htf()
    strat = RsiDivergenceSniperStrategy()

    div = strat.detect_active_divergence(htf)
    assert div is not None
    assert div.kind == "BULLISH"
    assert 5 <= div.bars_between <= 45
    assert div.p2.price < div.p1.price - 0.25 * div.atr
    assert div.p2.rsi > div.p1.rsi + 2.5
    assert div.p1.rsi <= 35.0

    # Execution candles STILL FALLING (falling knife: bearish candle, RSI < 35)
    exec_falling: list[Candle] = []
    price = div.p2.price + 18.0
    for _ in range(20):
        o = price
        c = price - 0.7
        exec_falling.append(_c(o, o + 0.15, c - 0.15, c))
        price = c

    assert (
        strat.evaluate_sniper_entry(htf, exec_falling, symbol="BTCUSDT", market="bybit") is None
    )

    # Now price stabilizes and prints a strong bullish engulfing candle that lifts RSI > 35
    exec_confirm = list(exec_falling)
    # Build a small base so RSI sits right around ~32 before the engulfing reversal bar
    for delta in [0.4, -0.2, 0.5, -0.3, 0.6, -0.4]:
        o = price
        c = price + delta
        exec_confirm.append(_c(o, max(o, c) + 0.2, min(o, c) - 0.2, c))
        price = c

    # Prev candle is bearish, final candle is a strong bullish engulfing closing higher
    prev_bar = _c(o=price, h=price + 0.2, l=price - 0.8, c=price - 0.6)
    exec_confirm.append(prev_bar)
    engulfing_bar = _c(
        o=prev_bar.close - 0.1,
        h=prev_bar.open + 3.2,
        l=prev_bar.close - 0.2,
        c=prev_bar.open + 3.0,
    )
    exec_confirm.append(engulfing_bar)

    exec_rsi = rsi_wilder([c.close for c in exec_confirm], 14)
    assert exec_rsi[-1] is not None and exec_rsi[-2] is not None
    assert exec_rsi[-1] > exec_rsi[-2] and exec_rsi[-1] > 35.0
    assert candle_metrics(exec_confirm[-1], exec_confirm[-2]).is_bullish_engulfing

    sig = strat.evaluate_sniper_entry(htf, exec_confirm, symbol="BTCUSDT", market="bybit")
    assert isinstance(sig, CandidateSignal)
    assert sig.strategy == "RSI_DIVERGENCE_SNIPER"
    assert sig.side == "LONG"
    assert sig.order_exit_type == "TAKE_PROFIT"
    assert sig.sl_price == pytest.approx(div.p2.price - 0.45 * div.atr)
    assert sig.rr_ratio >= 2.5 - 1e-6


def test_rsi_divergence_bearish_sniper_entry_after_bearish_confirmation() -> None:
    htf = _build_bearish_divergence_htf()
    strat = RsiDivergenceSniperStrategy()
    div = strat.detect_active_divergence(htf)
    assert div is not None and div.kind == "BEARISH"

    # Execution candles rallying strongly (RSI > 65) -> MUST refuse entry while still ripping up
    exec_candles: list[Candle] = []
    price = div.p2.price - 18.0
    for _ in range(20):
        o = price
        c = price + 0.7
        exec_candles.append(_c(o, c + 0.15, o - 0.15, c))
        price = c

    assert (
        strat.evaluate_sniper_entry(htf, exec_candles, symbol="XAUUSD", market="forex") is None
    )

    # Cool slightly then print bearish engulfing candle that turns RSI downward past 65
    for delta in [-0.4, 0.2, -0.5, 0.3, -0.6, 0.4]:
        o = price
        c = price + delta
        exec_candles.append(_c(o, max(o, c) + 0.2, min(o, c) - 0.2, c))
        price = c

    prev_bar = _c(o=price, h=price + 0.8, l=price - 0.2, c=price + 0.6)
    exec_candles.append(prev_bar)
    engulfing_bar = _c(
        o=prev_bar.close + 0.1,
        h=prev_bar.close + 0.2,
        l=prev_bar.open - 3.2,
        c=prev_bar.open - 3.0,
    )
    exec_candles.append(engulfing_bar)

    exec_rsi = rsi_wilder([c.close for c in exec_candles], 14)
    assert exec_rsi[-1] is not None and exec_rsi[-2] is not None
    assert exec_rsi[-1] < exec_rsi[-2] and exec_rsi[-1] < 65.0
    assert candle_metrics(exec_candles[-1], exec_candles[-2]).is_bearish_engulfing

    sig = strat.evaluate_sniper_entry(htf, exec_candles, symbol="XAUUSD", market="forex")
    assert isinstance(sig, CandidateSignal)
    assert sig.strategy == "RSI_DIVERGENCE_SNIPER"
    assert sig.side == "SHORT"
    assert sig.order_exit_type == "TAKE_PROFIT"
    assert sig.sl_price == pytest.approx(div.p2.price + 0.45 * div.atr)
    assert sig.rr_ratio >= 2.5 - 1e-6
