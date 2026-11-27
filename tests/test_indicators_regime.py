"""Zero-repaint proof tests for app/services/indicators.py and app/services/regime_router.py."""

from __future__ import annotations

import math

import pytest

from app.services.indicators import (
    Candle,
    Pivot,
    adx_wilder,
    atr_percentile,
    atr_wilder,
    candle_metrics,
    confirmed_pivots,
    ema,
    ema_slope,
    extract_ohlcv,
    filter_closed_candles,
    key_sr_zones,
    rsi_wilder,
)
from app.services.regime_router import (
    RegimeRouter,
    RegimeState,
    classify_regime,
    detect_regular_rsi_divergence,
)


def _make_candle(
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


def _generate_wave_candles(n: int = 300, base: float = 100.0) -> list[Candle]:
    """Generate deterministic multi-swing OHLCV candles for zero-lookahead proofs."""
    candles: list[Candle] = []
    price = base
    for i in range(n):
        wave = 6.0 * math.sin(i * 0.35) + 2.5 * math.cos(i * 0.13) + 0.08 * i
        open_p = price
        close_p = base + wave
        spread = 1.2 + 0.6 * abs(math.sin(i * 0.27))
        high_p = max(open_p, close_p) + spread
        low_p = min(open_p, close_p) - spread
        candles.append(_make_candle(open_p, high_p, low_p, close_p, v=100.0 + i))
        price = close_p
    return candles


def _generate_trend_candles(
    n: int = 260,
    start: float = 100.0,
    step: float = 0.6,
    wick: float = 0.35,
) -> list[Candle]:
    """Generate a clean directional trend (positive step = uptrend, negative = downtrend)."""
    candles: list[Candle] = []
    price = start
    for _ in range(n):
        open_p = price
        close_p = price + step
        high_p = max(open_p, close_p) + wick
        low_p = min(open_p, close_p) - wick
        candles.append(_make_candle(open_p, high_p, low_p, close_p))
        price = close_p
    return candles


def _generate_sideways_chop_candles(n: int = 260, center: float = 100.0) -> list[Candle]:
    """Generate tight sideways oscillating candles that produce low ADX (< 18)."""
    candles: list[Candle] = []
    for i in range(n):
        # Alternating up/down around center so +DM and -DM cancel out -> ADX stays very low
        sign = 1.0 if (i % 2 == 0) else -1.0
        open_p = center - 0.15 * sign
        close_p = center + 0.15 * sign
        high_p = center + 0.60
        low_p = center - 0.60
        candles.append(_make_candle(open_p, high_p, low_p, close_p))
    return candles


# ---------------------------------------------------------------------------
# 1. STRICT ZERO LOOK-AHEAD PROOF
# ---------------------------------------------------------------------------


def test_strict_zero_lookahead_indicators_and_pivots() -> None:
    """Indicators & confirmed pivots on candles[:k] vs candles[:k+25] are bit-for-bit identical."""
    full_candles = _generate_wave_candles(280)
    k = 230

    # Inject a dramatic future shock in candles[k : k + 25] to catch any future peeking
    shocked_candles = list(full_candles[:k])
    last_close = shocked_candles[-1].close
    for j in range(25):
        jump = 25.0 if j % 2 == 0 else -20.0
        o = last_close
        c = max(10.0, last_close + jump)
        shocked_candles.append(_make_candle(o, max(o, c) + 8.0, min(o, c) - 8.0, c))
        last_close = c

    prefix = shocked_candles[:k]
    extended = shocked_candles[: k + 25]

    _, h_pre, l_pre, c_pre, _ = extract_ohlcv(prefix)
    _, h_ext, l_ext, c_ext, _ = extract_ohlcv(extended)

    # EMA 20 / 50 / 200 and slope
    for period in (20, 50, 200):
        e_pre = ema(c_pre, period)
        e_ext = ema(c_ext, period)
        assert e_pre == e_ext[:k], f"EMA({period}) lookahead mismatch"
        assert ema_slope(e_pre, 5) == ema_slope(e_ext, 5)[:k], f"EMA({period}) slope mismatch"

    # Wilder RSI(14)
    assert rsi_wilder(c_pre, 14) == rsi_wilder(c_ext, 14)[:k]

    # Wilder ADX / +DI / -DI
    adx_pre, pdi_pre, mdi_pre = adx_wilder(h_pre, l_pre, c_pre, 14)
    adx_ext, pdi_ext, mdi_ext = adx_wilder(h_ext, l_ext, c_ext, 14)
    assert adx_pre == adx_ext[:k]
    assert pdi_pre == pdi_ext[:k]
    assert mdi_pre == mdi_ext[:k]

    # Wilder ATR & ATR percentile
    atr_pre = atr_wilder(h_pre, l_pre, c_pre, 14)
    atr_ext = atr_wilder(h_ext, l_ext, c_ext, 14)
    assert atr_pre == atr_ext[:k]
    assert atr_percentile(atr_pre, 100) == atr_percentile(atr_ext, 100)[:k]

    # Strictly lagging confirmed pivots (left=3, right=3)
    pivots_pre = confirmed_pivots(h_pre, l_pre, left=3, right=3)
    pivots_ext = confirmed_pivots(h_ext, l_ext, left=3, right=3)
    pivots_ext_visible_at_k = [p for p in pivots_ext if p.confirmed_at_index < k]

    assert len(pivots_pre) > 0
    assert pivots_pre == pivots_ext_visible_at_k
    for p in pivots_pre:
        assert p.confirmed_at_index == p.index + 3
        assert p.confirmed_at_index < k

    # S/R zones built from confirmed pivots at k are also bit-for-bit identical
    zones_pre = key_sr_zones(pivots_pre, atr_pre[-1])
    zones_ext = key_sr_zones(pivots_ext_visible_at_k, atr_ext[k - 1])
    assert zones_pre == zones_ext


def test_pivot_invisible_until_i_plus_3_closes() -> None:
    """A swing high at index i must NOT appear until candle index i + 3 has closed."""
    # Construct a clear peak at index 5
    highs = [10, 11, 12, 13, 14, 25, 14, 13, 12, 11]
    lows = [9, 10, 11, 12, 13, 20, 13, 12, 11, 10]

    # Up to index 7 (length 8 -> indices 0..7): i+3 = 8 has NOT closed yet -> invisible
    pivots_before = confirmed_pivots(highs[:8], lows[:8], left=3, right=3)
    assert not any(p.index == 5 and p.kind == "HIGH" for p in pivots_before)

    # Once index 8 closes (length 9 -> indices 0..8): i+3 = 8 is closed -> visible!
    pivots_at_confirm = confirmed_pivots(highs[:9], lows[:9], left=3, right=3)
    matches = [p for p in pivots_at_confirm if p.index == 5 and p.kind == "HIGH"]
    assert len(matches) == 1
    assert matches[0].confirmed_at_index == 8
    assert matches[0].price == 25.0


def test_forming_candles_are_strictly_excluded() -> None:
    """Incomplete/forming candles must be dropped before indicator & regime computation."""
    h1 = _generate_trend_candles(240, start=100.0, step=0.6)
    h4 = _generate_trend_candles(240, start=100.0, step=1.8)

    # Append an extreme forming spike candle that would trigger DEAD_CHOP if included
    forming_spike = _make_candle(240.0, 400.0, 50.0, 60.0, is_closed=False)
    h1_with_forming = h1 + [forming_spike]

    assert len(filter_closed_candles(h1_with_forming)) == len(h1)

    d_clean = classify_regime("BTCUSDT", h1, h4)
    d_with_forming = classify_regime("BTCUSDT", h1_with_forming, h4)

    assert d_clean.state == RegimeState.TRENDING_BULL
    assert d_with_forming.state == d_clean.state
    assert d_with_forming.allowed_strategies == d_clean.allowed_strategies
    assert d_with_forming.telemetry == d_clean.telemetry


# ---------------------------------------------------------------------------
# 2. CHOP FILTER TEST
# ---------------------------------------------------------------------------


def test_chop_filter_sideways_oscillating_dead_chop() -> None:
    """Sideways oscillating candles (ADX < 18) classify as DEAD_CHOP with allowed_strategies == []."""
    h1 = _generate_sideways_chop_candles(260)
    h4 = _generate_sideways_chop_candles(260)

    decision = classify_regime("EURUSD", h1, h4)

    assert decision.state == RegimeState.DEAD_CHOP
    assert decision.allowed_strategies == []
    assert decision.bias == "NEUTRAL"
    assert decision.telemetry["h1_adx14"] is not None
    assert decision.telemetry["h1_adx14"] < 18.0


def test_chop_filter_abnormal_spike_candle_dead_chop() -> None:
    """Abnormal spike candle (last_candle_range > 3.2 * ATR) forces DEAD_CHOP even in a strong trend."""
    h1 = _generate_trend_candles(240, start=100.0, step=0.6, wick=0.3)
    h4 = _generate_trend_candles(240, start=100.0, step=1.8, wick=0.6)

    # Append a massive closed shock candle (> 3.2x ATR)
    last = h1[-1]
    h1.append(_make_candle(last.close, last.close + 15.0, last.close - 15.0, last.close + 2.0))

    decision = classify_regime("BTCUSDT", h1, h4)
    assert decision.state == RegimeState.DEAD_CHOP
    assert decision.allowed_strategies == []
    assert "spike" in decision.reason.lower()


def test_chop_filter_low_atr_percentile_dead_chop() -> None:
    """Compressed volatility (ATR percentile < 15) forces DEAD_CHOP cash mode."""
    # Start with wide-range uptrend, then compress bar ranges to ultra-tiny volatility
    h1 = _generate_trend_candles(180, start=100.0, step=1.5, wick=1.2)
    price = h1[-1].close
    for _ in range(60):
        h1.append(_make_candle(price, price + 0.08, price - 0.02, price + 0.06))
        price += 0.06
    h4 = _generate_trend_candles(240, start=100.0, step=1.5, wick=0.5)

    decision = classify_regime("XAUUSD", h1, h4)
    assert decision.state == RegimeState.DEAD_CHOP
    assert decision.allowed_strategies == []
    assert decision.telemetry["h1_atr_percentile"] < 15.0


# ---------------------------------------------------------------------------
# 3. TREND CLASSIFICATION TEST
# ---------------------------------------------------------------------------


def test_trend_classification_bull_and_bear() -> None:
    """Clean uptrend -> TRENDING_BULL (EMA_50_PULLBACK only); clean downtrend -> TRENDING_BEAR."""
    router = RegimeRouter()

    # Clean Uptrend
    h1_bull = _generate_trend_candles(260, start=100.0, step=0.6, wick=0.35)
    h4_bull = _generate_trend_candles(260, start=100.0, step=1.8, wick=0.8)

    bull_decision = router.classify("BTCUSDT", h1_bull, h4_bull)
    assert bull_decision.state == RegimeState.TRENDING_BULL
    assert bull_decision.allowed_strategies == ["EMA_50_PULLBACK"]
    assert bull_decision.bias == "LONG"
    assert bull_decision.veto_longs is False
    assert bull_decision.veto_shorts is False
    assert bull_decision.telemetry["h1_adx14"] >= 22.0
    assert bull_decision.telemetry["h1_plus_di"] > bull_decision.telemetry["h1_minus_di"]

    # Clean Downtrend
    h1_bear = _generate_trend_candles(260, start=500.0, step=-0.6, wick=0.35)
    h4_bear = _generate_trend_candles(260, start=500.0, step=-1.8, wick=0.8)

    bear_decision = router.classify("ETHUSDT", h1_bear, h4_bear)
    assert bear_decision.state == RegimeState.TRENDING_BEAR
    assert bear_decision.allowed_strategies == ["EMA_50_PULLBACK"]
    assert bear_decision.bias == "SHORT"
    assert bear_decision.veto_longs is False
    assert bear_decision.veto_shorts is False
    assert bear_decision.telemetry["h1_adx14"] >= 22.0
    assert bear_decision.telemetry["h1_minus_di"] > bear_decision.telemetry["h1_plus_di"]


# ---------------------------------------------------------------------------
# 4. WILDER RSI & ADX SANITY
# ---------------------------------------------------------------------------


def test_wilder_rsi_and_adx_monotonic_rally_and_selloff() -> None:
    """Monotonic rally yields RSI > 70 and +DI > -DI; monotonic selloff yields RSI < 30 and -DI > +DI."""
    rally = _generate_trend_candles(80, start=100.0, step=1.0, wick=0.2)
    _, h_r, l_r, c_r, _ = extract_ohlcv(rally)
    rsi_r = rsi_wilder(c_r, 14)
    adx_r, pdi_r, mdi_r = adx_wilder(h_r, l_r, c_r, 14)

    assert rsi_r[-1] is not None and rsi_r[-1] > 70.0
    assert adx_r[-1] is not None and adx_r[-1] > 25.0
    assert pdi_r[-1] is not None and mdi_r[-1] is not None
    assert pdi_r[-1] > mdi_r[-1]

    selloff = _generate_trend_candles(80, start=300.0, step=-1.0, wick=0.2)
    _, h_s, l_s, c_s, _ = extract_ohlcv(selloff)
    rsi_s = rsi_wilder(c_s, 14)
    adx_s, pdi_s, mdi_s = adx_wilder(h_s, l_s, c_s, 14)

    assert rsi_s[-1] is not None and rsi_s[-1] < 30.0
    assert adx_s[-1] is not None and adx_s[-1] > 25.0
    assert mdi_s[-1] is not None and pdi_s[-1] is not None
    assert mdi_s[-1] > pdi_s[-1]

    # Flat closes guard -> RSI == 50.0
    flat_rsi = rsi_wilder([100.0] * 30, 14)
    assert flat_rsi[-1] == 50.0


# ---------------------------------------------------------------------------
# 5. REVERSAL_EXHAUSTION, BREAKOUT_EXPANSION, S/R ZONES & CANDLE METRICS
# ---------------------------------------------------------------------------


def test_reversal_exhaustion_bearish_divergence_vetoes_longs() -> None:
    """Confirmed H4 regular bearish RSI(14) divergence routes to RSI_DIVERGENCE_SNIPER + veto_longs."""
    # Healthy trending H1 so DEAD_CHOP is not triggered
    h1 = _generate_trend_candles(240, start=100.0, step=0.6, wick=0.35)

    # Construct H4 series with:
    # 1) Strong rally to first swing high (high RSI)
    # 2) Sharp pullback
    # 3) Slower, choppy grind to a HIGHER price high (lower RSI)
    # 4) 3 closed pullback candles to confirm the second pivot (right=3)
    closes: list[float] = []
    p = 100.0
    # Base warmup
    for i in range(180):
        p += 0.1 if i % 2 == 0 else -0.05
        closes.append(p)
    # Strong impulsive rally to Peak 1
    for _ in range(18):
        p += 3.0
        closes.append(p)
    peak1_close = p
    # Pullback
    for _ in range(10):
        p -= 2.2
        closes.append(p)
    # Weak choppy push to slightly higher Peak 2 (higher high, lower RSI)
    steps_to_peak2 = [1.8, -0.4, 1.9, -0.5, 2.0, -0.3, 2.2, -0.2, 2.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    for s in steps_to_peak2:
        p += s
        closes.append(p)
    # Ensure peak2 strictly exceeds peak1
    if p <= peak1_close:
        p = peak1_close + 2.0
        closes.append(p)
    # 3 closed candles down to confirm Peak 2 (right=3)
    for _ in range(3):
        p -= 1.5
        closes.append(p)

    h4: list[Candle] = []
    for idx, c in enumerate(closes):
        o = closes[idx - 1] if idx > 0 else c
        h4.append(_make_candle(o, max(o, c) + 0.4, min(o, c) - 0.4, c))

    div = detect_regular_rsi_divergence(h4, "H4")
    assert div is not None
    assert div.kind == "BEARISH"
    assert div.price_2 > div.price_1
    assert div.rsi_2 < div.rsi_1

    decision = classify_regime("BTCUSDT", h1, h4)
    assert decision.state == RegimeState.REVERSAL_EXHAUSTION
    assert decision.allowed_strategies == ["RSI_DIVERGENCE_SNIPER"]
    assert decision.veto_longs is True
    assert decision.veto_shorts is False
    assert decision.bias == "SHORT"


def test_breakout_expansion_beyond_confirmed_sr_zone() -> None:
    """Strong momentum close beyond a confirmed key S/R zone classifies as BREAKOUT_EXPANSION."""
    # Build H1 with 2 confirmed swing highs near 120.0 (forming a resistance zone),
    # enough directional energy to pass ADX >= 18, then a strong breakout candle closing > zone_high + 0.25*ATR
    closes: list[float] = []
    # Warmup uptrend so ADX >= 18
    p = 80.0
    for _ in range(180):
        p += 0.20
        closes.append(p)

    # Two clear peaks at ~120.0 separated by pullback (each confirmed with left=3, right=3)
    # Peak 1 at 120.0
    for target in [117.0, 118.0, 119.0, 120.0, 118.5, 117.0, 115.5, 114.0]:
        closes.append(target)
    # Peak 2 at 120.1
    for target in [116.0, 117.5, 119.0, 120.1, 118.6, 117.2, 116.5, 117.5, 119.0]:
        closes.append(target)

    h1: list[Candle] = []
    for idx, c in enumerate(closes):
        o = closes[idx - 1] if idx > 0 else c
        h1.append(_make_candle(o, max(o, c) + 0.3, min(o, c) - 0.3, c))

    # Append strong momentum breakout candle (body_ratio >= 0.55, 1.15*ATR <= range <= 3.2*ATR)
    # Current ATR is ~1.5-2.0 -> range ~3.5 (between 1.15x and 3.2x ATR), body_ratio ~0.88
    h1.append(_make_candle(119.2, 122.9, 119.0, 122.7))

    h4 = _generate_trend_candles(240, start=80.0, step=0.2, wick=0.4)

    decision = classify_regime("SOLUSDT", h1, h4)
    assert decision.state == RegimeState.BREAKOUT_EXPANSION
    assert decision.allowed_strategies == ["BREAKOUT_RETEST"]
    assert decision.bias == "LONG"
    assert decision.telemetry["breakout_zone"] is not None


def test_key_sr_zones_clustering_and_min_touches() -> None:
    """Pivots within merge_tolerance_atr * ATR merge into a single zone; isolated pivots are dropped."""
    pivots = [
        Pivot(index=10, confirmed_at_index=13, kind="HIGH", price=100.0),
        Pivot(index=25, confirmed_at_index=28, kind="HIGH", price=100.4),
        Pivot(index=40, confirmed_at_index=43, kind="LOW", price=90.0),  # single touch -> excluded
        Pivot(index=55, confirmed_at_index=58, kind="HIGH", price=100.2),
    ]
    zones = key_sr_zones(pivots, atr=1.0, merge_tolerance_atr=0.5, min_touches=2)
    assert len(zones) == 1
    assert zones[0].zone_low == 100.0
    assert zones[0].zone_high == 100.4
    assert zones[0].touches == 3
    assert zones[0].last_touch_index == 55


def test_candle_metrics_rejection_and_engulfing() -> None:
    """Verify body/wick ratios, bullish/bearish rejection pins, and engulfing patterns."""
    # Bullish rejection pinbar: open=100, close=101, low=95, high=101.2
    bull_pin = _make_candle(100.0, 101.2, 95.0, 101.0)
    m_bull = candle_metrics(bull_pin)
    assert m_bull.lower_wick_ratio > 0.75
    assert m_bull.is_bullish_rejection is True
    assert m_bull.is_bearish_rejection is False

    # Bearish rejection shooting star: open=101, close=100, high=106, low=99.8
    bear_pin = _make_candle(101.0, 106.0, 99.8, 100.0)
    m_bear = candle_metrics(bear_pin)
    assert m_bear.upper_wick_ratio > 0.75
    assert m_bear.is_bearish_rejection is True
    assert m_bear.is_bullish_rejection is False

    # Bullish engulfing
    prev_bear = _make_candle(102.0, 102.5, 99.5, 100.0)
    curr_bull = _make_candle(99.8, 103.5, 99.5, 103.0)
    m_bull_eng = candle_metrics(curr_bull, prev_bear)
    assert m_bull_eng.is_bullish_engulfing is True
    assert m_bull_eng.is_bearish_engulfing is False

    # Bearish engulfing
    prev_bull = _make_candle(100.0, 102.5, 99.5, 102.0)
    curr_bear = _make_candle(102.2, 102.5, 99.0, 99.5)
    m_bear_eng = candle_metrics(curr_bear, prev_bull)
    assert m_bear_eng.is_bearish_engulfing is True
    assert m_bear_eng.is_bullish_engulfing is False
