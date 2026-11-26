"""Automated Phase 2 verification suite: indicators, anti-repainting, regimes, strategies, risk, runner."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import math

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import sessionmaker

from app.brokers.paper_sim import PaperSimBroker
from app.database import create_db_engine, init_db
from app.engine.indicators import MIN_CANDLES, add_all_indicators, donchian_channels, rsi
from app.engine.regime import MarketRegime, RegimeDetector
from app.engine.risk_manager import RiskManager
from app.engine.runner import BotRunner
from app.engine.strategies import (
    MeanReversionStrategy,
    SafetyStandbyStrategy,
    StrategyDispatcher,
    TrendRiderStrategy,
    VolatilityBreakoutStrategy,
)
from app.models import BotState, Trade


def _make_timestamps(n: int) -> list[datetime]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(minutes=15 * i) for i in range(n)]


def _make_trending_df(n: int = 250) -> pd.DataFrame:
    """Construct a clean sustained uptrend DataFrame classified as TRENDING."""
    idx = np.arange(n, dtype=float)
    # Smooth upward trend with small oscillating pullbacks so RSI stays moderate and ADX >= 25
    base = 100.0 + idx * 0.45 + 0.6 * np.sin(idx * 0.35)
    open_p = base - 0.15
    close_p = base + 0.15
    high_p = np.maximum(open_p, close_p) + 0.35
    low_p = np.minimum(open_p, close_p) - 0.35
    return pd.DataFrame(
        {
            "timestamp": _make_timestamps(n),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": np.full(n, 200.0),
        }
    )


def _make_ranging_df(n: int = 250) -> pd.DataFrame:
    """Construct a sideways oscillating DataFrame classified as RANGING (normal volatility, ADX < 22)."""
    idx = np.arange(n, dtype=float)
    # Normal sideways oscillation with a brief wider swing earlier in the 100-bar window
    # and a slightly tighter section ~30 bars ago so current ATR% < p95 and BBW percentile is mid-range (~50%).
    amp = np.full(n, 1.5)
    amp[(idx >= n - 90) & (idx < n - 65)] = 2.2
    amp[(idx >= n - 45) & (idx < n - 25)] = 1.1
    close_p = 100.0 + amp * np.sin(idx * 0.85)
    open_p = np.roll(close_p, 1)
    open_p[0] = close_p[0]
    high_p = np.maximum(open_p, close_p) + 0.35
    low_p = np.minimum(open_p, close_p) - 0.35
    return pd.DataFrame(
        {
            "timestamp": _make_timestamps(n),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": np.full(n, 150.0),
        }
    )


def _make_squeeze_df(n: int = 250) -> pd.DataFrame:
    """Construct a volatility compression DataFrame classified as BREAKOUT_SQUEEZE."""
    idx = np.arange(n, dtype=float)
    # First 215 bars have wider oscillation, then final 35 bars compress tightly -> low BBW percentile & low ADX
    amp = np.where(idx < n - 35, 2.5, 0.08)
    close_p = 100.0 + amp * np.sin(idx * 0.7)
    open_p = np.roll(close_p, 1)
    open_p[0] = close_p[0]
    wick = np.where(idx < n - 35, 0.5, 0.03)
    high_p = np.maximum(open_p, close_p) + wick
    low_p = np.minimum(open_p, close_p) - wick
    return pd.DataFrame(
        {
            "timestamp": _make_timestamps(n),
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": np.full(n, 120.0),
        }
    )


def _make_chaotic_shock_df(n: int = 250) -> pd.DataFrame:
    """Construct a normal series with an extreme shock candle at iloc[-2] (closed candle)."""
    df = _make_ranging_df(n).copy()
    # Inject massive shock candle on iloc[-2] (high - low >> 3.2 * atr)
    c_idx = n - 2
    df.loc[c_idx, "open"] = 100.0
    df.loc[c_idx, "high"] = 125.0
    df.loc[c_idx, "low"] = 80.0
    df.loc[c_idx, "close"] = 105.0
    return df


# ---------------------------------------------------------------------------
# 1. Indicators & Edge Cases
# ---------------------------------------------------------------------------


def test_indicators_flat_price_zero_nan_crashes() -> None:
    """Flat candles (zero range, zero ATR, zero gains/losses) compute cleanly without NaN crashes."""
    n = 250
    flat_df = pd.DataFrame(
        {
            "timestamp": _make_timestamps(n),
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10.0),
        }
    )
    original_cols = list(flat_df.columns)
    enriched = add_all_indicators(flat_df)

    # Non-mutation invariant
    assert list(flat_df.columns) == original_cols
    assert enriched.attrs["sufficient_history"] is True

    closed = enriched.iloc[-2]
    assert closed["rsi"] == pytest.approx(50.0)
    assert closed["atr"] == pytest.approx(0.0)
    assert closed["atr_pct"] == pytest.approx(0.0)
    assert closed["adx"] == pytest.approx(0.0)
    assert closed["plus_di"] == pytest.approx(0.0)
    assert closed["minus_di"] == pytest.approx(0.0)
    assert closed["bb_width"] == pytest.approx(0.0)
    assert math.isfinite(float(closed["bbw_percentile"]))
    assert closed["dc_high"] == pytest.approx(100.0)
    assert closed["dc_low"] == pytest.approx(100.0)


def test_rsi_pure_uptrend_equals_100() -> None:
    """When avg_loss == 0 and avg_gain > 0, RSI must cleanly equal 100.0."""
    rising = pd.Series(np.linspace(100.0, 200.0, 60))
    rsi_series = rsi(rising, period=14)
    assert rsi_series.iloc[-2] == pytest.approx(100.0)


def test_donchian_channels_zero_lookahead_shift() -> None:
    """Donchian high/low at bar i must strictly exclude bar i itself (shifted by 1)."""
    df = _make_ranging_df(60)
    # Spike bar 58 far above prior bars
    df.loc[58, "high"] = 500.0
    dc_high, _ = donchian_channels(df, period=20)
    # Bar 58's dc_high must NOT include bar 58's 500.0 high
    assert dc_high.iloc[58] < 200.0
    # Bar 59's dc_high MUST include bar 58's 500.0 high
    assert dc_high.iloc[59] == pytest.approx(500.0)


def test_indicators_insufficient_history_flag() -> None:
    """Fewer than MIN_CANDLES (205) marks sufficient_history=False and RegimeDetector blocks."""
    short_df = _make_trending_df(100)
    enriched = add_all_indicators(short_df)
    assert enriched.attrs["sufficient_history"] is False

    res = RegimeDetector().detect(short_df)
    assert res.regime == MarketRegime.CHAOTIC_NO_TRADE
    assert "Insufficient" in res.reason


# ---------------------------------------------------------------------------
# 2. Anti-Repainting Invariant
# ---------------------------------------------------------------------------


def test_anti_repainting_mutating_forming_candle_does_not_change_output() -> None:
    """Mutating the live forming candle (iloc[-1]) wildly must produce identical RegimeResult & TradeSignal."""
    dispatcher = StrategyDispatcher()
    enabled = {"trend_rider", "mean_reversion", "volatility_breakout", "safety_standby"}

    for base_df in (_make_trending_df(), _make_ranging_df(), _make_squeeze_df(), _make_chaotic_shock_df()):
        regime_before, signal_before = dispatcher.evaluate(base_df, enabled)

        mutated = base_df.copy(deep=True)
        mutated.loc[mutated.index[-1], "open"] = 999999.0
        mutated.loc[mutated.index[-1], "high"] = 1500000.0
        mutated.loc[mutated.index[-1], "low"] = 0.0001
        mutated.loc[mutated.index[-1], "close"] = 1234567.0
        mutated.loc[mutated.index[-1], "volume"] = 99999999.0

        regime_after, signal_after = dispatcher.evaluate(mutated, enabled)
        assert regime_after == regime_before
        assert signal_after == signal_before


# ---------------------------------------------------------------------------
# 3. RegimeDetector Synthetic Classification
# ---------------------------------------------------------------------------


def test_regime_detector_classifies_all_four_regimes() -> None:
    """RegimeDetector accurately classifies TRENDING, RANGING, BREAKOUT_SQUEEZE, and CHAOTIC_NO_TRADE."""
    detector = RegimeDetector()

    trending_res = detector.detect(_make_trending_df())
    assert trending_res.regime == MarketRegime.TRENDING
    assert trending_res.adx >= 25.0

    ranging_res = detector.detect(_make_ranging_df())
    assert ranging_res.regime == MarketRegime.RANGING
    assert ranging_res.adx < 22.0

    squeeze_res = detector.detect(_make_squeeze_df())
    assert squeeze_res.regime == MarketRegime.BREAKOUT_SQUEEZE
    assert squeeze_res.bbw_percentile <= 20.0
    assert squeeze_res.adx < 25.0

    chaotic_res = detector.detect(_make_chaotic_shock_df())
    assert chaotic_res.regime == MarketRegime.CHAOTIC_NO_TRADE


# ---------------------------------------------------------------------------
# 4. Strategies & Dispatcher Toggle Enforcement
# ---------------------------------------------------------------------------


def test_disabled_strategy_always_returns_hold() -> None:
    """If the regime's matched strategy is disabled in enabled_strategy_names, dispatcher returns HOLD."""
    dispatcher = StrategyDispatcher()
    df = _make_trending_df()
    # Disable trend_rider while enabling all others
    regime_res, signal = dispatcher.evaluate(df, {"mean_reversion", "volatility_breakout", "safety_standby"})
    assert regime_res.regime == MarketRegime.TRENDING
    assert signal.action == "HOLD"
    assert signal.strategy_name == "trend_rider"
    assert signal.reason == "Strategy disabled by user"


def test_strategy_signal_math_and_safety_standby() -> None:
    """Verify each strategy produces exact SL/TP geometry on qualifying closed candles."""
    detector = RegimeDetector()

    # 1. TrendRider BUY -> SL = 1.5*ATR, TP = 3.75*ATR (1:2.5 R:R)
    trend_df = add_all_indicators(_make_trending_df())
    c = trend_df.index[-2]
    trend_df.loc[c, "ema200"] = 100.0
    trend_df.loc[c, "ema50"] = 120.0
    trend_df.loc[c, "ema21"] = 130.0
    trend_df.loc[c, "ema9"] = 135.0
    trend_df.loc[c, "close"] = 136.0
    trend_df.loc[c, "rsi"] = 58.0
    trend_df.loc[c, "atr"] = 2.0
    trend_sig = TrendRiderStrategy().evaluate(trend_df, detector.detect(_make_trending_df()))
    assert trend_sig.action == "BUY"
    assert trend_sig.stop_loss == pytest.approx(133.0)
    assert trend_sig.take_profit == pytest.approx(143.5)
    assert (trend_sig.take_profit - trend_sig.entry_price) / (
        trend_sig.entry_price - trend_sig.stop_loss
    ) == pytest.approx(2.5)

    # 2. MeanReversion BUY -> SL = 1.2*ATR, TP >= entry + 1.8*risk
    range_df = add_all_indicators(_make_ranging_df())
    c = range_df.index[-2]
    range_df.loc[c, "open"] = 98.0
    range_df.loc[c, "close"] = 98.5  # bullish close > open
    range_df.loc[c, "low"] = 97.5
    range_df.loc[c, "high"] = 98.8
    range_df.loc[c, "bb_lower"] = 97.8
    range_df.loc[c, "bb_mid"] = 99.0  # closer than 1.8*risk so floor kicks in
    range_df.loc[c, "rsi"] = 28.0
    range_df.loc[c, "atr"] = 1.0
    mr_sig = MeanReversionStrategy().evaluate(range_df, detector.detect(_make_ranging_df()))
    assert mr_sig.action == "BUY"
    assert mr_sig.stop_loss == pytest.approx(98.5 - 1.2)
    assert (mr_sig.take_profit - mr_sig.entry_price) / (
        mr_sig.entry_price - mr_sig.stop_loss
    ) >= 1.8 - 1e-9

    # 3. VolatilityBreakout BUY -> close > dc_high & range >= 1.1*ATR, SL=1.5*ATR, TP=3.5*ATR
    sq_df = add_all_indicators(_make_squeeze_df())
    c = sq_df.index[-2]
    sq_df.loc[c, "dc_high"] = 101.0
    sq_df.loc[c, "dc_low"] = 99.0
    sq_df.loc[c, "atr"] = 1.0
    sq_df.loc[c, "open"] = 100.5
    sq_df.loc[c, "low"] = 100.4
    sq_df.loc[c, "high"] = 102.0  # range = 1.6 >= 1.1
    sq_df.loc[c, "close"] = 101.8
    vb_sig = VolatilityBreakoutStrategy().evaluate(sq_df, detector.detect(_make_squeeze_df()))
    assert vb_sig.action == "BUY"
    assert vb_sig.stop_loss == pytest.approx(101.8 - 1.5)
    assert vb_sig.take_profit == pytest.approx(101.8 + 3.5)

    # 4. SafetyStandby always HOLD
    chaotic_res = detector.detect(_make_chaotic_shock_df())
    sb_sig = SafetyStandbyStrategy().evaluate(_make_chaotic_shock_df(), chaotic_res)
    assert sb_sig.action == "HOLD"
    assert sb_sig.strategy_name == "safety_standby"


# ---------------------------------------------------------------------------
# 5. RiskManager Survival Layer
# ---------------------------------------------------------------------------


def test_risk_manager_exact_1pct_sizing_and_25pct_notional_cap() -> None:
    """Verify exact 1% equity risk sizing and 25% notional cap on tight stops."""
    rm = RiskManager(max_risk_per_trade_pct=1.0, max_daily_loss_pct=3.0)

    # Equity $10,000, 1% risk = $100, stop distance = $50 -> qty = 2.0 (notional $2000 <= $2500 cap)
    dec = rm.approve_trade(
        side="BUY",
        symbol="ETHUSDT",
        entry_price=1000.0,
        stop_loss=950.0,
        take_profit=1100.0,
        equity=10000.0,
        daily_pnl_pct=0.0,
        open_positions=[],
    )
    assert dec.approved is True
    assert dec.qty == pytest.approx(2.0)
    assert dec.risk_usd == pytest.approx(100.0)
    assert dec.rr_ratio == pytest.approx(2.0)
    assert dec.trigger_kill_switch is False

    # Tiny stop ($1) -> raw_qty = 100 (notional $100,000), must be capped at 25% of $10,000 = $2,500 -> qty = 2.5
    capped = rm.approve_trade(
        side="BUY",
        symbol="ETHUSDT",
        entry_price=1000.0,
        stop_loss=999.0,
        take_profit=1002.0,
        equity=10000.0,
        daily_pnl_pct=0.0,
        open_positions=[],
    )
    assert capped.approved is True
    assert capped.qty == pytest.approx(2.5)
    assert capped.qty * 1000.0 == pytest.approx(2500.0)


def test_risk_manager_rejects_bad_sl_tp_and_low_rr() -> None:
    """Reject missing/zero/NaN/inverted SL/TP and R:R < 1.5."""
    rm = RiskManager()

    # Missing SL
    d_missing = rm.approve_trade("BUY", "BTCUSDT", 100.0, None, 110.0, 10000.0, 0.0, [])
    assert d_missing.approved is False

    # NaN SL
    d_nan = rm.approve_trade("BUY", "BTCUSDT", 100.0, float("nan"), 110.0, 10000.0, 0.0, [])
    assert d_nan.approved is False

    # Inverted SL on BUY (SL >= entry)
    d_inv_buy = rm.approve_trade("BUY", "BTCUSDT", 100.0, 102.0, 110.0, 10000.0, 0.0, [])
    assert d_inv_buy.approved is False

    # Inverted SL on SELL (SL <= entry)
    d_inv_sell = rm.approve_trade("SELL", "BTCUSDT", 100.0, 98.0, 90.0, 10000.0, 0.0, [])
    assert d_inv_sell.approved is False

    # R:R < 1.5 (risk 10, reward 12 -> R:R = 1.2)
    d_low_rr = rm.approve_trade("BUY", "BTCUSDT", 100.0, 90.0, 112.0, 10000.0, 0.0, [])
    assert d_low_rr.approved is False
    assert d_low_rr.rr_ratio == pytest.approx(1.2)


def test_risk_manager_exposure_caps_cooldown_and_kill_switch() -> None:
    """Block at 3 open trades, duplicate symbol, <3-candle cooldown, and trigger kill switch at <= -3%."""
    rm = RiskManager(max_daily_loss_pct=3.0)

    # 3 open trades blocks 4th
    three_open = [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}, {"symbol": "EUR_USD"}]
    d_max = rm.approve_trade("BUY", "XAU_USD", 100.0, 95.0, 110.0, 10000.0, 0.0, three_open)
    assert d_max.approved is False

    # Duplicate symbol blocks
    d_dup = rm.approve_trade("BUY", "BTCUSDT", 100.0, 95.0, 110.0, 10000.0, 0.0, [{"symbol": "BTCUSDT"}])
    assert d_dup.approved is False

    # 3-candle cooldown blocks at 0, 1, 2 and allows at 3
    for elapsed in (0, 1, 2):
        d_cool = rm.approve_trade(
            "BUY", "BTCUSDT", 100.0, 95.0, 110.0, 10000.0, 0.0, [], candles_since_last_loss=elapsed
        )
        assert d_cool.approved is False
    d_clear = rm.approve_trade(
        "BUY", "BTCUSDT", 100.0, 95.0, 110.0, 10000.0, 0.0, [], candles_since_last_loss=3
    )
    assert d_clear.approved is True

    # Daily loss <= -3.0% triggers kill switch FIRST
    d_kill = rm.approve_trade("BUY", "BTCUSDT", 100.0, 95.0, 110.0, 10000.0, -3.0, [])
    assert d_kill.approved is False
    assert d_kill.trigger_kill_switch is True


# ---------------------------------------------------------------------------
# 6. Trailing Stop Loss
# ---------------------------------------------------------------------------


def test_compute_trailing_sl_long_and_short_never_moves_backward() -> None:
    """Trailing SL locks in +0.1R at +1.5R profit and never moves backward."""
    rm = RiskManager()

    # LONG: entry=100, initial_risk=10 (initial SL=90)
    # Profit = +14 (< 1.5R = 15) -> unchanged
    assert rm.compute_trailing_sl("LONG", 100.0, 114.0, 90.0, 10.0) == pytest.approx(90.0)
    # Profit = +15 (== 1.5R) -> moves to 100 + 0.1*10 = 101.0
    assert rm.compute_trailing_sl("LONG", 100.0, 115.0, 90.0, 10.0) == pytest.approx(101.0)
    # Already tighter SL at 105.0 -> never moves backward to 101.0
    assert rm.compute_trailing_sl("LONG", 100.0, 120.0, 105.0, 10.0) == pytest.approx(105.0)

    # SHORT: entry=100, initial_risk=10 (initial SL=110)
    # Profit = +14 (< 1.5R) -> unchanged
    assert rm.compute_trailing_sl("SHORT", 100.0, 86.0, 110.0, 10.0) == pytest.approx(110.0)
    # Profit = +15 (== 1.5R) -> moves to 100 - 0.1*10 = 99.0
    assert rm.compute_trailing_sl("SHORT", 100.0, 85.0, 110.0, 10.0) == pytest.approx(99.0)
    # Already tighter SL at 95.0 -> never moves backward to 99.0
    assert rm.compute_trailing_sl("SHORT", 100.0, 80.0, 95.0, 10.0) == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# 7. PaperSimBroker & BotRunner Integration
# ---------------------------------------------------------------------------


def test_paper_sim_ohlcv_and_bot_runner_persist_regimes(tmp_path: Path) -> None:
    """PaperSimBroker returns evolving 250-bar DataFrames and BotRunner persists regimes + trailing SL."""
    db_file = tmp_path / "phase2_runner.db"
    engine = create_db_engine(f"sqlite:///{db_file}")
    init_db(target_engine=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    broker = PaperSimBroker(session_factory=session_factory)
    df1 = broker.get_ohlcv("BTCUSDT", interval="15m", limit=250)
    assert isinstance(df1, pd.DataFrame)
    assert len(df1) == 250
    assert list(df1.columns) == ["timestamp", "open", "high", "low", "close", "volume"]

    runner = BotRunner(broker=broker, session_factory=session_factory)
    runner.run_once()

    df2 = broker.get_ohlcv("BTCUSDT", interval="15m", limit=250)
    assert len(df2) == 250
    assert df2["timestamp"].iloc[-1] > df1["timestamp"].iloc[-1]

    valid_regimes = {r.value for r in MarketRegime}
    db = session_factory()
    try:
        state = db.query(BotState).first()
        assert state is not None
        assert state.active_regime_bybit in valid_regimes
        assert state.active_regime_forex in valid_regimes

        # Verify trailing SL updates an open trade in profit >= 1.5R
        open_trade = db.query(Trade).filter(Trade.status == "OPEN", Trade.side == "LONG").first()
        if open_trade is not None:
            initial_r = abs(open_trade.entry_price - open_trade.stop_loss)
            open_trade.current_price = open_trade.entry_price + 2.0 * initial_r
            db.commit()
            runner._apply_trailing_stops(db)
            db.commit()
            db.refresh(open_trade)
            assert open_trade.stop_loss >= open_trade.entry_price + 0.1 * initial_r - 0.02
    finally:
        db.close()
