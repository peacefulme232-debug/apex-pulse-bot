"""Tests for 0-100 ConfluenceEvaluator, 6 Hard Vetoes, TradeManager, and Brain API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.main import create_app
from app.models import ExecutionRejection, Trade, WalletAccount
from app.services.confluence_evaluator import BrainConfig, ConfluenceEvaluator
from app.services.indicators import Candle
from app.services.realism_engine import bybit_fee_rate
from app.services.strategies import CandidateSignal, RsiDivergenceSniperStrategy
from app.services.trade_manager import TradeManager
from app.services.wallet_service import lock_wallet_account, execute_transfer


def _c(o: float, h: float, l: float, c: float, v: float = 100.0) -> Candle:
    return Candle(
        open=o,
        high=max(h, o, c),
        low=min(l, o, c),
        close=c,
        volume=v,
        is_closed=True,
    )


def _trend_candles(n: int = 230, start: float = 100.0, step: float = 0.5) -> list[Candle]:
    candles: list[Candle] = []
    price = start
    for i in range(n):
        o = price
        c = price + step
        candles.append(_c(o, max(o, c) + 0.4, min(o, c) - 0.4, c, v=100.0 + (i % 5)))
        price = c
    return candles


def _build_bearish_divergence_htf() -> list[Candle]:
    """Build closed HTF candles with a confirmed regular bearish RSI(14) divergence."""
    candles: list[Candle] = []
    price = 100.0
    for i in range(20):
        delta = 0.2 if i % 2 == 0 else -0.2
        candles.append(_c(price, price + 0.6, price - 0.6, price + delta))
        price += delta
    for _ in range(10):
        o = price
        c = price + 2.2
        candles.append(_c(o, c + 0.4, o - 0.2, c))
        price = c
    p1_high = price + 2.6
    candles.append(_c(price, p1_high, price - 0.2, p1_high - 0.2))
    price = p1_high - 0.2
    for _ in range(6):
        o = price
        c = price - 1.5
        candles.append(_c(o, o + 0.2, c - 0.3, c))
        price = c
    for s in [1.4, -0.5, 1.5, -0.4, 1.6, -0.5, 1.7, -0.4, 1.8, -0.3, 2.0, -0.4, 2.2]:
        o = price
        c = price + s
        candles.append(_c(o, max(o, c) + 0.3, min(o, c) - 0.3, c))
        price = c
    p2_high = max(price + 1.5, p1_high + 2.5)
    candles.append(_c(price, p2_high, price - 0.2, p2_high - 0.2))
    price = p2_high - 0.2
    for _ in range(3):
        o = price
        c = price - 1.2
        candles.append(_c(o, o + 0.1, c - 0.3, c))
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
# 1. FEE_DRAG_TOO_HIGH VETO + EXECUTION_REJECTIONS AUDIT LOG
# ===========================================================================


def test_tight_scalp_exceeding_15pct_fee_drag_rejected_and_logged(db_session: Session) -> None:
    """A tight scalp where fees/spread exceed 15% of TP is rejected with FEE_DRAG_TOO_HIGH."""
    wallet = lock_wallet_account(db_session, "bybit")
    wallet.active_balance = Decimal("20.000000")
    wallet.equity_baseline_day = Decimal("20.000000")
    db_session.commit()

    # Ultra-tight 0.20% TP scalp on SOLUSDT: Bybit fees (0.055% + 0.020%) + spread/slippage
    # easily consume > 50% of gross TP!
    tight_scalp = CandidateSignal(
        symbol="SOLUSDT",
        market="bybit",
        strategy="EMA_50_PULLBACK",
        side="LONG",
        entry_price=100.00,
        sl_price=99.90,
        tp_price=100.22,
        order_entry_type="MARKET",
        order_exit_type="TAKE_PROFIT",
        reason="Tight scalp test",
        metadata={"pattern": "bullish_rejection"},
    )

    evaluator = ConfluenceEvaluator(BrainConfig(max_fee_drag_pct=15.0))
    res = evaluator.evaluate(
        tight_scalp,
        db=db_session,
        h1_candles=_trend_candles(230, start=80.0, step=0.1),
        h4_candles=_trend_candles(230, start=80.0, step=0.1),
        d1_candles=_trend_candles(230, start=80.0, step=0.1),
        estimated_qty=Decimal("0.5"),
    )

    assert res.approved is False
    assert res.fee_drag_pct_of_tp > 15.0
    assert "FEE_DRAG_TOO_HIGH" in res.vetoes
    assert res.primary_veto == "FEE_DRAG_TOO_HIGH"

    logged = (
        db_session.query(ExecutionRejection)
        .filter(
            ExecutionRejection.symbol == "SOLUSDT",
            ExecutionRejection.reason_code == "FEE_DRAG_TOO_HIGH",
        )
        .all()
    )
    assert len(logged) == 1
    assert logged[0].strategy == "EMA_50_PULLBACK"
    assert logged[0].confluence_score is not None


# ===========================================================================
# 2. OPPOSING_DIVERGENCE_VETO
# ===========================================================================


def test_bearish_rsi_divergence_blocks_long_ema_pullback(db_session: Session) -> None:
    """Active H4 Bearish RSI(14) divergence blocks a Long EMA_50_PULLBACK signal."""
    wallet = lock_wallet_account(db_session, "bybit")
    wallet.active_balance = Decimal("500.000000")
    wallet.equity_baseline_day = Decimal("500.000000")
    db_session.commit()

    htf_bearish_div = _build_bearish_divergence_htf()
    div = RsiDivergenceSniperStrategy().detect_active_divergence(htf_bearish_div)
    assert div is not None and div.kind == "BEARISH"

    long_signal = CandidateSignal(
        symbol="BTCUSDT",
        market="bybit",
        strategy="EMA_50_PULLBACK",
        side="LONG",
        entry_price=50000.0,
        sl_price=49000.0,
        tp_price=52500.0,
        order_entry_type="MARKET",
        order_exit_type="TAKE_PROFIT",
        reason="Textbook EMA 50 pullback LONG",
        metadata={"pattern": "bullish_rejection"},
    )

    evaluator = ConfluenceEvaluator()
    res = evaluator.evaluate(
        long_signal,
        db=db_session,
        h1_candles=_trend_candles(230, start=40000.0, step=40.0),
        h4_candles=htf_bearish_div,
        estimated_qty=Decimal("0.01"),
    )

    assert res.approved is False
    assert "OPPOSING_DIVERGENCE_VETO" in res.vetoes
    assert res.primary_veto == "OPPOSING_DIVERGENCE_VETO"
    assert res.opposing_divergence == "BEARISH"

    logged = (
        db_session.query(ExecutionRejection)
        .filter(ExecutionRejection.reason_code == "OPPOSING_DIVERGENCE_VETO")
        .first()
    )
    assert logged is not None
    assert logged.symbol == "BTCUSDT"


# ===========================================================================
# 3. $20 MICRO-ACCOUNT SHIELD (MICRO_CORRELATION_LIMIT)
# ===========================================================================


def test_micro_account_shield_blocks_correlated_ethusdt_when_solusdt_open(
    db_session: Session,
) -> None:
    """$20 Micro-Account Shield blocks opening ETHUSDT while SOLUSDT is already open."""
    wallet = lock_wallet_account(db_session, "bybit")
    wallet.active_balance = Decimal("20.000000")
    wallet.equity_baseline_day = Decimal("20.000000")

    # Existing open SOLUSDT position on Bybit
    db_session.add(
        Trade(
            market="BYBIT",
            symbol="SOLUSDT",
            side="LONG",
            strategy_name="EMA_50_PULLBACK",
            status="OPEN",
            qty=0.2,
            entry_price=150.0,
            current_price=151.0,
            stop_loss=145.0,
            take_profit=162.0,
        )
    )
    db_session.commit()

    eth_signal = CandidateSignal(
        symbol="ETHUSDT",
        market="bybit",
        strategy="EMA_50_PULLBACK",
        side="LONG",
        entry_price=3000.0,
        sl_price=2900.0,
        tp_price=3260.0,
        order_entry_type="MARKET",
        order_exit_type="TAKE_PROFIT",
        reason="ETH pullback while SOL open",
        metadata={"pattern": "bullish_engulfing"},
    )

    evaluator = ConfluenceEvaluator(BrainConfig(micro_shield_enabled=True))
    res = evaluator.evaluate(
        eth_signal,
        db=db_session,
        h1_candles=_trend_candles(230, start=2500.0, step=2.0),
        h4_candles=_trend_candles(230, start=2500.0, step=2.0),
        d1_candles=_trend_candles(230, start=2500.0, step=2.0),
        estimated_qty=Decimal("0.01"),
    )

    assert res.approved is False
    assert "MICRO_CORRELATION_LIMIT" in res.vetoes
    assert res.primary_veto == "MICRO_CORRELATION_LIMIT"


# ===========================================================================
# 4. SMART TRADE MANAGER: +1.0R BREAKEVEN + FEES SHIELD (RUNS ONCE)
# ===========================================================================


def test_breakeven_plus_fees_shield_triggers_at_1r_and_runs_once(db_session: Session) -> None:
    """At +1.0R, SL shifts to filled_entry + round_turn_fees, sets breakeven_applied=True, runs once."""
    lock_wallet_account(db_session, "bybit")

    entry_px = Decimal("100.00000000")
    qty = 2.0
    entry_fee = entry_px * Decimal(str(qty)) * bybit_fee_rate("MARKET")  # 200 * 0.00055 = 0.11

    trade = Trade(
        market="BYBIT",
        symbol="SOLUSDT",
        side="LONG",
        strategy_name="EMA_50_PULLBACK",
        status="OPEN",
        qty=qty,
        entry_price=float(entry_px),
        filled_entry_price=entry_px,
        current_price=104.0,  # Only +0.8R (initial risk = 5.0)
        stop_loss=95.0,
        take_profit=112.0,
        order_entry_type="MARKET",
        entry_fee=entry_fee,
        breakeven_applied=False,
    )
    db_session.add(trade)
    db_session.commit()

    manager = TradeManager(BrainConfig(breakeven_r_trigger=1.0))

    # Below +1.0R -> no change
    assert manager.apply_breakeven_shield(db_session, trade) is None
    assert trade.breakeven_applied is False
    assert trade.stop_loss == pytest.approx(95.0)

    # Reach +1.0R (100.0 + 5.0 = 105.0) -> shifts SL to filled_entry + round_turn_fee_offset
    trade.current_price = 105.0
    update = manager.apply_breakeven_shield(db_session, trade)
    assert update is not None
    assert trade.breakeven_applied is True

    expected_round_turn_fees = float(entry_fee) + (100.0 * qty * float(bybit_fee_rate("STOP_LOSS")))
    expected_offset = expected_round_turn_fees / qty
    assert update.round_turn_fees_usd == pytest.approx(expected_round_turn_fees, rel=1e-6)
    assert trade.stop_loss == pytest.approx(100.0 + expected_offset, rel=1e-6)
    assert trade.stop_loss > 100.0

    # Calling again at +1.5R MUST be a no-op (strictly runs once)
    saved_sl = trade.stop_loss
    trade.current_price = 107.5
    assert manager.apply_breakeven_shield(db_session, trade) is None
    assert trade.stop_loss == pytest.approx(saved_sl)


# ===========================================================================
# 5. LOSS STREAK CIRCUIT BREAKER + WIN RESET + VAULT WITHDRAWAL ISOLATION
# ===========================================================================


def test_circuit_breaker_two_losses_win_reset_and_vault_withdrawal_never_triggers(
    db_session: Session,
) -> None:
    """2 consecutive losses lock entries for 24h; a win resets counter; vault withdrawal never trips it."""
    wallet = lock_wallet_account(db_session, "bybit")
    wallet.active_balance = Decimal("100.000000")
    wallet.equity_baseline_day = Decimal("100.000000")
    wallet.consecutive_losses = 0
    wallet.circuit_breaker_until = None
    db_session.commit()

    manager = TradeManager(BrainConfig(max_consecutive_losses=2, max_daily_loss_pct=4.0))
    now = datetime(2026, 11, 27, 14, 0, tzinfo=timezone.utc)

    # Vault withdrawal of $50 (50% of active balance!) MUST NEVER trigger circuit breaker
    execute_transfer(
        db=db_session,
        market="bybit",
        tx_type="TRADING_TO_VAULT",
        amount=Decimal("50.00"),
        network="INTERNAL",
        note="Lock profits in vault",
    )
    db_session.refresh(wallet)
    assert wallet.active_balance == Decimal("50.000000")
    assert wallet.equity_baseline_day == Decimal("50.000000")
    assert wallet.circuit_breaker_until is None

    evaluator = ConfluenceEvaluator(BrainConfig(max_consecutive_losses=2, max_daily_loss_pct=4.0))
    breaker_hit, _ = evaluator._check_circuit_breaker(
        db=db_session,
        wallet=wallet,
        current_active_equity=wallet.active_balance,
        now_utc=now,
    )
    assert breaker_hit is False

    # Loss #1 (-$0.50) -> consecutive_losses = 1, not locked yet
    loss_1 = Trade(
        market="BYBIT",
        symbol="BTCUSDT",
        side="LONG",
        strategy_name="EMA_50_PULLBACK",
        status="CLOSED",
        qty=0.01,
        entry_price=50000.0,
        current_price=49950.0,
        stop_loss=49950.0,
        take_profit=50150.0,
        pnl_usd=-0.50,
        net_pnl=Decimal("-0.500000"),
        order_exit_type="STOP_LOSS",
    )
    db_session.add(loss_1)
    db_session.flush()
    out1 = manager.record_trade_close(db_session, loss_1, now_utc=now)
    assert out1.consecutive_losses == 1
    assert out1.circuit_breaker_triggered is False

    # Winning trade (+$1.20) -> resets consecutive_losses to 0
    win_1 = Trade(
        market="BYBIT",
        symbol="BTCUSDT",
        side="LONG",
        strategy_name="EMA_50_PULLBACK",
        status="CLOSED",
        qty=0.01,
        entry_price=50000.0,
        current_price=50120.0,
        stop_loss=49950.0,
        take_profit=50120.0,
        pnl_usd=1.20,
        net_pnl=Decimal("1.200000"),
        order_exit_type="TAKE_PROFIT",
    )
    db_session.add(win_1)
    db_session.flush()
    out_win = manager.record_trade_close(db_session, win_1, now_utc=now + timedelta(minutes=10))
    assert out_win.consecutive_losses == 0
    assert out_win.circuit_breaker_triggered is False

    # Now 2 consecutive losses -> triggers 24h circuit breaker
    for idx in range(2):
        t = Trade(
            market="BYBIT",
            symbol="SOLUSDT",
            side="LONG",
            strategy_name="BREAKOUT_RETEST",
            status="CLOSED",
            qty=0.1,
            entry_price=150.0,
            current_price=146.0,
            stop_loss=146.0,
            take_profit=160.0,
            pnl_usd=-0.40,
            net_pnl=Decimal("-0.400000"),
            order_exit_type="STOP_LOSS",
        )
        db_session.add(t)
        db_session.flush()
        out = manager.record_trade_close(
            db_session, t, now_utc=now + timedelta(minutes=20 + idx)
        )

    assert out.consecutive_losses == 2
    assert out.circuit_breaker_triggered is True
    assert out.circuit_breaker_until is not None
    assert out.circuit_breaker_until >= now + timedelta(hours=23, minutes=59)

    # Evaluator now blocks new signals with LOSS_STREAK_CIRCUIT_BREAKER
    candidate = CandidateSignal(
        symbol="BTCUSDT",
        market="bybit",
        strategy="EMA_50_PULLBACK",
        side="LONG",
        entry_price=50000.0,
        sl_price=49000.0,
        tp_price=52500.0,
        metadata={"pattern": "bullish_rejection"},
    )
    eval_res = evaluator.evaluate(
        candidate,
        db=db_session,
        h1_candles=_trend_candles(230, start=40000.0, step=40.0),
        h4_candles=_trend_candles(230, start=40000.0, step=40.0),
        d1_candles=_trend_candles(230, start=40000.0, step=40.0),
        estimated_qty=Decimal("0.01"),
        now_utc=now + timedelta(hours=1),
    )
    assert eval_res.approved is False
    assert "LOSS_STREAK_CIRCUIT_BREAKER" in eval_res.vetoes


# ===========================================================================
# 6. PROTECTED BRAIN API (/api/brain/telemetry & /api/brain/config)
# ===========================================================================


def test_brain_telemetry_and_config_api_endpoints(tmp_path: Path) -> None:
    db_file = tmp_path / "brain_api.db"
    app = create_app(database_url=f"sqlite:///{db_file}")

    with TestClient(app) as client:
        # GET /api/brain/telemetry returns expected top-level structure
        resp = client.get("/api/brain/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "symbols" in data
        assert "armed_retest_setups" in data
        assert "active_divergences" in data
        assert "circuit_breaker" in data
        assert data["config"]["min_confluence_score"] == 75
        assert data["config"]["max_fee_drag_pct"] == 15.0

        # PATCH /api/brain/config updates strategy toggles, thresholds, and resets circuit breaker
        patch_resp = client.patch(
            "/api/brain/config",
            json={
                "ema_50_pullback_enabled": False,
                "min_confluence_score": 82,
                "max_fee_drag_pct": 12.5,
                "breakeven_r_trigger": 1.2,
                "max_consecutive_losses": 3,
                "micro_shield_enabled": False,
                "reset_circuit_breaker": True,
            },
        )
        assert patch_resp.status_code == 200
        updated = patch_resp.json()
        assert updated["config"]["ema_50_pullback_enabled"] is False
        assert updated["config"]["min_confluence_score"] == 82
        assert updated["config"]["max_fee_drag_pct"] == 12.5
        assert updated["config"]["breakeven_r_trigger"] == 1.2
        assert updated["config"]["max_consecutive_losses"] == 3
        assert updated["config"]["micro_shield_enabled"] is False
        assert updated["circuit_breaker"]["bybit"]["consecutive_losses"] == 0
        assert updated["circuit_breaker"]["bybit"]["is_locked"] is False
