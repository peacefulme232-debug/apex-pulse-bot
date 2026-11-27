"""Tests for zero-lookahead Walk-Forward Out-of-Sample Validator and protected API endpoint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app
from app.services.indicators import Candle
from app.services.strategies import CandidateSignal
from app.services.walk_forward import WalkForwardEngine, compute_rolling_folds


def _make_trending_and_choppy_series(n: int = 420) -> list[Candle]:
    """Build a deterministic synthetic OHLCV series mixing uptrend, chop, and pullback waves."""
    candles: list[Candle] = []
    start_dt = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    price = 100.0

    for i in range(n):
        ts = start_dt + timedelta(hours=i)
        if 150 <= i < 210:
            # Sideways choppy regime (DEAD_CHOP)
            delta = 0.08 if i % 2 == 0 else -0.08
            o = price
            c = price + delta
            h = max(o, c) + 0.15
            l = min(o, c) - 0.15
        else:
            # Trending uptrend with periodic pullbacks
            wave = 0.45 * math.sin(i * 0.25)
            delta = 0.22 + wave
            o = price
            c = max(10.0, price + delta)
            h = max(o, c) + 0.55
            l = min(o, c) - 0.55
        candles.append(
            Candle(
                open=o,
                high=h,
                low=l,
                close=c,
                volume=120.0 + (i % 15) * 10.0,
                timestamp=ts,
                is_closed=True,
            )
        )
        price = c

    return candles


def test_walk_forward_synthetic_trending_choppy_20_usd_and_sl_before_tp_tie_break() -> None:
    """Walk-forward on $20 capital completes cleanly, populates all metrics, and breaks SL/TP ties conservatively."""
    candles = _make_trending_and_choppy_series(420)
    folds = compute_rolling_folds(len(candles), folds=3)
    assert len(folds) == 3

    # Inject a deterministic bar inside fold 1 OOS whose [low, high] spans BOTH SL and TP
    oos_bar_idx = folds[0][2] + 2
    entry_bar = candles[oos_bar_idx]
    entry_px = entry_bar.close
    sl_px = entry_px - 2.0
    tp_px = entry_px + 5.0

    # Next bar simultaneously breaches BOTH SL (low < sl_px) and TP (high > tp_px)
    tie_idx = oos_bar_idx + 1
    tie_orig = candles[tie_idx]
    candles[tie_idx] = Candle(
        open=entry_px,
        high=tp_px + 2.0,
        low=sl_px - 2.0,
        close=tp_px + 1.0,  # Closes high, yet conservative tie-break MUST assume SL hit first!
        volume=200.0,
        timestamp=tie_orig.timestamp,
        is_closed=True,
    )

    forced = {
        oos_bar_idx: CandidateSignal(
            symbol="SOLUSDT",
            market="bybit",
            strategy="EMA_50_PULLBACK",
            side="LONG",
            entry_price=entry_px,
            sl_price=sl_px,
            tp_price=tp_px,
            order_entry_type="MARKET",
            order_exit_type="TAKE_PROFIT",
            reason="Tie-break test signal",
            metadata={"pattern": "bullish_rejection"},
        )
    }

    engine = WalkForwardEngine()
    result = engine.run(
        candles_h1=candles,
        symbol="SOLUSDT",
        market="bybit",
        starting_capital=Decimal("20.00"),
        folds=3,
        leverage=5,
        profile="bybit_vip0_linear",
        realism_enabled=True,
        min_confluence_score=65,
        allow_min_lot_override=True,
        forced_signals=forced,
    )

    payload = result.to_dict()
    assert payload["starting_capital"] == pytest.approx(20.0)
    assert payload["folds_count"] == 3
    assert len(payload["folds"]) == 3
    assert len(payload["equity_curve"]) > 0
    assert payload["verdict"] in ("PASS_READY_FOR_PAPER", "FAIL_DO_NOT_TRADE_LIVE")
    if payload["verdict"] == "FAIL_DO_NOT_TRADE_LIVE":
        assert len(payload["failing_criteria"]) >= 1

    # Verify all required metrics exist
    for key in (
        "starting_capital",
        "ending_capital",
        "gross_pnl",
        "total_fees_paid",
        "net_pnl",
        "win_rate",
        "profit_factor",
        "expectancy_per_trade",
        "max_drawdown_pct",
        "sharpe_ratio",
        "trades_executed",
        "signals_vetoed_count",
        "veto_breakdown",
        "equity_curve",
    ):
        assert key in payload

    # Fold 1 executed the tie-break trade and MUST have closed at SL (net_pnl < 0), NOT TP!
    fold1 = result.folds[0]
    assert fold1.trades_executed >= 1
    assert fold1.net_pnl < 0.0


def test_walk_forward_realism_true_vs_false_proves_fee_impact() -> None:
    """Running with realism_enabled=True vs False proves total_fees_paid > 0 and net_pnl < gross_pnl."""
    candles = _make_trending_and_choppy_series(360)
    folds = compute_rolling_folds(len(candles), folds=3)

    # Place a clean winning trade in each of the 3 folds
    forced: dict[int, CandidateSignal] = {}
    for _, _, oos_start, oos_end in folds:
        entry_idx = oos_start + 1
        tp_idx = min(entry_idx + 2, oos_end - 1)
        entry_px = candles[entry_idx].close
        sl_px = entry_px - 2.0
        tp_px = entry_px + 5.0

        # Ensure bar at entry_idx + 1 does not hit SL, and bar at tp_idx cleanly hits TP
        mid_bar = candles[entry_idx + 1]
        candles[entry_idx + 1] = Candle(
            open=entry_px,
            high=entry_px + 1.0,
            low=entry_px - 0.5,
            close=entry_px + 0.8,
            volume=150.0,
            timestamp=mid_bar.timestamp,
            is_closed=True,
        )
        tp_bar = candles[tp_idx]
        candles[tp_idx] = Candle(
            open=entry_px + 0.8,
            high=tp_px + 1.0,
            low=entry_px + 0.5,
            close=tp_px + 0.5,
            volume=180.0,
            timestamp=tp_bar.timestamp,
            is_closed=True,
        )

        forced[entry_idx] = CandidateSignal(
            symbol="SOLUSDT",
            market="bybit",
            strategy="EMA_50_PULLBACK",
            side="LONG",
            entry_price=entry_px,
            sl_price=sl_px,
            tp_price=tp_px,
            order_entry_type="MARKET",
            order_exit_type="TAKE_PROFIT",
            reason="Clean TP trade",
            metadata={"pattern": "bullish_rejection"},
        )

    engine = WalkForwardEngine()
    res_real = engine.run(
        candles_h1=candles,
        symbol="SOLUSDT",
        market="bybit",
        starting_capital=Decimal("100.00"),
        folds=3,
        leverage=5,
        realism_enabled=True,
        min_confluence_score=65,
        forced_signals=forced,
    )
    res_zero_fee = engine.run(
        candles_h1=candles,
        symbol="SOLUSDT",
        market="bybit",
        starting_capital=Decimal("100.00"),
        folds=3,
        leverage=5,
        realism_enabled=False,
        min_confluence_score=65,
        forced_signals=forced,
    )

    assert res_real.trades_executed >= 3
    assert res_real.total_fees_paid > 0.0
    assert res_real.net_pnl < res_real.gross_pnl

    assert res_zero_fee.trades_executed >= 3
    assert res_zero_fee.total_fees_paid == pytest.approx(0.0)
    assert res_zero_fee.net_pnl == pytest.approx(res_zero_fee.gross_pnl)


def test_walk_forward_endpoint_requires_pin_auth_when_pin_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unauthenticated POST /api/strategy-lab/walk-forward returns 401 when DASHBOARD_PIN is set."""
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "987654")
    db_file = tmp_path / "wf_auth.db"
    app = create_app(database_url=f"sqlite:///{db_file}")

    with TestClient(app) as client:
        resp = client.post(
            "/api/strategy-lab/walk-forward",
            json={
                "symbol": "SOLUSDT",
                "market": "bybit",
                "starting_capital": "20.00",
                "folds": 3,
                "realism_enabled": True,
            },
        )
        assert resp.status_code == 401
