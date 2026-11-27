"""Tests for execution_rejections NOT NULL fix, 6-strategy Regime Router, and MT5 Close / Close All."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.database import create_db_engine, init_db
from app.engine.runner import BotRunner
from app.main import create_app
from app.models import ExecutionRejection, Trade
from app.services.confluence_evaluator import (
    BrainConfig,
    ConfluenceBreakdown,
    ConfluenceEvaluation,
)
from app.services.regime_router import (
    RegimeDecision,
    RegimeState,
    get_unified_regime_strategies,
)
from app.services.strategies.base import CandidateSignal


def test_execution_rejections_legacy_not_null_migration_and_zero_defaults(tmp_path: Path) -> None:
    """Legacy SQLite DB with NOT NULL required_min_qty migrates cleanly and logs rejected signals without IntegrityError."""
    db_url = f"sqlite:///{tmp_path / 'legacy_rejections.db'}"
    engine = create_db_engine(db_url)

    # Create a legacy execution_rejections table with strict NOT NULL and NO defaults
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE execution_rejections (
                    id INTEGER NOT NULL PRIMARY KEY,
                    market VARCHAR(16) NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(16) NOT NULL,
                    reason_code VARCHAR(48) NOT NULL,
                    requested_qty NUMERIC(18, 8) NOT NULL,
                    required_min_qty NUMERIC(18, 8) NOT NULL,
                    required_margin NUMERIC(18, 6) NOT NULL,
                    available_free_margin NUMERIC(18, 6) NOT NULL,
                    details TEXT NOT NULL,
                    created_at DATETIME NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO execution_rejections (
                    market, symbol, side, reason_code, requested_qty,
                    required_min_qty, required_margin, available_free_margin, details, created_at
                ) VALUES (
                    'bybit', 'BTCUSDT', 'LONG', 'BELOW_MIN_NOTIONAL', 0.0001,
                    0.001, 5.0, 20.0, 'Legacy row preserved', '2025-01-01 00:00:00'
                )
                """
            )
        )

    # Run init_db twice (idempotent migration)
    init_db(target_engine=engine)
    init_db(target_engine=engine)

    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionCls()
    try:
        # Insert rejected BREAKOUT_RETEST signal WITHOUT specifying required_min_qty or required_margin
        rej = ExecutionRejection(
            market="bybit",
            symbol="SOLUSDT",
            side="LONG",
            strategy="BREAKOUT_RETEST",
            confluence_score=Decimal("64.00"),
            reason_code="LOW_CONFLUENCE_SCORE",
            details="Confluence score 64/100 < minimum 75.",
        )
        db.add(rej)
        db.commit()
        db.refresh(rej)

        assert rej.id is not None
        assert Decimal(str(rej.required_min_qty)) == Decimal("0")
        assert Decimal(str(rej.required_margin)) == Decimal("0")
        assert Decimal(str(rej.requested_qty)) == Decimal("0")
        assert Decimal(str(rej.available_free_margin)) == Decimal("0")

        # Verify legacy row was preserved
        total_rows = db.query(ExecutionRejection).count()
        assert total_rows == 2
    finally:
        db.close()


def test_unified_six_strategy_regime_router_and_highest_confluence_selection(tmp_path: Path) -> None:
    """Verify 6-strategy regime mapping, highest-confluence auto-selection, and 6-toggle PATCH /api/brain/config."""
    assert get_unified_regime_strategies(RegimeState.TRENDING_BULL) == ["EMA_50_PULLBACK", "TREND_RIDER"]
    assert get_unified_regime_strategies(RegimeState.TRENDING_BEAR) == ["EMA_50_PULLBACK", "TREND_RIDER"]
    assert get_unified_regime_strategies(RegimeState.BREAKOUT_EXPANSION) == ["BREAKOUT_RETEST", "VOLATILITY_BREAKOUT"]
    assert get_unified_regime_strategies(RegimeState.REVERSAL_EXHAUSTION) == ["RSI_DIVERGENCE_SNIPER", "MEAN_REVERSION"]
    assert get_unified_regime_strategies(RegimeState.DEAD_CHOP) == []

    # Test auto-selection chooses the single highest-confluence approved candidate (no conflicting entries)
    db_url = f"sqlite:///{tmp_path / 'six_strat.db'}"
    engine = create_db_engine(db_url)
    init_db(target_engine=engine)
    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    runner = BotRunner(session_factory=SessionCls, brain_config=BrainConfig())

    cand_ema = CandidateSignal(
        symbol="BTCUSDT",
        market="bybit",
        strategy="EMA_50_PULLBACK",
        side="LONG",
        entry_price=95000.0,
        sl_price=94000.0,
        tp_price=97500.0,
    )
    cand_trend = CandidateSignal(
        symbol="BTCUSDT",
        market="bybit",
        strategy="TREND_RIDER",
        side="LONG",
        entry_price=95000.0,
        sl_price=94100.0,
        tp_price=98000.0,
    )

    def _fake_eval(candidate, **_kwargs):
        if candidate.strategy == "TREND_RIDER":
            return ConfluenceEvaluation(
                approved=True,
                total_score=88,
                breakdown=ConfluenceBreakdown(25, 22, 16, 12, 13),
                vetoes=[],
                primary_veto=None,
                details="Approved 88/100",
                fee_drag_pct_of_tp=3.5,
                net_rr=2.9,
                estimated_total_friction_tp=1.0,
            )
        return ConfluenceEvaluation(
            approved=True,
            total_score=78,
            breakdown=ConfluenceBreakdown(20, 20, 14, 12, 12),
            vetoes=[],
            primary_veto=None,
            details="Approved 78/100",
            fee_drag_pct_of_tp=4.0,
            net_rr=2.3,
            estimated_total_friction_tp=1.1,
        )

    runner.confluence_evaluator.evaluate = MagicMock(side_effect=_fake_eval)
    decision = RegimeDecision(
        symbol="BTCUSDT",
        state=RegimeState.TRENDING_BULL,
        allowed_strategies=["EMA_50_PULLBACK"],
        bias="LONG",
        veto_longs=False,
        veto_shorts=False,
        reason="Bullish trend",
    )

    db = SessionCls()
    try:
        winner, winner_eval = runner.select_best_strategy_candidate(
            candidates=[cand_ema, cand_trend],
            db=db,
            h1_candles=None,
            h4_candles=None,
            d1_candles=None,
            regime_decision=decision,
            active_divergence=None,
        )
        assert winner is not None
        assert winner_eval is not None
        assert winner.strategy == "TREND_RIDER"
        assert winner_eval.total_score == 88
    finally:
        db.close()

    # Verify all 6 toggles via PATCH /api/brain/config and HTML UI
    app = create_app(database_url=db_url)
    with TestClient(app) as client:
        patch_res = client.patch(
            "/api/brain/config",
            json={
                "ema_50_pullback_enabled": True,
                "trend_rider_enabled": False,
                "breakout_retest_enabled": True,
                "volatility_breakout_enabled": False,
                "rsi_divergence_sniper_enabled": True,
                "mean_reversion_enabled": False,
            },
        )
        assert patch_res.status_code == 200
        cfg = patch_res.json()["config"]
        assert cfg["trend_rider_enabled"] is False
        assert cfg["volatility_breakout_enabled"] is False
        assert cfg["mean_reversion_enabled"] is False
        assert cfg["ema_50_pullback_enabled"] is True

        html = client.get("/").text
        assert 'id="toggle-brain-trend-rider"' in html
        assert 'id="toggle-brain-vol-breakout"' in html
        assert 'id="toggle-brain-mean-rev"' in html
        assert 'id="btn-close-all"' in html
        assert "/api/positions/close-all" in html


def test_mt5_single_close_and_close_all_endpoints_update_wallet_and_history(tmp_path: Path) -> None:
    """POST /api/positions/{id}/close and /api/positions/close-all close trades with Realism fees and update wallet balances."""
    db_url = f"sqlite:///{tmp_path / 'mt5_close.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        # Ensure DB is seeded, then inspect open positions
        dash_before = client.get("/api/dashboard?market=ALL").json()
        open_before = dash_before["open_positions"]
        assert len(open_before) >= 2  # Seeded BTCUSDT (BYBIT) and EUR_USD (FOREX)

        wallet_before = client.get("/api/wallet/summary").json()
        bybit_bal_before = float(wallet_before["markets"]["bybit"]["active_balance"])

        # Pick first BYBIT open position and close it individually
        bybit_pos = next(p for p in open_before if p["market"] == "BYBIT")
        close_res = client.post(f"/api/positions/{bybit_pos['id']}/close")
        assert close_res.status_code == 200
        close_body = close_res.json()
        closed_trade = close_body["closed_trade"]
        assert closed_trade["status"] == "CLOSED"
        assert closed_trade["exit_price"] is not None
        assert float(closed_trade["exit_fee"]) > 0.0
        assert float(closed_trade["spread_cost"]) >= 0.0

        # Double-closing the same trade returns 409 Conflict
        dup_res = client.post(f"/api/positions/{bybit_pos['id']}/close")
        assert dup_res.status_code == 409

        # Unknown position ID returns 404
        missing_res = client.post("/api/positions/999999/close")
        assert missing_res.status_code == 404

        # Wallet active_balance updated by realized net_pnl
        wallet_after_single = close_body["wallet"]
        bybit_bal_after = float(wallet_after_single["markets"]["bybit"]["active_balance"])
        expected_bal = round(bybit_bal_before + float(closed_trade["net_pnl"]), 2)
        assert abs(bybit_bal_after - expected_bal) <= 0.05

        # Now seed an extra BYBIT open trade so we can test market-filtered Close All
        session_factory = app.state.broker._session_factory
        db = session_factory()
        try:
            db.add(
                Trade(
                    market="BYBIT",
                    symbol="SOLUSDT",
                    side="LONG",
                    strategy_name="TREND_RIDER",
                    status="OPEN",
                    qty=1.0,
                    entry_price=150.0,
                    current_price=155.0,
                    stop_loss=145.0,
                    take_profit=165.0,
                    pnl_usd=5.0,
                    pnl_pct=3.33,
                    entry_reason="Test open position",
                    filled_entry_price=Decimal("150.0"),
                    initial_margin=Decimal("30.0"),
                    maintenance_margin=Decimal("1.5"),
                    opened_at=datetime.now(timezone.utc),
                )
            )
            db.commit()
        finally:
            db.close()

        # Close All BYBIT only -> FOREX stays open
        close_bybit_res = client.post("/api/positions/close-all", json={"market": "bybit"})
        assert close_bybit_res.status_code == 200
        assert close_bybit_res.json()["closed_count"] >= 1

        dash_mid = client.get("/api/dashboard?market=ALL").json()
        assert all(p["market"] != "BYBIT" for p in dash_mid["open_positions"])
        assert any(p["market"] == "FOREX" for p in dash_mid["open_positions"])

        # Close All remaining (ALL) -> zero open positions, all recorded in Trade History Feed
        close_all_res = client.post("/api/positions/close-all", json={"market": "ALL"})
        assert close_all_res.status_code == 200
        assert close_all_res.json()["closed_count"] >= 1

        dash_final = client.get("/api/dashboard?market=ALL").json()
        assert len(dash_final["open_positions"]) == 0
        closed_ids = {t["id"] for t in dash_final["closed_trades"]}
        assert bybit_pos["id"] in closed_ids
