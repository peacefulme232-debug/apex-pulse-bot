"""Prompt 10 E2E hardening & Frontend UI Sync verification tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.database import create_db_engine, init_db
from app.main import create_app
from app.models import ExecutionRejection, Trade

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_synthetic_h1_df(bars: int = 360) -> pd.DataFrame:
    """Build deterministic H1 OHLCV DataFrame for offline walk-forward API smoke tests."""
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(bars):
        # Gentle oscillating trend to produce deterministic candles
        drift = 0.25 if (i // 40) % 2 == 0 else -0.15
        o = price
        c = max(10.0, o + drift)
        h = max(o, c) + 0.6
        l = max(5.0, min(o, c) - 0.6)
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1000.0 + (i % 10) * 50.0,
            }
        )
        price = c
    return pd.DataFrame(rows)


def test_dockerfile_shell_form_port_expansion() -> None:
    """Verify Dockerfile uses /bin/sh -c shell form so Railway expands ${PORT:-8080}."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert '"/bin/sh", "-c"' in dockerfile
    assert "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1" in dockerfile


def test_index_html_strategy_brain_and_walk_forward_ui_sync(tmp_path: Path) -> None:
    """Verify GET / renders all Prompt 10 UI components and zero broken endpoints."""
    db_url = f"sqlite:///{tmp_path / 'ui_sync.db'}"
    app = create_app(database_url=db_url)
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        html = res.text

        # 1. Active Regime Card + H4 bias + active strategy pill
        assert 'id="kpi-regime"' in html
        assert 'id="kpi-h4-bias"' in html
        assert 'id="kpi-active-strategy"' in html

        # 2. Confluence Radar ("Stockfish Eval") + 5 pillar bars + badges + circuit breaker Unlock
        assert "Stockfish Eval" in html
        assert 'id="conf-gauge-bar"' in html
        assert 'id="conf-gauge-threshold"' in html
        assert 'id="conf-bar-htf"' in html
        assert 'id="conf-bar-level"' in html
        assert 'id="conf-bar-mom"' in html
        assert 'id="conf-bar-vol"' in html
        assert 'id="conf-bar-rr"' in html
        assert 'id="conf-armed-badge"' in html
        assert 'id="conf-div-badge"' in html
        assert 'id="conf-veto-pill"' in html
        assert 'id="btn-cb-unlock"' in html

        # 3. Strategy Control Board wired to PATCH /api/brain/config
        assert "/api/brain/telemetry" in html
        assert "/api/brain/config" in html
        assert 'id="toggle-brain-ema"' in html
        assert 'id="toggle-brain-breakout"' in html
        assert 'id="toggle-brain-rsi"' in html
        assert 'id="brain-min-conf"' in html
        assert 'id="brain-max-fee-drag"' in html
        assert 'id="brain-be-r"' in html
        assert 'id="brain-micro-shield"' in html

        # 4. Strategy Lab — Walk-Forward Validation (Real Fees)
        assert "Walk-Forward Validation (Real Fees)" in html
        assert "/api/strategy-lab/walk-forward" in html
        assert 'id="wf-symbol"' in html
        assert 'id="wf-capital"' in html
        assert 'id="wf-folds"' in html
        assert 'id="wf-realism"' in html
        assert 'id="btn-run-wf"' in html
        assert 'id="wf-verdict-banner"' in html
        assert "PASS — READY FOR $20 PAPER TEST" in html
        assert "FAIL — DO NOT TRADE LIVE" in html
        assert 'id="wf-veto-pills"' in html
        assert 'id="wf-folds-tbody"' in html

        # 5. Bot Summarizer Strategy Brain Performance + Wallet Skipped Signals Score column
        assert 'id="sum-brain-perf-tbody"' in html
        assert "EMA 50 Pullback" in html
        assert "Breakout + Retest" in html
        assert "RSI Divergence Sniper" in html
        assert 'id="wallet-rej-tbody"' in html
        assert "OPPOSING_DIVERGENCE_VETO" in html


def test_e2e_smoke_endpoints_and_summarizer_brain_metrics(tmp_path: Path) -> None:
    """Smoke-test /, /api/wallet/summary, /api/brain/telemetry, /api/brain/config, /api/summary, /api/wallet/rejections, and /api/strategy-lab/walk-forward."""
    db_url = f"sqlite:///{tmp_path / 'e2e_smoke.db'}"
    engine = create_db_engine(db_url)
    init_db(target_engine=engine)
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        # 1. Wallet summary
        w_res = client.get("/api/wallet/summary")
        assert w_res.status_code == 200
        w_data = w_res.json()
        assert "combined" in w_data
        assert "markets" in w_data

        # 2. Brain telemetry & config PATCH
        t_res = client.get("/api/brain/telemetry")
        assert t_res.status_code == 200
        t_data = t_res.json()
        assert t_data["config"]["min_confluence_score"] == 75
        assert "circuit_breaker" in t_data

        p_res = client.patch(
            "/api/brain/config",
            json={
                "ema_50_pullback_enabled": False,
                "min_confluence_score": 80,
                "max_fee_drag_pct": 12.5,
                "breakeven_r_trigger": 1.2,
                "micro_shield_enabled": True,
                "reset_circuit_breaker": True,
            },
        )
        assert p_res.status_code == 200
        p_data = p_res.json()
        assert p_data["config"]["ema_50_pullback_enabled"] is False
        assert p_data["config"]["min_confluence_score"] == 80
        assert p_data["config"]["max_fee_drag_pct"] == 12.5
        assert p_data["config"]["breakeven_r_trigger"] == 1.2

        # 3. Seed closed Strategy Brain trades + exchange/veto rejections and verify /api/summary + /api/wallet/rejections
        session_factory = app.state.broker._session_factory
        db = session_factory()
        try:
            now = datetime.now(timezone.utc)
            db.add(
                Trade(
                    market="BYBIT",
                    symbol="SOLUSDT",
                    side="LONG",
                    strategy_name="EMA_50_PULLBACK",
                    status="CLOSED",
                    qty=1.0,
                    entry_price=150.0,
                    current_price=156.0,
                    exit_price=156.0,
                    stop_loss=147.0,
                    take_profit=156.0,
                    pnl_usd=5.50,
                    pnl_pct=3.67,
                    filled_entry_price=Decimal("150.0"),
                    filled_exit_price=Decimal("156.0"),
                    gross_pnl=Decimal("6.000000"),
                    entry_fee=Decimal("0.150000"),
                    exit_fee=Decimal("0.150000"),
                    spread_cost=Decimal("0.100000"),
                    slippage_cost=Decimal("0.100000"),
                    funding_or_swap_fee=Decimal("0.000000"),
                    net_pnl=Decimal("5.500000"),
                    opened_at=now - timedelta(hours=2),
                    closed_at=now - timedelta(hours=1),
                )
            )
            db.add(
                ExecutionRejection(
                    market="bybit",
                    symbol="BTCUSDT",
                    side="LONG",
                    strategy=None,
                    confluence_score=None,
                    reason_code="BELOW_MIN_NOTIONAL",
                    required_margin=Decimal("5.00"),
                    available_free_margin=Decimal("20.00"),
                    details="Order below Bybit $5 min notional",
                )
            )
            db.add(
                ExecutionRejection(
                    market="bybit",
                    symbol="SOLUSDT",
                    side="LONG",
                    strategy="EMA_50_PULLBACK",
                    confluence_score=Decimal("62.00"),
                    reason_code="FEE_DRAG_TOO_HIGH",
                    details="Fee drag 18.4% > 15% cap",
                )
            )
            db.commit()
        finally:
            db.close()

        s_res = client.get("/api/summary?market=ALL")
        assert s_res.status_code == 200
        s_data = s_res.json()
        brain_perf = {row["name"]: row for row in s_data["brain_strategy_performance"]}
        assert set(brain_perf.keys()) == {"EMA_50_PULLBACK", "BREAKOUT_RETEST", "RSI_DIVERGENCE_SNIPER"}
        ema_row = brain_perf["EMA_50_PULLBACK"]
        assert ema_row["total_trades"] == 1
        assert ema_row["win_rate_pct"] == 100.0
        assert ema_row["gross_pnl_usd"] == 6.0
        assert ema_row["fees_paid_usd"] == 0.5
        assert ema_row["net_pnl_usd"] == 5.5

        r_res = client.get("/api/wallet/rejections?limit=50")
        assert r_res.status_code == 200
        r_items = r_res.json()["items"]
        assert any(item["reason_code"] == "BELOW_MIN_NOTIONAL" for item in r_items)
        assert any(
            item["reason_code"] == "FEE_DRAG_TOO_HIGH"
            and item["confluence_score"] == 62.0
            and item["strategy"] == "EMA_50_PULLBACK"
            for item in r_items
        )

        # 4. Walk-forward validation endpoint smoke test (mocked offline data)
        fake_df = _build_synthetic_h1_df(360)
        with patch("app.main.HistoricalDataFetcher.fetch", return_value=(fake_df, "synthetic_test")):
            wf_res = client.post(
                "/api/strategy-lab/walk-forward",
                json={
                    "symbol": "SOLUSDT",
                    "starting_capital": "20.00",
                    "strategies": ["EMA_50_PULLBACK", "BREAKOUT_RETEST", "RSI_DIVERGENCE_SNIPER"],
                    "folds": 3,
                    "realism_enabled": True,
                },
            )
        assert wf_res.status_code == 200
        wf_data = wf_res.json()
        assert wf_data["verdict"] in {"PASS_READY_FOR_PAPER", "FAIL_DO_NOT_TRADE_LIVE"}
        assert wf_data["folds_count"] == 3
        assert len(wf_data["folds"]) == 3
        for key in (
            "gross_pnl",
            "total_fees_paid",
            "net_pnl",
            "profit_factor",
            "win_rate",
            "max_drawdown_pct",
            "signals_vetoed_count",
            "veto_breakdown",
        ):
            assert key in wf_data
