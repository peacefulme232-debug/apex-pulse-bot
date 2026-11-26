"""Automated Phase 1 verification suite for APEX-PULSE."""

from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    """Create an isolated TestClient backed by a temporary SQLite database."""
    db_file = tmp_path / "test_apex_pulse.db"
    test_app = create_app(database_url=f"sqlite:///{db_file}")
    with TestClient(test_app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """GET /health returns HTTP 200 and mode == 'PAPER'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["mode"] == "PAPER"
    assert data["kill_switch"] is False


def test_dashboard_html_renders(client: TestClient) -> None:
    """GET / returns HTTP 200 and renders APEX-PULSE dashboard HTML cleanly."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "APEX-PULSE" in resp.text
    assert "PAPER / DEMO MODE" in resp.text
    assert "KILL SWITCH" in resp.text


@pytest.mark.parametrize("market", ["ALL", "BYBIT", "FOREX"])
def test_dashboard_api_market_filters(client: TestClient, market: str) -> None:
    """GET /api/dashboard works for ALL, BYBIT, and FOREX with valid KPIs and 4 strategies."""
    resp = client.get(f"/api/dashboard?market={market}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["market"] == market
    assert data["trading_mode"] == "PAPER"
    assert len(data["strategies"]) == 4

    kpis = data["kpis"]
    assert isinstance(kpis["total_equity"], (int, float))
    assert isinstance(kpis["bybit_equity"], (int, float))
    assert isinstance(kpis["forex_equity"], (int, float))
    assert isinstance(kpis["daily_pnl_usd"], (int, float))
    assert isinstance(kpis["daily_pnl_pct"], (int, float))
    assert isinstance(kpis["win_rate"], (int, float))
    assert kpis["max_daily_loss_pct"] == 3.0
    assert len(data["equity_curve"]) > 0


def test_strategy_toggle_persists(client: TestClient) -> None:
    """POST /api/strategy/toggle persists toggle state across requests."""
    initial = client.get("/api/dashboard").json()["strategies"]
    target = next(s for s in initial if s["name"] == "trend_rider")
    original_enabled = target["enabled"]

    toggle_resp = client.post("/api/strategy/toggle", json={"name": "trend_rider"})
    assert toggle_resp.status_code == 200
    assert toggle_resp.json()["enabled"] is (not original_enabled)

    after_toggle = client.get("/api/dashboard").json()["strategies"]
    updated = next(s for s in after_toggle if s["name"] == "trend_rider")
    assert updated["enabled"] is (not original_enabled)

    # Toggle back and verify persistence
    revert_resp = client.post("/api/strategy/toggle", json={"name": "trend_rider"})
    assert revert_resp.status_code == 200
    assert revert_resp.json()["enabled"] is original_enabled


def test_kill_switch_and_resume(client: TestClient) -> None:
    """POST /api/kill-switch locks bot and closes open positions; /api/bot/resume unlocks."""
    before = client.get("/api/dashboard").json()
    assert len(before["open_positions"]) > 0
    assert before["bot_state"]["kill_switch_active"] is False

    kill_resp = client.post("/api/kill-switch", json={"reason": "Pytest emergency halt"})
    assert kill_resp.status_code == 200
    kill_data = kill_resp.json()
    assert kill_data["kill_switch_active"] is True
    assert kill_data["is_running"] is False
    assert kill_data["closed_positions"] >= 1

    health_locked = client.get("/health").json()
    assert health_locked["kill_switch"] is True

    after_kill = client.get("/api/dashboard").json()
    assert after_kill["bot_state"]["kill_switch_active"] is True
    assert len(after_kill["open_positions"]) == 0
    assert any(log["level"] == "CRITICAL" for log in after_kill["system_logs"])

    resume_resp = client.post("/api/bot/resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["kill_switch_active"] is False
    assert resume_resp.json()["is_running"] is True

    after_resume = client.get("/api/dashboard").json()
    assert after_resume["bot_state"]["kill_switch_active"] is False
    assert after_resume["bot_state"]["is_running"] is True
