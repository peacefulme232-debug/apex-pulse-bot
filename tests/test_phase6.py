"""Phase 6 Tests: Phone PIN Gate, Anti-Spoof Rate Limiter, HTML Telegram Alerts, Cron, Crash Recovery, & Cloud Run Audit."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.database import create_db_engine, init_db
from app.engine.recovery import reconcile_startup_state
from app.main import create_app
from app.models import BotState, SystemLog, Trade
from app.security import SESSION_COOKIE_NAME, reset_rate_limiter
from app.services.notifier import TelegramNotifier

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_phase6_settings_and_limiter(monkeypatch: pytest.MonkeyPatch):
    """Ensure Phase 6 config overrides and rate limiter state never leak to other tests."""
    reset_rate_limiter()
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "")
    monkeypatch.setattr(settings, "SESSION_SECRET", "")
    monkeypatch.setattr(settings, "CRON_SECRET", "")
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", True)
    monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setattr(settings, "TELEGRAM_CHAT_ID", "")
    yield
    reset_rate_limiter()


def test_pin_gate_protects_apis_and_keeps_health_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When DASHBOARD_PIN is set, /health stays 200 while all protected APIs return 401."""
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "246810")
    monkeypatch.setattr(settings, "SESSION_SECRET", "test-stable-secret-key-phase6")
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", False)

    db_url = f"sqlite:///{tmp_path / 'phase6_pin.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        hdata = health.json()
        assert hdata["status"] == "ok"
        assert hdata["auth_enabled"] is True
        assert hdata["background_loop"] is False

        auth_status = client.get("/api/auth/status")
        assert auth_status.status_code == 200
        assert auth_status.json() == {"auth_enabled": True, "authenticated": False}

        # All protected GET & POST routes must reject unauthenticated callers with 401
        assert client.get("/api/dashboard").status_code == 401
        assert client.get("/api/summary").status_code == 401
        assert client.get("/api/brokers/status").status_code == 401
        assert client.get("/api/chart/BTCUSDT").status_code == 401
        assert client.post("/api/strategy/toggle", json={"name": "trend_rider"}).status_code == 401
        assert client.post("/api/kill-switch", json={"reason": "test"}).status_code == 401
        assert client.post("/api/bot/resume").status_code == 401
        assert client.post("/api/backtest/run", json={"symbol": "BTCUSDT"}).status_code == 401
        assert client.post("/api/alerts/test").status_code == 401


def test_per_ip_and_global_anti_spoofing_rate_limiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-IP lockout fires 429 + Retry-After on 6th attempt; global lockout fires on 11th across rotated IPs."""
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "998877")
    monkeypatch.setattr(settings, "SESSION_SECRET", "test-rate-limit-secret")
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", False)

    db_url = f"sqlite:///{tmp_path / 'phase6_ratelimit.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        # 1) Per-IP lockout: 5 failed attempts -> 401, 6th attempt -> 429 with Retry-After
        # Also verify leftmost spoofed X-Forwarded-For hop cannot bypass rightmost IP tracking
        for i in range(5):
            resp = client.post(
                "/api/auth/login",
                json={"pin": "000000"},
                headers={"X-Forwarded-For": f"198.51.100.{i + 1}, 203.0.113.50"},
            )
            assert resp.status_code == 401

        locked = client.post(
            "/api/auth/login",
            json={"pin": "998877"},  # Even valid PIN is blocked during lockout
            headers={"X-Forwarded-For": "198.51.100.99, 203.0.113.50"},
        )
        assert locked.status_code == 429
        assert "Retry-After" in locked.headers
        assert int(locked.headers["Retry-After"]) >= 1

        # 2) Global lockout across 10 rotated distinct IPs
        reset_rate_limiter()
        for i in range(10):
            resp = client.post(
                "/api/auth/login",
                json={"pin": "wrong"},
                headers={"X-Forwarded-For": f"10.0.0.{i + 1}"},
            )
            assert resp.status_code == 401

        global_locked = client.post(
            "/api/auth/login",
            json={"pin": "998877"},
            headers={"X-Forwarded-For": "10.0.0.250"},
        )
        assert global_locked.status_code == 429
        assert "Retry-After" in global_locked.headers
        assert int(global_locked.headers["Retry-After"]) >= 1


def test_pin_login_sets_httponly_cookie_and_logout_revokes_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid PIN login sets HttpOnly cookie, unlocks routes, and logout revokes access back to 401."""
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "135790")
    monkeypatch.setattr(settings, "SESSION_SECRET", "test-login-secret-key")
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", False)

    db_url = f"sqlite:///{tmp_path / 'phase6_auth_flow.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app, base_url="https://testserver") as client:
        login_res = client.post(
            "/api/auth/login",
            json={"pin": "135790"},
            headers={"X-Forwarded-Proto": "https"},
        )
        assert login_res.status_code == 200
        assert login_res.json()["authenticated"] is True

        set_cookie_hdr = login_res.headers.get("set-cookie", "")
        assert f"{SESSION_COOKIE_NAME}=" in set_cookie_hdr
        assert "httponly" in set_cookie_hdr.lower()
        assert "samesite=lax" in set_cookie_hdr.lower()
        assert "secure" in set_cookie_hdr.lower()

        # Authenticated session now unlocks protected endpoints
        assert client.get("/api/auth/status").json() == {"auth_enabled": True, "authenticated": True}
        assert client.get("/api/dashboard").status_code == 200
        assert client.get("/api/summary").status_code == 200
        alert_res = client.post("/api/alerts/test")
        assert alert_res.status_code == 200
        assert alert_res.json()["sent"] is False  # Telegram unconfigured

        # Logout clears cookie and revokes access back to 401
        logout_res = client.post("/api/auth/logout")
        assert logout_res.status_code == 200
        assert client.get("/api/dashboard").status_code == 401
        assert client.get("/api/auth/status").json() == {"auth_enabled": True, "authenticated": False}


def test_cron_endpoints_enforce_secret_and_execute_locked_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cron routes return 503 when CRON_SECRET is blank, 401 on wrong secret, and 200 on valid secret."""
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", False)
    db_url = f"sqlite:///{tmp_path / 'phase6_cron.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        # Background loop must not be spawned when ENABLE_BACKGROUND_LOOP=False
        assert app.state.bot_runner._task is None

        # Blank CRON_SECRET -> 503
        assert client.post("/api/cron/tick").status_code == 503
        assert client.post("/api/cron/daily-summary").status_code == 503

        # Configure CRON_SECRET
        monkeypatch.setattr(settings, "CRON_SECRET", "super-secret-cron-token")

        # Missing or wrong secret -> 401
        assert client.post("/api/cron/tick").status_code == 401
        assert client.post("/api/cron/tick", headers={"X-Cron-Secret": "wrong"}).status_code == 401
        assert client.post("/api/cron/daily-summary", headers={"X-Cron-Secret": "wrong"}).status_code == 401

        # Valid secret -> 200
        tick_res = client.post(
            "/api/cron/tick",
            headers={"X-Cron-Secret": "super-secret-cron-token"},
        )
        assert tick_res.status_code == 200
        tdata = tick_res.json()
        assert tdata["status"] == "ok"
        assert isinstance(tdata["kill_switch"], bool)
        assert isinstance(tdata["duration_ms"], float)
        assert "bybit" in tdata["regimes"] and "forex" in tdata["regimes"]

        summary_res = client.post(
            "/api/cron/daily-summary",
            headers={"X-Cron-Secret": "super-secret-cron-token"},
        )
        assert summary_res.status_code == 200
        sdata = summary_res.json()
        assert sdata["status"] == "ok"
        assert sdata["sent"] is False  # Telegram unconfigured


def test_reconcile_startup_state_preserves_kill_switch_and_audits_trades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup recovery preserves kill_switch_active=True across restarts and audits missing SL / broker errors."""
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", False)
    db_url = f"sqlite:///{tmp_path / 'phase6_recovery.db'}"

    # Boot 1: engage Kill Switch
    app1 = create_app(database_url=db_url)
    with TestClient(app1) as client1:
        ks = client1.post("/api/kill-switch", json={"reason": "Simulated crash test lock"})
        assert ks.status_code == 200
        assert ks.json()["kill_switch_active"] is True

    # Boot 2 (simulated container restart against same DB): Kill Switch MUST stay locked
    app2 = create_app(database_url=db_url)
    with TestClient(app2) as client2:
        health = client2.get("/health")
        assert health.status_code == 200
        assert health.json()["kill_switch"] is True
        report = app2.state.bot_runner.last_recovery_report
        assert report is not None
        assert report["kill_switch_preserved"] is True

    # Direct unit audit: missing SL + failing broker API must NOT crash startup
    engine = create_db_engine(db_url)
    init_db(target_engine=engine)
    from sqlalchemy.orm import sessionmaker

    SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionFactory()
    try:
        db.add(
            Trade(
                market="BYBIT",
                symbol="BTCUSDT",
                side="LONG",
                strategy_name="trend_rider",
                status="OPEN",
                qty=0.1,
                entry_price=50000.0,
                current_price=50100.0,
                stop_loss=0.0,  # Invalid SL
                take_profit=52000.0,
            )
        )
        db.commit()

        failing_broker = MagicMock()
        failing_broker.is_demo_active.return_value = True
        failing_broker.get_broker.return_value.get_open_positions.side_effect = RuntimeError("API timeout")

        audit = reconcile_startup_state(db=db, bybit_broker=failing_broker, forex_broker=None)
        assert audit["kill_switch_preserved"] is True
        assert audit["missing_sl_count"] == 1
        assert audit["level"] == "CRITICAL"

        recovery_logs = (
            db.query(SystemLog).filter(SystemLog.source == "StartupRecovery").all()
        )
        assert len(recovery_logs) >= 2
    finally:
        db.close()


def test_telegram_notifier_html_safety_escaping_truncation_and_noop() -> None:
    """TelegramNotifier no-ops when blank, uses HTML parse_mode, escapes XSS & underscores, and truncates <= 4096."""
    # 1) Unconfigured -> silent no-op
    noop_notifier = TelegramNotifier(bot_token="", chat_id="")
    assert noop_notifier.is_configured() is False
    sent_noop = asyncio.run(
        noop_notifier.notify_trade_opened(
            symbol="EUR_USD",
            side="LONG",
            market="FOREX",
            qty=1000.0,
            entry_price=1.085,
            stop_loss=1.080,
            take_profit=1.095,
            strategy_name="trend_rider",
        )
    )
    assert sent_noop is False

    # 2) Configured -> verify HTML parse_mode, escaping, underscores, and <= 4096 char cap
    captured_payloads: list[dict[str, Any]] = []

    async def _fake_post(self, url: str, json: dict[str, Any], **kwargs: Any):
        captured_payloads.append({"url": url, "json": json})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"ok": true}'
        return mock_resp

    notifier = TelegramNotifier(bot_token="123456:FAKE_TOKEN", chat_id="987654321")
    with patch("httpx.AsyncClient.post", new=_fake_post):
        ok = asyncio.run(
            notifier.notify_trade_opened(
                symbol="EUR_USD",
                side="LONG",
                market="FOREX",
                qty=1000.0,
                entry_price=1.085,
                stop_loss=1.080,
                take_profit=1.095,
                strategy_name="trend_rider",
                reason="<script>alert('xss')</script> on R_100 " + ("A" * 5000),
            )
        )
        assert ok is True

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]["json"]
    assert payload["parse_mode"] == "HTML"
    assert payload["chat_id"] == "987654321"
    assert "EUR_USD" in payload["text"]
    assert "trend_rider" in payload["text"]
    assert "<script>" not in payload["text"]
    assert "&lt;script&gt;" in payload["text"]
    assert len(payload["text"]) <= 4096


def test_structural_and_deployment_audit() -> None:
    """Audit Dockerfile, database.py dialect gating, and index.html tabs + PIN overlay."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "exec uvicorn" in dockerfile
    assert "--workers 1" in dockerfile
    assert "--proxy-headers" in dockerfile
    assert "USER 10001" in dockerfile

    db_src = (REPO_ROOT / "app" / "database.py").read_text(encoding="utf-8")
    assert 'db_engine.dialect.name == "sqlite"' in db_src
    assert 'target_engine.dialect.name != "sqlite"' in db_src
    assert '"pool_pre_ping": True' in db_src
    assert '"pool_recycle"] = 300' in db_src

    html_src = (REPO_ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "Live Terminal" in html_src
    assert "Bot Summarizer" in html_src
    assert "Strategy Lab" in html_src
    assert 'id="pin-lock-overlay"' in html_src
    assert 'inputmode="numeric"' in html_src
    assert "/api/auth/status" in html_src
    assert "/api/auth/login" in html_src
    assert "/api/auth/logout" in html_src

    deploy_script = (REPO_ROOT / "deploy" / "deploy_cloud_run.sh").read_text(encoding="utf-8")
    assert "africa-south1" in deploy_script
    assert "ENABLE_BACKGROUND_LOOP=false" in deploy_script
    assert "--max-instances=1" in deploy_script
