"""Phase 4 deterministic unit & API tests (100% mocked, zero live network calls)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import httpx
import pytest

from app.brokers.forex_deriv import (
    DERIV_LEGACY_WS_TEMPLATE,
    DERIV_PUBLIC_WS_URL,
    DerivBrokerError,
    DerivForexBroker,
)
from app.brokers.forex_oanda import OANDA_PRACTICE_URL, OandaBrokerError, OandaForexBroker
from app.brokers.forex_router import ForexRouter
from app.brokers.paper_sim import PaperSimBroker
from app.config import Settings
from app.engine.session_guard import SessionGuard
from app.engine.summarizer import BotSummarizer
from app.main import create_app


class _FakeWS:
    """Deterministic in-memory WebSocket mock for DerivForexBroker tests."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.sent: list[dict[str, Any]] = []

    def __enter__(self) -> "_FakeWS":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def recv(self) -> str:
        return json.dumps(self._responses.pop(0))


def test_phase4_config_safety_locks() -> None:
    """Blank Forex keys force FOREX_PROVIDER=paper, and live OANDA URL is rejected."""
    s_oanda_blank = Settings(FOREX_PROVIDER="oanda", OANDA_API_KEY="", OANDA_ACCOUNT_ID="")
    assert s_oanda_blank.FOREX_PROVIDER == "paper"

    s_deriv_blank = Settings(FOREX_PROVIDER="deriv", DERIV_API_TOKEN="")
    assert s_deriv_blank.FOREX_PROVIDER == "paper"

    with pytest.raises(ValueError):
        Settings(OANDA_BASE_URL="https://api-fxtrade.oanda.com/v3")

    s_ok = Settings(
        FOREX_PROVIDER="oanda",
        OANDA_API_KEY="test-key",
        OANDA_ACCOUNT_ID="101-001-123456-001",
        OANDA_BASE_URL=OANDA_PRACTICE_URL,
    )
    assert s_ok.FOREX_PROVIDER == "oanda"


def test_session_guard_weekend_rollover_killzone_news_and_synthetic_bypass() -> None:
    """SessionGuard enforces weekend, rollover, kill-zone, news blackout, and Deriv 24/7 bypass."""
    cpi_event = {
        "name": "US CPI YoY",
        "timestamp": "2026-06-10T12:30:00Z",
        "currency": "USD",
    }
    guard = SessionGuard(
        killzone_only=True,
        news_blackout_minutes=30,
        news_events=[cpi_event],
    )

    # 1. Weekend close (Saturday 12:00 UTC)
    sat = guard.evaluate("EUR_USD", datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc))
    assert sat.can_trade is False
    assert sat.is_weekend_closed is True
    assert sat.reason == "Forex weekend market close"

    # 2. Friday 21:50 UTC is also weekend close
    fri_late = guard.evaluate("EUR_USD", datetime(2026, 6, 5, 21, 50, tzinfo=timezone.utc))
    assert fri_late.can_trade is False
    assert fri_late.is_weekend_closed is True

    # 3. Daily rollover spread spike pause (Tuesday 22:00 UTC)
    roll = guard.evaluate("EUR_USD", datetime(2026, 6, 9, 22, 0, tzinfo=timezone.utc))
    assert roll.can_trade is False
    assert roll.is_rollover_pause is True
    assert roll.reason == "Daily rollover spread spike protection"

    # 4. London/NY Overlap peak liquidity (Tuesday 14:00 UTC)
    overlap = guard.evaluate("EUR_USD", datetime(2026, 6, 9, 14, 0, tzinfo=timezone.utc))
    assert overlap.can_trade is True
    assert overlap.session_name == "LONDON_NY_OVERLAP"
    assert overlap.is_news_blackout is False

    # 5. Asian low-liquidity pause when killzone_only=True (Tuesday 03:00 UTC)
    asian = guard.evaluate("EUR_USD", datetime(2026, 6, 9, 3, 0, tzinfo=timezone.utc))
    assert asian.can_trade is False
    assert asian.session_name == "ASIAN_LOW_LIQUIDITY"
    assert asian.reason == "Low-liquidity session pause"

    # 6. Explicit News Blackout window (+/- 30m around 2026-06-10 12:30 UTC)
    cpi_hit = guard.evaluate("EUR_USD", datetime(2026, 6, 10, 12, 45, tzinfo=timezone.utc))
    assert cpi_hit.can_trade is False
    assert cpi_hit.is_news_blackout is True
    assert "US CPI YoY" in cpi_hit.reason

    # 7. Default recurring NFP guard (First Friday of June 2026 = June 5, 13:00 UTC)
    nfp_hit = guard.evaluate("EUR_USD", datetime(2026, 6, 5, 13, 0, tzinfo=timezone.utc))
    assert nfp_hit.can_trade is False
    assert nfp_hit.is_news_blackout is True
    assert "Non-Farm Payrolls" in nfp_hit.reason

    # 8. Deriv 24/7 Synthetic Index bypass (R_100 and 1HZ100V trade cleanly on Saturday)
    syn_r100 = guard.evaluate("R_100", datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc))
    assert syn_r100.can_trade is True
    assert syn_r100.is_weekend_closed is False
    assert syn_r100.session_name == "SYNTHETIC_24_7"

    syn_1hz = guard.evaluate("1HZ100V", datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc))
    assert syn_1hz.can_trade is True


def test_oanda_candles_granularity_precision_orders_and_retry() -> None:
    """OandaForexBroker parses mid OHLC, formats pip precision, signs units, and retries 429."""
    assert OandaForexBroker.map_granularity("15m") == "M15"
    assert OandaForexBroker.map_granularity("4h") == "H4"
    assert OandaForexBroker.map_granularity("1d") == "D"

    assert OandaForexBroker.format_price("EUR_USD", 1.084567) == "1.08457"
    assert OandaForexBroker.format_price("USD_JPY", 154.1234) == "154.123"
    assert OandaForexBroker.format_price("XAU_USD", 2651.456) == "2651.46"

    with pytest.raises(ValueError):
        OandaForexBroker(api_key="k", account_id="a", base_url="https://api-fxtrade.oanda.com/v3")

    calls: list[dict[str, Any]] = []
    attempt_counter = {"summary": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls.append({"method": request.method, "url": str(request.url), "body": request.content.decode("utf-8")})
        assert request.headers.get("Authorization") == "Bearer oanda-test-key"

        if request.url.path.endswith("/candles"):
            return httpx.Response(
                200,
                json={
                    "candles": [
                        {
                            "complete": True,
                            "volume": 120,
                            "time": "2026-06-09T12:00:00.000000000Z",
                            "mid": {"o": "1.08000", "h": "1.08200", "l": "1.07900", "c": "1.08150"},
                        },
                        {
                            "complete": True,
                            "volume": 150,
                            "time": "2026-06-09T12:15:00.000000000Z",
                            "mid": {"o": "1.08150", "h": "1.08300", "l": "1.08100", "c": "1.08250"},
                        },
                        {
                            "complete": False,
                            "volume": 40,
                            "time": "2026-06-09T12:30:00.000000000Z",
                            "mid": {"o": "1.08250", "h": "1.08350", "l": "1.08200", "c": "1.08310"},
                        },
                    ]
                },
            )

        if request.url.path.endswith("/orders") and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "orderFillTransaction": {
                        "id": "9001",
                        "price": "1.08250",
                        "tradeOpened": {"tradeID": "9001"},
                    }
                },
            )

        if request.url.path.endswith("/trades/9001/orders") and request.method == "PUT":
            return httpx.Response(200, json={"lastTransactionID": "9002"})

        if request.url.path.endswith("/summary"):
            attempt_counter["summary"] += 1
            if attempt_counter["summary"] == 1:
                return httpx.Response(429, text="Rate limited")
            return httpx.Response(200, json={"account": {"NAV": "10450.25", "balance": "10400.00"}})

        return httpx.Response(404)

    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(_handler))
    broker = OandaForexBroker(
        api_key="oanda-test-key",
        account_id="101-001-999999-001",
        base_url=OANDA_PRACTICE_URL,
        client=client,
        sleep_fn=sleeps.append,
    )

    df = broker.get_ohlcv("EUR_USD", interval="15m", limit=3)
    assert len(df) == 3
    assert bool(df["complete"].iloc[-2]) is True
    assert list(df["close"]) == [1.0815, 1.0825, 1.0831]

    order = broker.place_order(
        symbol="EUR_USD",
        side="SHORT",
        qty=2500.4,
        stop_loss=1.086004,
        take_profit=1.075001,
        strategy_name="trend_rider",
        reason="Test short",
    )
    assert order["units"] == -2500
    assert order["side"] == "SHORT"
    order_call = [c for c in calls if c["method"] == "POST"][0]
    sent_payload = json.loads(order_call["body"])["order"]
    assert sent_payload["type"] == "MARKET"
    assert sent_payload["timeInForce"] == "FOK"
    assert sent_payload["units"] == "-2500"
    assert sent_payload["stopLossOnFill"] == {"price": "1.08600", "timeInForce": "GTC"}
    assert sent_payload["takeProfitOnFill"] == {"price": "1.07500", "timeInForce": "GTC"}

    sl_upd = broker.update_stop_loss("9001", 1.08400, symbol="EUR_USD")
    assert sl_upd["stop_loss"] == 1.084

    bal = broker.get_balance()
    assert bal["forex_equity"] == 10450.25
    assert len(sleeps) == 1


def test_deriv_otp_and_legacy_flows_candles_and_multiplier_order() -> None:
    """DerivForexBroker supports OTP REST URL, legacy authorize, ticks_history, and MULTUP/MULTDOWN."""
    assert DerivForexBroker.map_granularity("15m") == 900
    assert DerivForexBroker.map_granularity("4h") == 14400
    assert DerivForexBroker.build_ping_payload() == {"ping": 1}

    # 1. Public candles via ticks_history (synthetic volume=1.0)
    ws_urls_called: list[str] = []
    candle_ws = _FakeWS(
        [
            {
                "candles": [
                    {"epoch": 1700000900, "open": 500.0, "high": 505.0, "low": 499.0, "close": 504.0},
                    {"epoch": 1700000000, "open": 495.0, "high": 501.0, "low": 494.0, "close": 500.0},
                ]
            }
        ]
    )

    def _ws_factory_candles(url: str):
        ws_urls_called.append(url)
        return candle_ws

    deriv_pub = DerivForexBroker(api_token="tok", app_id="1089", ws_connect=_ws_factory_candles)
    df = deriv_pub.get_ohlcv("R_100", interval="15m", limit=2)
    assert ws_urls_called[0] == DERIV_PUBLIC_WS_URL
    assert list(df["timestamp"]) == [1700000000000, 1700000900000]
    assert list(df["volume"]) == [1.0, 1.0]
    assert candle_ws.sent[0]["ticks_history"] == "R_100"
    assert candle_ws.sent[0]["granularity"] == 900

    # 2. Modern 2026 OTP flow (auto mode when account_id provided) + MULTUP proposal -> buy
    otp_headers_seen: dict[str, str] = {}

    def _otp_handler(request: httpx.Request) -> httpx.Response:
        otp_headers_seen.update(dict(request.headers))
        assert request.url.path.endswith("/accounts/VRTC12345/otp")
        return httpx.Response(200, json={"data": {"url": "wss://api.derivws.com/otp/single-use-123"}})

    otp_ws = _FakeWS(
        [
            {"proposal": {"id": "prop-abc-1", "ask_price": 25.0}},
            {"buy": {"contract_id": 778899, "buy_price": 25.0}},
        ]
    )
    otp_urls: list[str] = []

    def _ws_factory_otp(url: str):
        otp_urls.append(url)
        return otp_ws

    deriv_otp = DerivForexBroker(
        api_token="deriv-secret-tok",
        app_id="1089",
        account_id="VRTC12345",
        auth_mode="auto",
        http_client=httpx.Client(transport=httpx.MockTransport(_otp_handler)),
        ws_connect=_ws_factory_otp,
    )
    assert deriv_otp.resolve_auth_mode() == "otp"
    res_up = deriv_otp.place_order(
        symbol="R_100",
        side="LONG",
        qty=25.0,
        stop_loss=490.0,
        take_profit=525.0,
        strategy_name="trend_rider",
        reason="Test MULTUP",
        entry_price_ref=500.0,
        stake_usd=25.0,
        multiplier=10,
    )
    assert otp_headers_seen.get("authorization") == "Bearer deriv-secret-tok"
    assert otp_headers_seen.get("deriv-app-id") == "1089"
    assert otp_urls == ["wss://api.derivws.com/otp/single-use-123"]
    assert res_up["contract_type"] == "MULTUP"
    assert res_up["contract_id"] == "778899"
    assert 0 < res_up["stop_loss_usd"] <= 25.0
    assert otp_ws.sent[0]["proposal"] == 1
    assert otp_ws.sent[0]["limit_order"]["stop_loss"] == res_up["stop_loss_usd"]
    assert otp_ws.sent[1] == {"buy": "prop-abc-1", "price": 25.0}

    # 3. Legacy v3 WS flow (auto mode when account_id blank) + MULTDOWN
    legacy_ws = _FakeWS(
        [
            {"authorize": {"loginid": "VRTC999", "is_virtual": 1}},
            {"proposal": {"id": "prop-down-2", "ask_price": 20.0}},
            {"buy": {"contract_id": 889900, "buy_price": 20.0}},
        ]
    )
    legacy_urls: list[str] = []

    def _ws_factory_legacy(url: str):
        legacy_urls.append(url)
        return legacy_ws

    deriv_legacy = DerivForexBroker(
        api_token="legacy-tok",
        app_id="1089",
        account_id="",
        auth_mode="auto",
        ws_connect=_ws_factory_legacy,
    )
    assert deriv_legacy.resolve_auth_mode() == "legacy"
    res_dn = deriv_legacy.place_order(
        symbol="frxEURUSD",
        side="SHORT",
        qty=20.0,
        stop_loss=1.0950,
        take_profit=1.0700,
        strategy_name="trend_rider",
        reason="Test MULTDOWN",
        entry_price_ref=1.0850,
        stake_usd=20.0,
        multiplier=20,
    )
    assert legacy_urls == [DERIV_LEGACY_WS_TEMPLATE.format(app_id="1089")]
    assert legacy_ws.sent[0] == {"authorize": "legacy-tok"}
    assert legacy_ws.sent[1]["contract_type"] == "MULTDOWN"
    assert legacy_ws.sent[2] == {"buy": "prop-down-2", "price": 20.0}
    assert res_dn["contract_id"] == "889900"


def test_forex_router_paper_fallback_when_keys_blank() -> None:
    """ForexRouter falls back cleanly to PaperSimBroker with PAPER_FALLBACK when keys are blank."""
    r_oanda = ForexRouter(provider="oanda", oanda_api_key="", oanda_account_id="")
    assert isinstance(r_oanda.get_broker(), PaperSimBroker)
    assert not r_oanda.is_cloud_active()
    assert r_oanda.get_status()["status"] == "PAPER_FALLBACK"

    r_deriv = ForexRouter(provider="deriv", deriv_api_token="")
    assert isinstance(r_deriv.get_broker(), PaperSimBroker)
    assert not r_deriv.is_cloud_active()
    assert r_deriv.get_status()["status"] == "PAPER_FALLBACK"


def test_summarizer_math_edge_cases_and_api_summary(tmp_path) -> None:
    """BotSummarizer computes exact accuracy/edge/PF/expectancy, handles 0-loss/0-trade, and serves /api/summary."""
    # 1. Zero trades -> zero-division safe + CALIBRATING
    zero = BotSummarizer.compute_core_metrics([])
    assert zero["total_closed_trades"] == 0
    assert zero["win_rate_pct"] == 0.0
    assert zero["profit_factor"] == 0.0
    assert zero["health_grade"] == "CALIBRATING"

    # 2. Fewer than 5 trades with 0 losses -> zero-division safe + CALIBRATING
    few_wins = BotSummarizer.compute_core_metrics([{"pnl_usd": 50.0}, {"pnl_usd": 40.0}])
    assert few_wins["winning_trades"] == 2
    assert few_wins["losing_trades"] == 0
    assert few_wins["win_rate_pct"] == 100.0
    assert few_wins["profit_factor"] > 0.0
    assert few_wins["health_grade"] == "CALIBRATING"

    # 3. Exact math check with 5 trades: 3 wins (+100 each), 2 losses (-50 each)
    # win_rate = 60%, avg_win = 100, avg_loss = 50 -> avg_rr = 2.0
    # break_even_win_rate = 100 / (1 + 2) = 33.33%
    # edge_margin = 60.0 - 33.33 = 26.67%
    # profit_factor = 300 / 100 = 3.0
    # expectancy = (0.6 * 100) - (0.4 * 50) = 60 - 20 = +40.0
    sample = [
        {"pnl_usd": 100.0},
        {"pnl_usd": -50.0},
        {"pnl_usd": 100.0},
        {"pnl_usd": -50.0},
        {"pnl_usd": 100.0},
    ]
    m = BotSummarizer.compute_core_metrics(sample, starting_equity=10000.0)
    assert m["total_closed_trades"] == 5
    assert m["win_rate_pct"] == 60.0
    assert m["avg_rr_ratio"] == 2.0
    assert m["break_even_win_rate_pct"] == 33.33
    assert m["edge_margin_pct"] == 26.67
    assert m["profit_factor"] == 3.0
    assert m["expectancy_usd"] == 40.0
    assert m["net_pnl_usd"] == 200.0
    assert m["health_grade"] == "A+"

    # 4. API integration test for GET /api/summary (ALL, BYBIT, FOREX) & /api/brokers/status forex_session
    db_path = tmp_path / "phase4_api.db"
    app = create_app(database_url=f"sqlite:///{db_path}")
    with TestClient(app) as client:
        r_status = client.get("/api/brokers/status")
        assert r_status.status_code == 200
        status_data = r_status.json()
        assert "forex_session" in status_data
        assert {"can_trade", "session_name", "is_news_blackout"}.issubset(
            status_data["forex_session"].keys()
        )

        for market in ("ALL", "BYBIT", "FOREX"):
            r_sum = client.get(f"/api/summary?market={market}")
            assert r_sum.status_code == 200
            data = r_sum.json()
            assert data["market"] == market
            assert {
                "total_closed_trades",
                "winning_trades",
                "losing_trades",
                "win_rate_pct",
                "break_even_win_rate_pct",
                "edge_margin_pct",
                "profit_factor",
                "expectancy_usd",
                "health_grade",
                "strategy_rankings",
                "regime_breakdown",
                "risk_shield_report",
                "plain_english",
            }.issubset(data.keys())
            assert isinstance(data["plain_english"]["headline"], str)
            assert len(data["plain_english"]["bullets"]) >= 3
