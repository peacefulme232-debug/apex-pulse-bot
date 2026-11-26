"""Phase 3 deterministic unit & API tests (100% mocked, zero live network calls)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from app.brokers.bybit_broker import BybitBroker, BybitBrokerError
from app.brokers.bybit_router import BybitRouter
from app.brokers.paper_sim import PaperSimBroker
from app.config import MAX_LEVERAGE, Settings
from app.engine.confluence import ConfluenceFilter
from app.engine.strategies import TradeSignal
from app.main import create_app


def _make_ohlcv_df(
    bars: int = 120,
    start_price: float = 100.0,
    step: float = 0.5,
    last_closed_volume: float = 2000.0,
    base_volume: float = 1000.0,
) -> pd.DataFrame:
    closes = [start_price + i * step for i in range(bars)]
    opens = [c - step * 0.4 for c in closes]
    highs = [max(o, c) + abs(step) * 0.5 + 0.2 for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(step) * 0.5 - 0.2 for o, c in zip(opens, closes)]
    vols = [base_volume for _ in range(bars)]
    vols[-2] = last_closed_volume
    return pd.DataFrame(
        {
            "timestamp": [1700000000000 + i * 900_000 for i in range(bars)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        }
    )


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.get_instruments_info.return_value = {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "lotSizeFilter": {
                        "qtyStep": "0.001",
                        "minOrderQty": "0.001",
                        "maxOrderQty": "100.0",
                        "minNotionalValue": "5",
                    },
                    "priceFilter": {"tickSize": "0.10"},
                }
            ]
        },
    }
    session.set_leverage.return_value = {"retCode": 0, "retMsg": "OK"}
    session.place_order.return_value = {
        "retCode": 0,
        "result": {"orderId": "demo-ord-123", "orderLinkId": "link-123"},
    }
    session.set_trading_stop.return_value = {"retCode": 0, "retMsg": "OK"}
    session.get_wallet_balance.return_value = {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "coin": [
                        {"coin": "USDT", "equity": "10250.50", "walletBalance": "10200.00"}
                    ]
                }
            ]
        },
    }
    session.get_positions.return_value = {
        "retCode": 0,
        "result": {
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.015",
                    "avgPrice": "94000.0",
                    "markPrice": "94500.0",
                    "stopLoss": "93200.0",
                    "takeProfit": "96000.0",
                    "unrealisedPnl": "7.50",
                }
            ]
        },
    }
    return session


def test_config_safety_locks() -> None:
    """Live mode is blocked, blank keys force paper mode, and leverage is capped at 3."""
    s_blank = Settings(BYBIT_MODE="demo", BYBIT_API_KEY="", BYBIT_API_SECRET="")
    assert s_blank.BYBIT_MODE == "paper"

    with pytest.raises(ValueError):
        Settings(BYBIT_MODE="live")

    with pytest.raises(ValueError):
        Settings(BYBIT_LEVERAGE=MAX_LEVERAGE + 1)

    s_valid = Settings(BYBIT_MODE="demo", BYBIT_API_KEY="k", BYBIT_API_SECRET="s", BYBIT_LEVERAGE=3)
    assert s_valid.BYBIT_MODE == "demo"
    assert s_valid.BYBIT_LEVERAGE == 3


def test_bybit_interval_mapping_and_kline_reverse_sorting() -> None:
    """BybitBroker maps 15m->15, 4h->240 and reverses newest-first klines to ascending."""
    assert BybitBroker.map_interval("15m") == "15"
    assert BybitBroker.map_interval("4h") == "240"
    assert BybitBroker.map_interval("1d") == "D"
    assert BybitBroker.map_interval("15") == "15"

    session = _mock_session()
    # Bybit returns newest-first
    session.get_kline.return_value = {
        "retCode": 0,
        "result": {
            "list": [
                ["1700001800000", "102", "105", "101", "104", "300"],
                ["1700000900000", "101", "103", "100", "102", "200"],
                ["1700000000000", "100", "102", "99", "101", "100"],
            ]
        },
    }
    broker = BybitBroker(api_key="k", api_secret="s", mode="demo", session=session)
    df = broker.get_ohlcv("BTCUSDT", interval="15m", limit=3)

    session.get_kline.assert_called_once_with(category="linear", symbol="BTCUSDT", interval="15", limit=3)
    assert list(df["timestamp"]) == [1700000000000, 1700000900000, 1700001800000]
    assert df["close"].dtype == float
    assert list(df["close"]) == [101.0, 102.0, 104.0]


def test_bybit_decimal_quantization_zero_float_artifacts() -> None:
    """Decimal ROUND_DOWN quantization against qtyStep and tickSize produces exact strings."""
    session = _mock_session()
    broker = BybitBroker(api_key="k", api_secret="s", mode="demo", session=session)

    q_qty = broker.quantize_qty("BTCUSDT", 0.12399999999999999, price=94000.0)
    q_price = broker.quantize_price("BTCUSDT", 94123.49999999999)

    assert q_qty == "0.123"
    assert q_price == "94123.40"
    assert "e" not in q_qty.lower()
    assert Decimal(q_qty) == Decimal("0.123")

    with pytest.raises(ValueError):
        broker.quantize_qty("BTCUSDT", 0.0005, price=94000.0)


def test_bybit_leverage_110043_non_fatal_and_order_and_trailing_sl() -> None:
    """retCode 110043 is non-fatal, place_order uses REST Market+Full TPSL, and set_trading_stop updates SL."""
    session = _mock_session()
    session.set_leverage.return_value = {"retCode": 110043, "retMsg": "leverage not modified"}

    broker = BybitBroker(api_key="k", api_secret="s", mode="demo", leverage=2, session=session)
    order = broker.place_order(
        symbol="BTCUSDT",
        side="BUY",
        qty=0.0159,
        stop_loss=93123.49,
        take_profit=96500.09,
        strategy_name="trend_rider",
        reason="Test",
        entry_price_ref=94500.0,
    )
    assert order["order_id"] == "demo-ord-123"
    assert order["qty"] == 0.015
    assert order["stop_loss"] == 93123.4

    session.place_order.assert_called_once_with(
        category="linear",
        symbol="BTCUSDT",
        side="Buy",
        orderType="Market",
        qty="0.015",
        stopLoss="93123.40",
        takeProfit="96500.00",
        tpslMode="Full",
    )

    sl_res = broker.update_stop_loss("BTCUSDT", 94000.09)
    assert sl_res["stop_loss"] == 94000.0
    session.set_trading_stop.assert_called_once_with(
        category="linear",
        symbol="BTCUSDT",
        stopLoss="94000.00",
        positionIdx=0,
    )


def test_bybit_error_handling_10003_and_10006_retry_and_10009() -> None:
    """Verify 10006 exponential backoff retry, 10003 auth error, and 10009 geo-block."""
    session = _mock_session()
    sleeps: list[float] = []

    # 10006 twice then success
    session.get_wallet_balance.side_effect = [
        {"retCode": 10006, "retMsg": "Too many visits"},
        {"retCode": 10006, "retMsg": "Too many visits"},
        {"retCode": 0, "result": {"list": [{"coin": [{"coin": "USDT", "equity": "5000.0", "walletBalance": "5000.0"}]}]}},
    ]
    broker = BybitBroker(api_key="k", api_secret="s", mode="demo", session=session, sleep_fn=sleeps.append)
    bal = broker.get_balance()
    assert bal["bybit_equity"] == 5000.0
    assert len(sleeps) == 2

    # 10003 auth mismatch
    session.get_wallet_balance.side_effect = [{"retCode": 10003, "retMsg": "Invalid api_key"}]
    with pytest.raises(BybitBrokerError) as exc_10003:
        broker.get_balance()
    assert exc_10003.value.ret_code == 10003
    assert broker.status == "ERROR"
    assert "Mainnet Demo Trading" in broker.last_sync_message

    # 10009 geo-block
    session.get_wallet_balance.side_effect = [{"retCode": 10009, "retMsg": "Forbidden IP"}]
    with pytest.raises(BybitBrokerError) as exc_10009:
        broker.get_balance()
    assert exc_10009.value.ret_code == 10009
    assert broker.status == "GEO_BLOCKED"


def test_bybit_router_paper_fallback_when_keys_empty() -> None:
    """BybitRouter falls back cleanly to PaperSimBroker with PAPER_FALLBACK when keys are blank."""
    router = BybitRouter(bybit_mode="demo", api_key="", api_secret="")
    assert isinstance(router.get_broker(), PaperSimBroker)
    assert not router.is_demo_active()
    status = router.get_status()
    assert status["status"] == "PAPER_FALLBACK"
    assert status["mode"] == "paper"


def test_confluence_filter_approves_aligned_and_blocks_counter_trend_fakeout() -> None:
    """ConfluenceFilter approves high-score MTF+VWAP+Vol setups and blocks counter-HTF low-vol fakeouts."""
    cf = ConfluenceFilter(min_score=65)
    ltf_bull = _make_ohlcv_df(bars=60, start_price=100.0, step=0.5, last_closed_volume=1800.0, base_volume=1000.0)
    htf_bull = _make_ohlcv_df(bars=70, start_price=80.0, step=0.8, last_closed_volume=5000.0, base_volume=4000.0)

    buy_signal = TradeSignal(
        action="BUY",
        strategy_name="trend_rider",
        regime="TRENDING",
        confidence=0.85,
        entry_price=float(ltf_bull["close"].iloc[-2]),
        stop_loss=float(ltf_bull["close"].iloc[-2]) - 2.0,
        take_profit=float(ltf_bull["close"].iloc[-2]) + 5.0,
        reason="Aligned bullish breakout",
    )
    res_good = cf.evaluate(buy_signal, ltf_bull, htf_df=htf_bull)
    assert res_good.passed is True
    assert res_good.score >= 85
    assert res_good.htf_bias == "BULLISH"
    assert res_good.vwap_aligned is True
    assert res_good.volume_ratio >= 1.15

    # Counter-trend SELL against 4H Bullish + above VWAP + dead volume (0.4x)
    ltf_low_vol = _make_ohlcv_df(bars=60, start_price=100.0, step=0.5, last_closed_volume=400.0, base_volume=1000.0)
    sell_fakeout = TradeSignal(
        action="SELL",
        strategy_name="trend_rider",
        regime="TRENDING",
        confidence=0.4,
        entry_price=float(ltf_low_vol["close"].iloc[-2]),
        stop_loss=float(ltf_low_vol["close"].iloc[-2]) + 2.0,
        take_profit=float(ltf_low_vol["close"].iloc[-2]) - 2.0,
        reason="Counter-trend short",
    )
    res_bad = cf.evaluate(sell_fakeout, ltf_low_vol, htf_df=htf_bull)
    assert res_bad.passed is False
    assert res_bad.score < 65
    assert any("Counter-HTF warning" in r for r in res_bad.reasons)

    # HOLD passes through untouched
    hold_sig = TradeSignal(
        action="HOLD",
        strategy_name="safety_standby",
        regime="DEFENSIVE",
        confidence=0.0,
        entry_price=0.0,
        stop_loss=0.0,
        take_profit=0.0,
        reason="Hold",
    )
    res_hold = cf.evaluate(hold_sig, ltf_bull, htf_df=htf_bull)
    assert res_hold.passed is True
    assert res_hold.score == 0


def test_api_brokers_status_and_dashboard_additive_fields(tmp_path) -> None:
    """GET /api/brokers/status and GET /api/dashboard return Phase 3 telemetry without breaking Phase 1/2 keys."""
    db_path = tmp_path / "phase3_api.db"
    app = create_app(database_url=f"sqlite:///{db_path}")
    with TestClient(app) as client:
        r_status = client.get("/api/brokers/status")
        assert r_status.status_code == 200
        b_data = r_status.json()
        assert "bybit" in b_data and "forex" in b_data
        assert b_data["bybit"]["status"] in {"CONNECTED_DEMO", "PAPER_FALLBACK", "GEO_BLOCKED", "ERROR"}
        assert b_data["forex"]["status"] == "PAPER_FALLBACK"

        r_dash = client.get("/api/dashboard?market=ALL")
        assert r_dash.status_code == 200
        d = r_dash.json()
        assert "brokers_status" in d
        assert "last_confluence" in d
        assert {"score", "htf_bias", "vwap_aligned", "volume_ratio"}.issubset(d["last_confluence"].keys())
        assert len(d["strategies"]) == 4
        for strat in d["strategies"]:
            assert "profit_factor" in strat
            assert "max_drawdown_pct" in strat
