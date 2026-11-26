"""Phase 5 offline unit & integration tests: Partial Scale-Out, Zero-Lookahead Backtester, Geo-Fallback & API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pandas as pd
import pytest

from app.database import create_db_engine, init_db
from app.engine.backtester import Backtester
from app.engine.historical_data import HistoricalDataFetcher
from app.engine.regime import MarketRegime, RegimeResult
from app.engine.risk_manager import PartialScaleOutDecision, RiskManager
from app.engine.strategies import StrategyDispatcher, TradeSignal
from app.main import create_app


def _make_deterministic_ohlcv(bars: int = 360, start_price: float = 100.0) -> pd.DataFrame:
    """Build deterministic multi-regime OHLCV suitable for fast offline backtest tests."""
    return HistoricalDataFetcher.generate_synthetic_fallback(
        symbol="BTCUSDT",
        interval="1h",
        days=max(16, int(bars / 24) + 1),
        now=datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc),
    ).iloc[:bars].reset_index(drop=True)


def test_evaluate_partial_scale_out_long_and_short_rules() -> None:
    """Verify +1.0R partial scale-out for LONG/SHORT, no-op below +1.0R, idempotency, and never moving SL backward."""
    # LONG: entry 100, SL 95 (R = 5). At 104.5 (+0.9R) -> no-op
    below = RiskManager.evaluate_partial_scale_out(
        side="LONG",
        entry_price=100.0,
        current_price=104.5,
        current_sl=95.0,
        initial_risk=5.0,
        current_qty=2.0,
        partial_already_taken=False,
    )
    assert isinstance(below, PartialScaleOutDecision)
    assert below.should_scale_out is False
    assert below.close_qty == 0.0
    assert below.remaining_qty == 2.0
    assert below.new_stop_loss == 95.0

    # LONG at exactly +1.0R (105.0) -> closes 50% (1.0), moves SL to entry + 0.05*R = 100.25
    at_1r = RiskManager.evaluate_partial_scale_out(
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        current_sl=95.0,
        initial_risk=5.0,
        current_qty=2.0,
        partial_already_taken=False,
    )
    assert at_1r.should_scale_out is True
    assert at_1r.close_qty == pytest.approx(1.0)
    assert at_1r.remaining_qty == pytest.approx(1.0)
    assert at_1r.new_stop_loss == pytest.approx(100.25)
    assert at_1r.locked_r == pytest.approx(1.0)

    # LONG never triggers twice when partial_already_taken=True
    twice = RiskManager.evaluate_partial_scale_out(
        side="LONG",
        entry_price=100.0,
        current_price=110.0,
        current_sl=100.25,
        initial_risk=5.0,
        current_qty=1.0,
        partial_already_taken=True,
    )
    assert twice.should_scale_out is False
    assert twice.close_qty == 0.0
    assert twice.new_stop_loss == pytest.approx(100.25)

    # LONG never moves SL backward if current_sl is already above +0.05R
    tight_sl = RiskManager.evaluate_partial_scale_out(
        side="LONG",
        entry_price=100.0,
        current_price=106.0,
        current_sl=102.0,
        initial_risk=5.0,
        current_qty=2.0,
        partial_already_taken=False,
    )
    assert tight_sl.should_scale_out is True
    assert tight_sl.new_stop_loss == pytest.approx(102.0)

    # SHORT: entry 100, SL 105 (R = 5). At 95.0 (+1.0R) -> closes 50%, moves SL to 99.75
    short_1r = RiskManager.evaluate_partial_scale_out(
        side="SHORT",
        entry_price=100.0,
        current_price=95.0,
        current_sl=105.0,
        initial_risk=5.0,
        current_qty=4.0,
        partial_already_taken=False,
    )
    assert short_1r.should_scale_out is True
    assert short_1r.close_qty == pytest.approx(2.0)
    assert short_1r.remaining_qty == pytest.approx(2.0)
    assert short_1r.new_stop_loss == pytest.approx(99.75)

    # SHORT never moves SL backward if current_sl is already lower than entry - 0.05*R
    short_tight = RiskManager.evaluate_partial_scale_out(
        side="SHORT",
        entry_price=100.0,
        current_price=94.0,
        current_sl=98.0,
        initial_risk=5.0,
        current_qty=4.0,
        partial_already_taken=False,
    )
    assert short_tight.should_scale_out is True
    assert short_tight.new_stop_loss == pytest.approx(98.0)


def test_sqlite_idempotent_migration_upgrades_legacy_trades_table(tmp_path: Path) -> None:
    """Verify init_db cleanly adds Phase 5 columns to a pre-existing legacy SQLite database."""
    db_file = tmp_path / "legacy_apex.db"
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            market VARCHAR(16),
            symbol VARCHAR(32),
            side VARCHAR(8),
            strategy_name VARCHAR(64),
            status VARCHAR(16),
            qty FLOAT,
            entry_price FLOAT,
            current_price FLOAT,
            exit_price FLOAT,
            stop_loss FLOAT,
            take_profit FLOAT,
            pnl_usd FLOAT,
            pnl_pct FLOAT,
            entry_reason TEXT,
            opened_at DATETIME,
            closed_at DATETIME
        )
        """
    )
    conn.commit()
    conn.close()

    eng = create_db_engine(f"sqlite:///{db_file}")
    init_db(target_engine=eng)
    # Run twice to guarantee idempotency
    init_db(target_engine=eng)

    conn = sqlite3.connect(db_file)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    conn.close()
    assert {"initial_qty", "partial_tp_taken", "realized_partial_pnl_usd"}.issubset(cols)


def test_backtester_zero_lookahead_proof() -> None:
    """Bars 301..350 with a massive +500% spike MUST NOT alter any trade entry up to bar 300."""
    df_full = _make_deterministic_ohlcv(bars=355)
    df_300 = df_full.iloc[:300].copy().reset_index(drop=True)
    df_350_spiked = df_full.iloc[:350].copy().reset_index(drop=True)

    # Inject a massive +500% spike in bars 301..349
    for col in ("open", "high", "low", "close"):
        df_350_spiked.loc[301:, col] = df_350_spiked.loc[301:, col] * 6.0
    df_350_spiked.loc[301:, "volume"] = df_350_spiked.loc[301:, "volume"] * 10.0

    cutoff_iso = df_300.iloc[-1]["timestamp"].isoformat()

    bt_short = Backtester().run(
        df=df_300,
        symbol="BTCUSDT",
        interval="1h",
        starting_balance=10000.0,
        enable_partial_tp=True,
        enable_confluence=False,
    )
    bt_long = Backtester().run(
        df=df_350_spiked,
        symbol="BTCUSDT",
        interval="1h",
        starting_balance=10000.0,
        enable_partial_tp=True,
        enable_confluence=False,
    )

    entries_short = [
        (t["entry_time"], t["side"], t["entry_price"], t["strategy_name"])
        for t in bt_short.recent_trades
        if t["entry_time"] <= cutoff_iso
    ]
    entries_long = [
        (t["entry_time"], t["side"], t["entry_price"], t["strategy_name"])
        for t in bt_long.recent_trades
        if t["entry_time"] <= cutoff_iso
    ]
    assert entries_short == entries_long


class _SingleSignalDispatcher(StrategyDispatcher):
    """Deterministic stub dispatcher emitting a single BUY signal at a target timestamp."""

    def __init__(self, target_ts: pd.Timestamp, stop_dist: float, reward_dist: float) -> None:
        super().__init__()
        self._target_ts = target_ts
        self._stop_dist = stop_dist
        self._reward_dist = reward_dist

    def evaluate(self, df: pd.DataFrame, enabled_strategy_names: set[str]):
        closed = df.iloc[-2]
        regime = RegimeResult(
            regime=MarketRegime.TRENDING,
            confidence=0.9,
            adx=32.0,
            atr_pct=0.8,
            bbw_percentile=50.0,
            reason="Test Trending",
        )
        if pd.Timestamp(closed["timestamp"]) == self._target_ts:
            close = float(closed["close"])
            return regime, TradeSignal(
                action="BUY",
                strategy_name="trend_rider",
                regime=MarketRegime.TRENDING.value,
                confidence=0.9,
                entry_price=close,
                stop_loss=close - self._stop_dist,
                take_profit=close + self._reward_dist,
                reason="Test BUY signal",
            )
        return regime, TradeSignal(
            action="HOLD",
            strategy_name="trend_rider",
            regime=MarketRegime.TRENDING.value,
            confidence=0.0,
            entry_price=float(closed["close"]),
            stop_loss=0.0,
            take_profit=0.0,
            reason="Hold",
        )


def test_backtester_pessimistic_intrabar_tie_breaker_and_fees_slippage() -> None:
    """When one candle hits BOTH SL and TP simultaneously, backtester MUST close at SL (loss) and deduct fees/slippage."""
    n = 220
    base_ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": base_ts + timedelta(hours=i),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 100.0,
            }
        )
    df = pd.DataFrame(rows)
    signal_bar_idx = 206
    # Entry executes at open of bar 207 = 100.0 * (1 + slippage)
    # Bar 207 itself has a giant candle hitting BOTH low=80 (below SL ~98) AND high=130 (above TP ~105)
    df.loc[207, "open"] = 100.0
    df.loc[207, "high"] = 130.0
    df.loc[207, "low"] = 80.0
    df.loc[207, "close"] = 125.0

    dispatcher = _SingleSignalDispatcher(
        target_ts=pd.Timestamp(df.loc[signal_bar_idx, "timestamp"]),
        stop_dist=2.0,
        reward_dist=5.0,
    )
    bt = Backtester(dispatcher=dispatcher)
    res = bt.run(
        df=df,
        symbol="BTCUSDT",
        interval="1h",
        starting_balance=10000.0,
        fee_pct=0.0010,
        slippage_pct=0.0005,
        enable_partial_tp=True,
        enable_confluence=False,
    )

    assert res.total_trades == 1
    assert res.losing_trades == 1
    assert res.winning_trades == 0
    trade = res.recent_trades[0]
    assert trade["exit_reason"] == "Stop Loss"
    assert trade["entry_price"] == pytest.approx(100.05, rel=1e-4)  # 100 * (1 + 0.0005)
    assert trade["fees_usd"] > 0.0
    assert trade["pnl_usd"] < 0.0
    assert res.ending_balance < 10000.0


def test_backtester_daily_3pct_circuit_breaker_trips() -> None:
    """Verify the 3.0% UTC daily drawdown circuit breaker trips and blocks further same-day entries."""
    n = 230
    base_ts = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    rows = [
        {
            "timestamp": base_ts + timedelta(hours=i),
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
            "volume": 100.0,
        }
        for i in range(n)
    ]
    df = pd.DataFrame(rows)

    # Use a 2% risk-per-trade RiskManager with 3% daily max loss and 0 cooldown so 2 consecutive losses breach -3%
    class _AlwaysBuyDispatcher(StrategyDispatcher):
        def evaluate(self, frame: pd.DataFrame, enabled_strategy_names: set[str]):
            close = float(frame.iloc[-2]["close"])
            reg = RegimeResult(MarketRegime.TRENDING, 0.9, 30.0, 1.0, 50.0, "Test")
            return reg, TradeSignal("BUY", "trend_rider", "TRENDING", 0.9, close, close - 2.0, close + 4.0, "Buy")

    # Bars 206..220 all happen on the same UTC day and each crashes below SL
    for idx in range(206, 220):
        df.loc[idx, "timestamp"] = datetime(2026, 2, 10, idx - 206, 0, tzinfo=timezone.utc)
        df.loc[idx, "low"] = 90.0

    rm = RiskManager(
        max_risk_per_trade_pct=2.0,
        max_daily_loss_pct=3.0,
        max_notional_pct=100.0,
        loss_cooldown_candles=0,
    )
    res = Backtester(dispatcher=_AlwaysBuyDispatcher(), risk_manager=rm).run(
        df=df,
        symbol="BTCUSDT",
        interval="1h",
        starting_balance=10000.0,
        enable_partial_tp=False,
        enable_confluence=False,
    )

    assert res.circuit_breaker_trips >= 1
    assert res.total_trades == 2  # 2 losses (~-4%) trip the -3% breaker and halt further entries that day


def test_historical_data_fetcher_bybit_403_falls_back_to_coinbase_and_deriv_ws() -> None:
    """Verify Bybit HTTP 403 -> Coinbase Public fallback and Deriv Public WebSocket mocked pagination."""
    now_sec = int(datetime.now(timezone.utc).timestamp())
    cb_candles = [
        {
            "start": str(now_sec - 3600 * i),
            "low": "90000.0",
            "high": "91000.0",
            "open": "90200.0",
            "close": "90800.0",
            "volume": "150.5",
        }
        for i in range(30)
    ]

    seen_urls: list[str] = []

    def _mock_transport_handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen_urls.append(url)
        if "api.bybit.com" in url:
            return httpx.Response(403, json={"retCode": 10009, "retMsg": "Geo-blocked"})
        if "api.coinbase.com" in url:
            assert "BTC-USD/candles" in url
            return httpx.Response(200, json={"candles": cb_candles})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(_mock_transport_handler))
    fetcher = HistoricalDataFetcher(http_client=client)
    df, source = fetcher.fetch(symbol="BTCUSDT", interval="1h", days=7)

    assert source == "COINBASE_PUBLIC"
    assert len(df) == 30
    assert df["timestamp"].is_monotonic_increasing
    assert any("api.bybit.com" in u for u in seen_urls)
    assert any("api.coinbase.com" in u for u in seen_urls)

    # Test Deriv Public WebSocket mocked fetch for EUR_USD -> frxEURUSD
    sent_payloads: list[dict] = []

    class _FakeWS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def send(self, msg: str) -> None:
            sent_payloads.append(json.loads(msg))

        def recv(self) -> str:
            candles = [
                {
                    "epoch": now_sec - 3600 * (25 - idx),
                    "open": 1.08 + idx * 0.0001,
                    "high": 1.081 + idx * 0.0001,
                    "low": 1.079 + idx * 0.0001,
                    "close": 1.0805 + idx * 0.0001,
                }
                for idx in range(25)
            ]
            # First call returns candles, second stops pagination
            if len(sent_payloads) == 1:
                return json.dumps({"candles": candles})
            return json.dumps({"candles": []})

    deriv_fetcher = HistoricalDataFetcher(
        http_client=client,
        ws_connect=lambda url, **kwargs: _FakeWS(),
    )
    df_fx, fx_source = deriv_fetcher.fetch(symbol="EUR_USD", interval="1h", days=3)
    assert fx_source == "DERIV_PUBLIC"
    assert len(df_fx) == 25
    assert sent_payloads[0]["ticks_history"] == "frxEURUSD"
    assert sent_payloads[0]["style"] == "candles"
    assert sent_payloads[0]["count"] == 1000
    assert (df_fx["volume"] == 1.0).all()


def test_api_backtest_run_and_chart_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POST /api/backtest/run (mocked fetcher) and GET /api/chart/BTCUSDT ascending OHLCV + open trade lines."""
    sample_df = _make_deterministic_ohlcv(bars=260)
    monkeypatch.setattr(
        HistoricalDataFetcher,
        "fetch",
        lambda self, symbol, interval="1h", days=60: (sample_df, "COINBASE_PUBLIC"),
    )

    db_url = f"sqlite:///{tmp_path / 'phase5_api.db'}"
    app = create_app(database_url=db_url)
    with TestClient(app) as client:
        # 1. POST /api/backtest/run
        resp = client.post(
            "/api/backtest/run",
            json={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "days": 30,
                "starting_balance": 10000.0,
                "enable_partial_tp": True,
                "enable_confluence": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "BTCUSDT"
        assert data["data_source"] == "COINBASE_PUBLIC"
        assert data["candles_tested"] == 260
        assert "net_pnl_pct" in data
        assert "buy_and_hold_pct" in data
        assert "equity_curve" in data
        assert len(data["equity_curve"]) <= 200

        # Bounds validation check
        bad = client.post("/api/backtest/run", json={"symbol": "BTCUSDT", "days": 999})
        assert bad.status_code == 422

        # 2. GET /api/chart/BTCUSDT
        chart_resp = client.get("/api/chart/BTCUSDT?market=BYBIT&interval=15m&limit=120")
        assert chart_resp.status_code == 200
        chart = chart_resp.json()
        assert chart["symbol"] == "BTCUSDT"
        candles = chart["candles"]
        assert len(candles) >= 50
        times = [c["time"] for c in candles]
        assert times == sorted(set(times))  # strictly ascending & deduplicated
        # Seeded BTCUSDT demo trade should be returned with Entry/SL/TP levels
        assert chart["open_trade"] is not None
        assert chart["open_trade"]["side"] == "LONG"
        assert chart["open_trade"]["entry_price"] > 0
        assert chart["open_trade"]["stop_loss"] > 0
        assert chart["open_trade"]["take_profit"] > 0
