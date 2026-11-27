"""Phase 2 Verification Gate: Virtual Wallet / Vault, Withdrawal Fees, Auto-Sweep, $20 Challenge & Auth."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import create_db_engine, init_db
from app.main import create_app
from app.models import BotState, ExecutionRejection, Trade, WalletAccount, WalletTransaction
from app.security import reset_rate_limiter
from app.services.wallet_service import calculate_withdrawal_fees


@pytest.fixture(autouse=True)
def _isolate_phase2_auth(monkeypatch: pytest.MonkeyPatch):
    """Prevent PIN / background loop settings from leaking across tests."""
    reset_rate_limiter()
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "")
    monkeypatch.setattr(settings, "SESSION_SECRET", "")
    monkeypatch.setattr(settings, "ENABLE_BACKGROUND_LOOP", False)
    yield
    reset_rate_limiter()


# ---------------------------------------------------------------------------
# Gate 1: The $20,150.01 -> $20.00 Challenge Test (TRC20 $1.30 fee)
# ---------------------------------------------------------------------------


def test_20150_to_20_challenge_trc20_cashflow_isolation_and_reconciliation(tmp_path: Path) -> None:
    """Withdrawing $20,130.01 via TRC20 ($1.30) leaves $20.00 active, $20,128.71 vault, $0.00 Today's PnL, 0 drift, no Daily Loss Lock."""
    db_url = f"sqlite:///{tmp_path / 'challenge_20.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        # Close seeded demo open trades first, and set Bybit active balance to exact $20,150.01
        engine = create_db_engine(db_url)
        SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        with SessionCls() as db:
            db.query(Trade).delete()
            bybit_w = db.query(WalletAccount).filter(WalletAccount.market == "bybit").first()
            forex_w = db.query(WalletAccount).filter(WalletAccount.market == "forex").first()
            assert bybit_w is not None and forex_w is not None
            bybit_w.seed_capital = Decimal("20150.010000")
            bybit_w.active_balance = Decimal("20150.010000")
            bybit_w.vault_balance = Decimal("0.000000")
            bybit_w.equity_baseline_day = Decimal("20150.010000")
            bybit_w.cumulative_net_cashflow_day = Decimal("0.000000")
            db.commit()

        # Execute 1-click Leave $20.00 via TRC20 ($1.30 fixed fee)
        resp = client.post(
            "/api/wallet/leave-balance",
            json={
                "market": "bybit",
                "target_active_balance": "20.00",
                "network": "TRC20",
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["results"]) == 1
        tx = data["results"][0]["transaction"]
        assert tx["tx_type"] == "LEAVE_BALANCE_CHALLENGE"
        assert tx["network"] == "TRC20"
        assert Decimal(str(tx["gross_amount"])) == Decimal("20130.01")
        assert Decimal(str(tx["network_fee"])) == Decimal("1.3")
        assert Decimal(str(tx["net_amount"])) == Decimal("20128.71")

        # Verify /api/wallet/summary exact numbers & reconciliation drift == 0.000000
        summary = client.get("/api/wallet/summary").json()
        bybit_sum = summary["markets"]["bybit"]
        assert Decimal(str(bybit_sum["active_balance"])) == Decimal("20.0")
        assert Decimal(str(bybit_sum["vault_balance"])) == Decimal("20128.71")
        assert bybit_sum["daily_pnl_usd"] == 0.0
        assert bybit_sum["daily_pnl_pct"] == 0.0
        assert bybit_sum["reconciliation"]["reconciled"] is True
        assert bybit_sum["reconciliation"]["drift"] == "0.000000"

        # Verify /api/dashboard shows Today's PnL == $0.00 and cash-flow-adjusted curve fields
        dash = client.get("/api/dashboard?market=BYBIT").json()
        assert dash["kpis"]["daily_pnl_usd"] == 0.0
        assert dash["kpis"]["daily_pnl_pct"] == 0.0
        assert dash["kpis"]["daily_loss_used_pct"] == 0.0
        assert dash["bot_state"]["kill_switch_active"] is False
        assert "adjusted_pnl_usd" in dash["equity_curve"][0]
        assert "adjusted_equity" in dash["equity_curve"][0]
        assert len(dash["adjusted_trading_pnl_curve"]) == len(dash["equity_curve"])
        assert len(dash["raw_active_balance_curve"]) == len(dash["equity_curve"])
        assert "value" in dash["adjusted_trading_pnl_curve"][0]
        assert "value" in dash["raw_active_balance_curve"][0]

        # Run a BotRunner tick -> Daily Loss Lock MUST NOT trigger
        app.state.bot_runner.run_once()
        with SessionCls() as db:
            state = db.query(BotState).first()
            assert state is not None
            assert state.kill_switch_active is False

            # Next position sizing MUST use strictly $20.00 active_balance (ignoring $20,128.71 Vault)
            sizing = app.state.broker.size_and_validate_order(
                db=db,
                symbol="BTCUSDT",
                side="LONG",
                entry_price=Decimal("77000"),
                stop_loss=Decimal("76000"),
                max_risk_per_trade_pct=Decimal("1.0"),
            )
            db.commit()
            assert sizing.approved is False
            assert sizing.available_free_margin == Decimal("20.000000")
            assert sizing.reason_code in {"BELOW_MIN_NOTIONAL", "INSUFFICIENT_MARGIN"}

        # Rejections endpoint surfaces paginated logged rejections
        rej_resp = client.get("/api/wallet/rejections?market=bybit&limit=10&offset=0")
        assert rej_resp.status_code == 200
        rej_body = rej_resp.json()
        assert rej_body["count"] >= 1
        assert rej_body["total"] >= 1
        assert rej_body["limit"] == 10
        assert rej_body["offset"] == 0


# ---------------------------------------------------------------------------
# Gate 2: Margin Lock Guard ($12 margin on $20 balance -> $15 fails 400, $5 succeeds)
# ---------------------------------------------------------------------------


def test_margin_lock_guard_blocks_over_withdrawal_and_allows_safe_withdrawal(tmp_path: Path) -> None:
    """Open $12 margin position on $20 active balance -> withdraw $15 returns 400; withdraw $5 succeeds."""
    db_url = f"sqlite:///{tmp_path / 'margin_guard.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        engine = create_db_engine(db_url)
        SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        with SessionCls() as db:
            db.query(Trade).delete()
            bybit_w = db.query(WalletAccount).filter(WalletAccount.market == "bybit").first()
            assert bybit_w is not None
            bybit_w.seed_capital = Decimal("20.000000")
            bybit_w.active_balance = Decimal("20.000000")
            bybit_w.vault_balance = Decimal("0.000000")
            bybit_w.equity_baseline_day = Decimal("20.000000")
            bybit_w.bybit_leverage = 5
            db.commit()

            # Open a SOLUSDT position with ~ $12.00 initial_margin (notional ~$60 at 5x -> initial_margin ~$12.03)
            # SOLUSDT qty=0.6 @ $100 -> notional $60, 5x -> IM = $12.00 + fee_to_close($0.0264) = $12.0264
            app.state.broker.place_order(
                symbol="SOLUSDT",
                side="LONG",
                qty=0.6,
                stop_loss=90.0,
                take_profit=120.0,
                strategy_name="trend_rider",
                reason="Margin lock test",
                db=db,
                order_entry_type="LIMIT",
                requested_entry_price=Decimal("100.0"),
            )

        # Attempting to withdraw $15.00 must fail with HTTP 400
        blocked = client.post(
            "/api/wallet/transfer",
            json={
                "market": "bybit",
                "tx_type": "TRADING_TO_VAULT",
                "amount": "15.00",
                "network": "INTERNAL",
            },
        )
        assert blocked.status_code == 400
        assert "exceeds withdrawable" in blocked.json()["detail"].lower()

        # Withdrawing $5.00 succeeds cleanly
        ok = client.post(
            "/api/wallet/transfer",
            json={
                "market": "bybit",
                "tx_type": "TRADING_TO_VAULT",
                "amount": "5.00",
                "network": "INTERNAL",
            },
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["wallet"]["active_balance"] == 15.0
        assert ok.json()["wallet"]["vault_balance"] == 5.0


# ---------------------------------------------------------------------------
# Gate 3: Open Position Guard on Leave-Balance (HTTP 409)
# ---------------------------------------------------------------------------


def test_leave_balance_returns_409_when_open_positions_exist(tmp_path: Path) -> None:
    """Calling POST /api/wallet/leave-balance while a trade is open returns HTTP 409."""
    db_url = f"sqlite:///{tmp_path / 'leave_409.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        # Demo data starts with OPEN trades on both BYBIT and FOREX
        resp_bybit = client.post(
            "/api/wallet/leave-balance",
            json={
                "market": "bybit",
                "target_active_balance": "20.00",
                "network": "INTERNAL",
            },
        )
        assert resp_bybit.status_code == 409
        assert "Close open positions on bybit first" in resp_bybit.json()["detail"]

        resp_all = client.post(
            "/api/wallet/leave-balance",
            json={
                "market": "all",
                "target_active_balance": "20.00",
                "network": "INTERNAL",
            },
        )
        assert resp_all.status_code == 409


# ---------------------------------------------------------------------------
# Gate 4: Auto-Sweep +$50 Trigger (base=20, threshold=50, network=BEP20 $0.15)
# ---------------------------------------------------------------------------


def test_auto_sweep_triggers_on_profitable_trade_close_bep20(tmp_path: Path) -> None:
    """base=20, threshold=50, BEP20 ($0.15): trade profit brings active to $74 -> $54 sweeps ($53.85 vault, $0.15 fee, active=$20, Today's PnL > 0)."""
    db_url = f"sqlite:///{tmp_path / 'auto_sweep.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        engine = create_db_engine(db_url)
        SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        with SessionCls() as db:
            db.query(Trade).delete()
            bybit_w = db.query(WalletAccount).filter(WalletAccount.market == "bybit").first()
            assert bybit_w is not None
            bybit_w.seed_capital = Decimal("20.000000")
            bybit_w.active_balance = Decimal("20.000000")
            bybit_w.vault_balance = Decimal("0.000000")
            bybit_w.equity_baseline_day = Decimal("20.000000")
            bybit_w.cumulative_net_cashflow_day = Decimal("0.000000")
            bybit_w.realism_enabled = False  # Exact zero-fee fill so gross == net == exact $54.00
            db.commit()

        # Configure Auto-Sweep via PATCH /api/wallet/settings
        patch_resp = client.patch(
            "/api/wallet/settings",
            json={
                "market": "bybit",
                "auto_sweep_enabled": True,
                "auto_sweep_base_capital": "20.00",
                "auto_sweep_threshold": "50.00",
                "auto_sweep_network": "BEP20",
            },
        )
        assert patch_resp.status_code == 200, patch_resp.text
        assert patch_resp.json()["auto_sweep_enabled"] is True
        assert patch_resp.json()["auto_sweep_network"] == "BEP20"

        # Open and close a winning trade that earns exact +$54.00 (active $20 -> $74 -> triggers auto-sweep!)
        with SessionCls() as db:
            created = app.state.broker.place_order(
                symbol="SOLUSDT",
                side="LONG",
                qty=1.0,
                stop_loss=50.0,
                take_profit=200.0,
                strategy_name="trend_rider",
                reason="Auto-sweep profit test",
                db=db,
                order_entry_type="LIMIT",
                requested_entry_price=Decimal("100.0"),
            )
            app.state.broker.close_position(
                trade_id=created["id"],
                reason="Take Profit reached",
                db=db,
                order_exit_type="TAKE_PROFIT",
                requested_exit_price=Decimal("154.0"),
            )

        # Verify wallet state & AUTO_SWEEP ledger row
        summary = client.get("/api/wallet/summary").json()
        bybit_sum = summary["markets"]["bybit"]
        assert Decimal(str(bybit_sum["active_balance"])) == Decimal("20.0")
        assert Decimal(str(bybit_sum["vault_balance"])) == Decimal("53.85")
        assert bybit_sum["daily_pnl_usd"] == 54.0  # Today's PnL stays positive!
        assert bybit_sum["daily_pnl_pct"] > 0.0
        assert bybit_sum["reconciliation"]["reconciled"] is True
        assert bybit_sum["reconciliation"]["drift"] == "0.000000"

        txs = client.get("/api/wallet/transactions?market=bybit&tx_type=AUTO_SWEEP").json()
        assert txs["total"] == 1
        sweep_tx = txs["items"][0]
        assert sweep_tx["gross_amount"] == 54.0
        assert sweep_tx["network_fee"] == 0.15
        assert sweep_tx["net_amount"] == 53.85
        assert sweep_tx["network"] == "BEP20"


# ---------------------------------------------------------------------------
# Gate 5: Auth Protection on ALL /api/wallet/* Endpoints
# ---------------------------------------------------------------------------


def test_wallet_endpoints_require_pin_auth_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All /api/wallet/* routes return 401 unauthenticated when DASHBOARD_PIN is set, and succeed after login."""
    monkeypatch.setattr(settings, "DASHBOARD_PIN", "135790")
    monkeypatch.setattr(settings, "SESSION_SECRET", "phase2-wallet-auth-secret")

    db_url = f"sqlite:///{tmp_path / 'wallet_auth.db'}"
    app = create_app(database_url=db_url)

    with TestClient(app) as client:
        assert client.get("/api/wallet/summary").status_code == 401
        assert client.get("/api/wallet/transactions").status_code == 401
        assert client.get("/api/wallet/rejections").status_code == 401
        assert (
            client.post(
                "/api/wallet/transfer",
                json={"market": "bybit", "tx_type": "EXTERNAL_TO_TRADING", "amount": "10.00"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/wallet/leave-balance",
                json={"market": "bybit", "target_active_balance": "20.00"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/api/wallet/reset-challenge",
                json={"market": "bybit", "target_active_balance": "20.00"},
            ).status_code
            == 401
        )
        assert (
            client.patch(
                "/api/wallet/settings",
                json={"market": "bybit", "realism_enabled": True},
            ).status_code
            == 401
        )

        # Authenticate via PIN login -> wallet summary now returns 200
        login = client.post("/api/auth/login", json={"pin": "135790"})
        assert login.status_code == 200
        assert client.get("/api/wallet/summary").status_code == 200


# ---------------------------------------------------------------------------
# Additional Coverage: Fee Schedule & Reset Challenge
# ---------------------------------------------------------------------------


def test_withdrawal_fee_schedule_and_reset_challenge(tmp_path: Path) -> None:
    """Verify every rail in the fee schedule (including LOCAL_BANK_NGN 0.50% FX spread & $0 deposit) and reset-challenge."""
    # INTERNAL: $0.00
    f_int = calculate_withdrawal_fees("bybit", "INTERNAL", Decimal("10.00"))
    assert f_int.network_fee == Decimal("0.000000") and f_int.net_amount == Decimal("10.000000")

    # APTOS: $0.002 fixed (Bybit)
    f_apt = calculate_withdrawal_fees("bybit", "APTOS", Decimal("10.00"))
    assert f_apt.network_fee == Decimal("0.002000") and f_apt.net_amount == Decimal("9.998000")

    # POLYGON: $0.07 fixed (Bybit)
    f_pol = calculate_withdrawal_fees("bybit", "POLYGON", Decimal("10.00"))
    assert f_pol.network_fee == Decimal("0.070000") and f_pol.net_amount == Decimal("9.930000")

    # BEP20: $0.15 fixed (both)
    f_bep = calculate_withdrawal_fees("forex", "BEP20", Decimal("10.00"))
    assert f_bep.network_fee == Decimal("0.150000") and f_bep.net_amount == Decimal("9.850000")

    # ERC20: $0.80 fixed (Bybit)
    f_erc = calculate_withdrawal_fees("bybit", "ERC20", Decimal("10.00"))
    assert f_erc.network_fee == Decimal("0.800000") and f_erc.net_amount == Decimal("9.200000")

    # TRC20: $1.30 fixed (both)
    f_trc = calculate_withdrawal_fees("forex", "TRC20", Decimal("10.00"))
    assert f_trc.network_fee == Decimal("1.300000") and f_trc.net_amount == Decimal("8.700000")

    # LOCAL_BANK_NGN: $0.50 fixed + 0.50% FX spread
    f_ngn = calculate_withdrawal_fees("bybit", "LOCAL_BANK_NGN", Decimal("100.00"))
    assert f_ngn.network_fee == Decimal("0.500000")
    assert f_ngn.fx_spread_fee == Decimal("0.500000")
    assert f_ngn.net_amount == Decimal("99.000000")

    # BANK_WIRE: $15.00 fixed, min $50 (Forex only)
    f_wire = calculate_withdrawal_fees("forex", "BANK_WIRE", Decimal("100.00"))
    assert f_wire.network_fee == Decimal("15.000000") and f_wire.net_amount == Decimal("85.000000")

    # Deposit carries $0.00 fee
    f_dep = calculate_withdrawal_fees("bybit", "TRC20", Decimal("100.00"), is_deposit=True)
    assert f_dep.network_fee == Decimal("0.000000") and f_dep.net_amount == Decimal("100.000000")

    # Clean-slate reset-challenge endpoint
    db_url = f"sqlite:///{tmp_path / 'reset_challenge.db'}"
    app = create_app(database_url=db_url)
    with TestClient(app) as client:
        engine = create_db_engine(db_url)
        SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        with SessionCls() as db:
            db.query(Trade).delete()
            db.commit()

        res = client.post(
            "/api/wallet/reset-challenge",
            json={"market": "bybit", "target_active_balance": "50.00"},
        )
        assert res.status_code == 200
        summary = client.get("/api/wallet/summary").json()
        assert summary["markets"]["bybit"]["active_balance"] == 50.0
        assert summary["markets"]["bybit"]["vault_balance"] == 0.0
        assert summary["markets"]["bybit"]["reconciliation"]["reconciled"] is True
