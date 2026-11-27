"""Prompt 3 Verification Gate: Realism Engine Wired into Execution, Sizing, Liquidation & Swaps."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from app.brokers.paper_sim import PaperSimBroker
from app.database import create_db_engine, init_db
from app.models import ExecutionRejection, Trade, WalletAccount
from app.services.realism_engine import (
    calculate_net_pnl,
    forex_lots_to_units,
    q6,
)


def _make_broker(tmp_path: Path, name: str) -> tuple[sessionmaker, PaperSimBroker]:
    db_url = f"sqlite:///{tmp_path / f'{name}.db'}"
    eng = create_db_engine(db_url)
    init_db(eng)
    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return SessionCls, PaperSimBroker(session_factory=SessionCls)


# ---------------------------------------------------------------------------
# 1. Exact Open -> Close Accounting (Zero Double-Counting of Fees)
# ---------------------------------------------------------------------------


def test_bybit_open_close_realism_exact_net_pnl_zero_double_counting(tmp_path: Path) -> None:
    """Opening & closing a Bybit trade with realism_enabled=True updates active_balance by exact net_pnl to 6dp."""
    SessionCls, broker = _make_broker(tmp_path, "bybit_exact_ledger")

    with SessionCls() as db:
        wallet = broker.get_wallet_account(db, "bybit")
        wallet.active_balance = Decimal("1000.000000")
        wallet.vault_balance = Decimal("0.000000")
        wallet.realism_enabled = True
        wallet.bybit_leverage = 5
        wallet.auto_sweep_enabled = False
        db.commit()

        balance_before = wallet.active_balance

        opened = broker.place_order(
            symbol="SOLUSDT",
            side="LONG",
            qty=2.0,
            stop_loss=140.0,
            take_profit=170.0,
            strategy_name="trend_rider",
            reason="Exact ledger test",
            db=db,
            order_entry_type="MARKET",
            requested_entry_price=Decimal("150.00"),
            volatility_pct=Decimal("0.0020"),
        )
        trade = db.query(Trade).filter(Trade.id == opened["id"]).first()
        assert trade is not None
        assert trade.filled_entry_price is not None
        assert trade.filled_entry_price > Decimal("150.00")
        assert trade.entry_fee > Decimal("0")
        assert trade.spread_cost > Decimal("0")
        assert trade.slippage_cost > Decimal("0")
        assert trade.initial_margin > Decimal("0")
        assert trade.maintenance_margin > Decimal("0")
        assert trade.liquidation_price is not None and trade.liquidation_price > Decimal("0")

        # Close via MARKET exit at $160.00
        broker.close_position(
            trade_id=trade.id,
            reason="Manual market exit",
            db=db,
            order_exit_type="MARKET",
            requested_exit_price=Decimal("160.00"),
        )
        db.refresh(trade)
        db.refresh(wallet)

        assert trade.status == "CLOSED"
        assert trade.order_exit_type == "MARKET"
        assert trade.filled_exit_price is not None
        assert trade.filled_exit_price < Decimal("160.00")
        assert trade.exit_fee > Decimal("0")

        expected_gross = q6((trade.filled_exit_price - trade.filled_entry_price) * Decimal("2.0"))
        expected_net = calculate_net_pnl(
            expected_gross,
            trade.entry_fee,
            trade.exit_fee,
            trade.funding_or_swap_fee,
        )
        assert trade.gross_pnl == expected_gross
        assert trade.net_pnl == expected_net

        # Exact zero double-counting invariant to 6 decimal places
        balance_delta = q6(wallet.active_balance - balance_before)
        assert balance_delta == trade.net_pnl
        assert balance_delta == expected_net


# ---------------------------------------------------------------------------
# 2. Strict Active-Balance Sizing ($20 active vs $20,000 vault) & Override Caps
# ---------------------------------------------------------------------------


def test_strict_active_balance_sizing_ignores_vault_and_enforces_override_and_margin(tmp_path: Path) -> None:
    """Sizing with active_balance=20.00 & vault_balance=20000.00 sizes strictly off $20 and rejects BTCUSDT @ $77k 5x."""
    SessionCls, broker = _make_broker(tmp_path, "strict_active_sizing")

    with SessionCls() as db:
        wallet = broker.get_wallet_account(db, "bybit")
        wallet.active_balance = Decimal("20.000000")
        wallet.vault_balance = Decimal("20000.000000")  # Must NEVER be used for sizing!
        wallet.realism_enabled = True
        wallet.bybit_leverage = 5
        wallet.allow_min_lot_override = False
        db.commit()

        # BTCUSDT @ $77,000 -> effective_min_qty = 0.002 (notional $154, 5x margin ~$30.89 > $20 active)
        res = broker.size_and_validate_order(
            db=db,
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("77000.00"),
            stop_loss=Decimal("76000.00"),
            max_risk_per_trade_pct=Decimal("1.0"),
            order_entry_type="MARKET",
            strategy="trend_rider",
            confluence_score=Decimal("82.50"),
        )
        db.commit()

        assert res.approved is False
        assert res.effective_min_qty == Decimal("0.002")
        assert res.available_free_margin == Decimal("20.000000")
        assert res.reason_code in {"BELOW_MIN_NOTIONAL", "INSUFFICIENT_MARGIN"}

        rejections = db.query(ExecutionRejection).all()
        assert len(rejections) == 1
        rej = rejections[0]
        assert rej.market == "bybit"
        assert rej.symbol == "BTCUSDT"
        assert rej.strategy == "trend_rider"
        assert rej.confluence_score == Decimal("82.50")
        assert rej.available_free_margin == Decimal("20.000000")
        assert rej.required_margin > Decimal("20.000000")

        # Direct place_order when free_margin < required_cash MUST also reject & log INSUFFICIENT_MARGIN
        with pytest.raises(ValueError, match="INSUFFICIENT_MARGIN"):
            broker.place_order(
                symbol="BTCUSDT",
                side="LONG",
                qty=0.002,
                stop_loss=76000.0,
                take_profit=80000.0,
                strategy_name="trend_rider",
                reason="Should fail margin guard",
                db=db,
                requested_entry_price=Decimal("77000.00"),
            )
        assert db.query(Trade).count() == 0
        latest_rej = db.query(ExecutionRejection).order_by(ExecutionRejection.id.desc()).first()
        assert latest_rej is not None
        assert latest_rej.reason_code == "INSUFFICIENT_MARGIN"
        assert latest_rej.available_free_margin == Decimal("20.000000")
        assert latest_rej.required_margin > Decimal("20.000000")

        # Test allow_min_lot_override=True with 15% active_balance cap on SOLUSDT ($20 active -> 15% cap = $3.00)
        wallet.allow_min_lot_override = True
        db.commit()

        # SOLUSDT effective_min_qty=0.1 @ $150 -> notional $15, 5x margin ~$3.01 <= $20 free_margin.
        # Case A: SL at $130 ($20/unit move -> 0.1 * ~$20 = ~$2.02 <= $3.00 cap) -> APPROVED!
        res_ok = broker.size_and_validate_order(
            db=db,
            symbol="SOLUSDT",
            side="LONG",
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("130.00"),
            max_risk_per_trade_pct=Decimal("1.0"),
        )
        assert res_ok.approved is True
        assert res_ok.qty == Decimal("0.1")
        assert res_ok.estimated_risk_usd <= Decimal("3.000000")

        # Case B: SL at $110 ($40/unit move -> 0.1 * ~$40 = ~$4.02 > $3.00 15% cap) -> RISK_CAP_EXCEEDED!
        res_cap = broker.size_and_validate_order(
            db=db,
            symbol="SOLUSDT",
            side="LONG",
            entry_price=Decimal("150.00"),
            stop_loss=Decimal("110.00"),
            max_risk_per_trade_pct=Decimal("1.0"),
        )
        db.commit()
        assert res_cap.approved is False
        assert res_cap.reason_code == "RISK_CAP_EXCEEDED"
        cap_rej = db.query(ExecutionRejection).order_by(ExecutionRejection.id.desc()).first()
        assert cap_rej is not None
        assert cap_rej.reason_code == "RISK_CAP_EXCEEDED"


# ---------------------------------------------------------------------------
# 3. Bybit Isolated Liquidation + Negative Balance Protection
# ---------------------------------------------------------------------------


def test_bybit_liquidation_price_breach_triggers_clean_liquidation(tmp_path: Path) -> None:
    """Bybit price hitting liquidation_price force-closes with order_exit_type='LIQUIDATION' and active_balance >= 0."""
    SessionCls, broker = _make_broker(tmp_path, "bybit_liquidation")

    with SessionCls() as db:
        wallet = broker.get_wallet_account(db, "bybit")
        wallet.active_balance = Decimal("200.000000")
        wallet.realism_enabled = True
        wallet.bybit_leverage = 5
        db.commit()

        opened = broker.place_order(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.01,
            stop_loss=40000.0,  # Far below liquidation price so liquidation triggers first
            take_profit=120000.0,
            strategy_name="trend_rider",
            reason="Liquidation test",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("80000.0"),
        )
        trade = db.query(Trade).filter(Trade.id == opened["id"]).first()
        assert trade is not None
        assert trade.liquidation_price is not None

        # Price hits exact liquidation price
        trade.current_price = float(trade.liquidation_price)
        db.commit()

        result = broker.apply_periodic_fees_and_risk_checks(db=db)
        db.commit()
        db.refresh(trade)
        db.refresh(wallet)

        assert result["liquidated_ids"] == [trade.id]
        assert trade.status == "CLOSED"
        assert trade.order_exit_type == "LIQUIDATION"
        assert trade.exit_fee == Decimal("0.000000")
        # Negative balance protection guarantees active_balance >= 0
        assert wallet.active_balance >= Decimal("0.000000")


# ---------------------------------------------------------------------------
# 4. MT5 Forex Stop-Out (<= 50%) Closes Largest Loser First
# ---------------------------------------------------------------------------


def test_forex_stop_out_closes_largest_loser_first(tmp_path: Path) -> None:
    """Forex margin_level_pct <= 50% force-closes the largest unrealized loser with order_exit_type='STOP_OUT'."""
    SessionCls, broker = _make_broker(tmp_path, "forex_stop_out")

    with SessionCls() as db:
        fx_wallet = broker.get_wallet_account(db, "forex")
        fx_wallet.active_balance = Decimal("120.000000")
        fx_wallet.realism_enabled = True
        fx_wallet.forex_leverage = 500
        fx_wallet.account_profile = "mt5_raw_ecn"
        db.commit()

        # Smaller loser: EUR_USD 0.10 lot (10,000 units), margin ~$21.69
        small_t = broker.place_order(
            symbol="EUR_USD",
            side="LONG",
            qty=float(forex_lots_to_units("EUR_USD", Decimal("0.10"))),
            stop_loss=0.90000,
            take_profit=1.30000,
            strategy_name="mean_reversion",
            reason="Small loser",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("1.08450"),
        )
        # Biggest loser: XAU_USD 0.10 lot (10 oz), margin ~$53.00
        big_t = broker.place_order(
            symbol="XAU_USD",
            side="LONG",
            qty=float(forex_lots_to_units("XAU_USD", Decimal("0.10"))),
            stop_loss=2000.00,
            take_profit=3000.00,
            strategy_name="trend_rider",
            reason="Biggest loser",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("2650.00"),
        )

        row_small = db.query(Trade).filter(Trade.id == small_t["id"]).first()
        row_big = db.query(Trade).filter(Trade.id == big_t["id"]).first()
        assert row_small is not None and row_big is not None

        # Small loser -$5, big loser -$85 -> equity ~$29.30, used margin ~$74.69 -> margin_level ~39.2% <= 50%
        row_small.current_price = 1.08400
        row_big.current_price = 2641.50
        db.commit()

        result = broker.apply_periodic_fees_and_risk_checks(db=db)
        db.commit()
        db.refresh(row_small)
        db.refresh(row_big)

        assert result["stopped_out_ids"] == [row_big.id]
        assert row_big.status == "CLOSED"
        assert row_big.order_exit_type == "STOP_OUT"
        assert row_small.status == "OPEN"


# ---------------------------------------------------------------------------
# 5. Idempotent 8h Bybit Funding & Wednesday 3x NY Swap Across 5 Consecutive Ticks
# ---------------------------------------------------------------------------


def test_five_consecutive_ticks_charge_bybit_funding_and_wednesday_3x_swap_once(tmp_path: Path) -> None:
    """5 consecutive ticks past 08:00 UTC charge Bybit funding once; 5 ticks past Wed 17:00 NY charge 3x swap once."""
    SessionCls, broker = _make_broker(tmp_path, "idempotent_5_ticks")

    with SessionCls() as db:
        bybit_w = broker.get_wallet_account(db, "bybit")
        forex_w = broker.get_wallet_account(db, "forex")
        bybit_w.active_balance = Decimal("5000.000000")
        bybit_w.realism_enabled = True
        bybit_w.auto_sweep_enabled = False
        forex_w.active_balance = Decimal("5000.000000")
        forex_w.realism_enabled = True
        forex_w.auto_sweep_enabled = False
        db.commit()

        bybit_start_bal = bybit_w.active_balance
        forex_start_bal = forex_w.active_balance

        # --- Part A: Bybit 08:00 UTC boundary across 5 consecutive ticks ---
        bybit_opened_at = datetime(2026, 6, 3, 7, 30, tzinfo=timezone.utc)
        b_order = broker.place_order(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.1,
            stop_loss=10000.0,
            take_profit=200000.0,
            strategy_name="trend_rider",
            reason="8h funding idempotency",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("50000.0"),
        )
        b_trade = db.query(Trade).filter(Trade.id == b_order["id"]).first()
        assert b_trade is not None
        b_trade.opened_at = bybit_opened_at
        b_trade.current_price = 50000.0
        db.commit()

        # Run 5 consecutive ticks past 08:00 UTC (08:01, 08:05, 08:10, 08:15, 08:20 UTC)
        for i, minute in enumerate((1, 5, 10, 15, 20)):
            tick_ts = datetime(2026, 6, 3, 8, minute, tzinfo=timezone.utc)
            b_trade.current_price = 50000.0
            broker.apply_periodic_fees_and_risk_checks(db=db, now_utc=tick_ts)
            db.commit()
            db.refresh(b_trade)
            db.refresh(bybit_w)
            # Exact 1 period funding: 0.1 * 50000 * 0.0001 = $0.500000
            assert b_trade.funding_or_swap_fee == Decimal("0.500000"), f"Mismatch on tick {i + 1}"
            assert b_trade.last_funding_applied_at.replace(tzinfo=timezone.utc) == datetime(
                2026, 6, 3, 8, 0, tzinfo=timezone.utc
            )
            assert bybit_w.active_balance == Decimal("4999.500000"), f"Balance drifted on tick {i + 1}"

        # Close Bybit trade and verify zero double-counting of funding_or_swap_fee
        broker.close_position(
            trade_id=b_trade.id,
            reason="Close after funding",
            db=db,
            order_exit_type="LIMIT",
            requested_exit_price=Decimal("51000.0"),
        )
        db.refresh(b_trade)
        db.refresh(bybit_w)
        assert q6(bybit_w.active_balance - bybit_start_bal) == b_trade.net_pnl

        # --- Part B: Forex Wednesday 17:00 America/New_York (21:00 UTC EDT) 3x swap across 5 ticks ---
        # 2026-06-03 is a Wednesday (EDT = UTC-4 -> 17:00 NY = 21:00 UTC)
        fx_opened_at = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)
        f_order = broker.place_order(
            symbol="EUR_USD",
            side="LONG",
            qty=float(forex_lots_to_units("EUR_USD", Decimal("0.10"))),
            stop_loss=0.80000,
            take_profit=1.40000,
            strategy_name="mean_reversion",
            reason="Wed 3x swap idempotency",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("1.08450"),
        )
        f_trade = db.query(Trade).filter(Trade.id == f_order["id"]).first()
        assert f_trade is not None
        f_trade.opened_at = fx_opened_at
        f_trade.current_price = 1.08450
        db.commit()

        # Run 5 consecutive ticks past Wed 17:00 NY (21:01, 21:05, 21:10, 21:15, 21:20 UTC)
        for i, minute in enumerate((1, 5, 10, 15, 20)):
            tick_ts = datetime(2026, 6, 3, 21, minute, tzinfo=timezone.utc)
            f_trade.current_price = 1.08450
            broker.apply_periodic_fees_and_risk_checks(db=db, now_utc=tick_ts)
            db.commit()
            db.refresh(f_trade)
            db.refresh(forex_w)
            # EUR_USD swap_long = -6.50 * 0.10 lot * 3x Wed = $1.950000
            assert f_trade.funding_or_swap_fee == Decimal("1.950000"), f"Mismatch on tick {i + 1}"
            assert f_trade.last_swap_applied_date == date(2026, 6, 3)
            assert forex_w.active_balance == Decimal("4998.050000"), f"Balance drifted on tick {i + 1}"

        # Close Forex trade and verify zero double-counting of swap fee
        broker.close_position(
            trade_id=f_trade.id,
            reason="Close after swap",
            db=db,
            order_exit_type="LIMIT",
            requested_exit_price=Decimal("1.08650"),
        )
        db.refresh(f_trade)
        db.refresh(forex_w)
        assert q6(forex_w.active_balance - forex_start_bal) == f_trade.net_pnl
