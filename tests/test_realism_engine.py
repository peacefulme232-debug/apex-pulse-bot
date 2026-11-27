"""Phase 1 Verification Gate: Exact Decimal Realism Engine, Execution Wiring, & Startup Migrations."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from app.brokers.paper_sim import PaperSimBroker
from app.database import create_db_engine, init_db
from app.models import ExecutionRejection, Trade, WalletAccount, WalletTransaction
from app.services.realism_engine import (
    BYBIT_MAKER_FEE_RATE,
    BYBIT_TAKER_FEE_RATE,
    BybitContractSpecLoader,
    ForexPositionSnapshot,
    bybit_calculate_funding,
    bybit_calculate_margin_and_liquidation,
    bybit_effective_min_qty,
    bybit_fee_rate,
    bybit_simulate_fill,
    bybit_size_and_validate_order,
    calculate_net_pnl,
    estimate_round_turn_friction,
    forex_calculate_rollover_swap,
    forex_commission_per_side,
    forex_evaluate_margin_status,
    forex_gross_pnl,
    forex_lots_to_units,
    forex_required_margin,
    forex_round_turn_commission,
    get_bybit_contract_spec,
    select_stop_out_targets,
)


# ---------------------------------------------------------------------------
# Gate 1: Bybit UTA Isolated Margin, Exact Liquidation Prices, & Negative Balance Protection
# ---------------------------------------------------------------------------


def test_bybit_uta_long_and_short_liquidation_prices() -> None:
    """Verify exact official Bybit UTA isolated Long LP (36380.25) & Short LP (10956.18)."""
    long_metrics = bybit_calculate_margin_and_liquidation(
        side="LONG",
        entry_price=Decimal("40000"),
        qty=Decimal("1"),
        leverage=50,
        extra_margin=Decimal("3000"),
        mmr=Decimal("0.005"),
        mm_deduction=Decimal("0"),
        taker_rate=Decimal("0.00055"),
    )
    assert isinstance(long_metrics.liquidation_price, Decimal)
    assert abs(long_metrics.liquidation_price - Decimal("36380.25")) <= Decimal("0.01")
    # Fee to close Long = 40000 * (1 - 1/50) * 0.00055 = 21.56
    assert long_metrics.fee_to_close == Decimal("21.560000")
    # Initial margin = 40000/50 + 21.56 = 821.56
    assert long_metrics.initial_margin == Decimal("821.560000")
    # Entry fee (market taker) = 40000 * 0.00055 = 22.00; required_cash = 843.56
    assert long_metrics.entry_fee == Decimal("22.000000")
    assert long_metrics.required_cash == Decimal("843.560000")

    short_metrics = bybit_calculate_margin_and_liquidation(
        side="SHORT",
        entry_price=Decimal("10000"),
        qty=Decimal("1"),
        leverage=10,
        extra_margin=Decimal("0"),
        mmr=Decimal("0.004"),
        mm_deduction=Decimal("0"),
        taker_rate=Decimal("0.00055"),
    )
    assert isinstance(short_metrics.liquidation_price, Decimal)
    assert abs(short_metrics.liquidation_price - Decimal("10956.18")) <= Decimal("0.01")
    # Fee to close Short = 10000 * (1 + 1/10) * 0.00055 = 6.05
    assert short_metrics.fee_to_close == Decimal("6.050000")


def test_bybit_fees_spread_slippage_and_liquidation_negative_balance_protection(tmp_path: Path) -> None:
    """Verify maker/taker rates, exact limit fill, liquidation at bankruptcy price with 0% extra fee & balance floor >= 0."""
    assert bybit_fee_rate("LIMIT") == BYBIT_MAKER_FEE_RATE == Decimal("0.00020")
    assert bybit_fee_rate("TAKE_PROFIT") == BYBIT_MAKER_FEE_RATE
    assert bybit_fee_rate("MARKET") == BYBIT_TAKER_FEE_RATE == Decimal("0.00055")
    assert bybit_fee_rate("STOP_LOSS") == BYBIT_TAKER_FEE_RATE
    assert bybit_fee_rate("LIQUIDATION") == Decimal("0")

    # Limit entry fills exact with 0 spread and 0 slippage
    limit_fill = bybit_simulate_fill(
        symbol="BTCUSDT",
        side="LONG",
        is_entry=True,
        mid_price=Decimal("50000.0"),
        order_type="LIMIT",
        qty=Decimal("0.01"),
    )
    assert limit_fill.filled_price == Decimal("50000.00000000")
    assert limit_fill.spread_cost == Decimal("0")
    assert limit_fill.slippage_cost == Decimal("0")
    assert limit_fill.fee == Decimal("0.100000")  # 500 * 0.00020

    # Market long buys above mid; market short sells below mid
    mkt_long = bybit_simulate_fill(
        symbol="BTCUSDT",
        side="LONG",
        is_entry=True,
        mid_price=Decimal("50000.0"),
        order_type="MARKET",
        qty=Decimal("0.01"),
    )
    assert mkt_long.filled_price > Decimal("50000.0")
    assert mkt_long.spread_cost > Decimal("0")
    assert mkt_long.slippage_cost > Decimal("0")

    # DB-level liquidation trigger + negative balance protection
    db_url = f"sqlite:///{tmp_path / 'bybit_liq.db'}"
    eng = create_db_engine(db_url)
    init_db(eng)
    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    broker = PaperSimBroker(session_factory=SessionCls)

    with SessionCls() as db:
        wallet = broker.get_wallet_account(db, "bybit")
        wallet.active_balance = Decimal("200.000000")
        wallet.bybit_leverage = 5
        db.commit()

        created = broker.place_order(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.01,
            stop_loss=50000.0,  # Far below LP so liquidation triggers first
            take_profit=120000.0,
            strategy_name="trend_rider",
            reason="Test liquidation",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("80000.0"),
        )
        trade = db.query(Trade).filter(Trade.id == created["id"]).first()
        assert trade is not None
        assert trade.liquidation_price is not None
        # Move mark price below liquidation price (e.g. flash crash to 55000)
        trade.current_price = 55000.0
        db.commit()

        check = broker.apply_periodic_fees_and_risk_checks(db=db)
        db.commit()
        db.refresh(trade)
        db.refresh(wallet)

        assert trade.id in check["liquidated_ids"]
        assert trade.status == "CLOSED"
        assert trade.order_exit_type == "LIQUIDATION"
        assert trade.exit_fee == Decimal("0.000000")  # 0% extra penalty fee
        assert trade.net_pnl == calculate_net_pnl(
            trade.gross_pnl, trade.entry_fee, trade.exit_fee, trade.funding_or_swap_fee
        )
        # Negative balance protection: active_balance never goes below 0
        assert wallet.active_balance >= Decimal("0.000000")


# ---------------------------------------------------------------------------
# Gate 2: Bybit BTCUSDT $77,000 effective_min_qty=0.002 & $20 @ 5x Rejection + Vault Isolation
# ---------------------------------------------------------------------------


def test_bybit_btcusdt_77000_min_qty_and_small_balance_rejection(tmp_path: Path) -> None:
    """Verify BTCUSDT @ $77,000 -> effective_min_qty=0.002, logs BELOW_MIN_NOTIONAL/INSUFFICIENT_MARGIN on $20 @ 5x, and ignores vault_balance."""
    eff_min = bybit_effective_min_qty("BTCUSDT", Decimal("77000"))
    assert eff_min == Decimal("0.002")

    # Unit-level check on $20 balance at 5x leverage
    sizing = bybit_size_and_validate_order(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=Decimal("77000"),
        stop_loss=Decimal("76000"),
        active_balance=Decimal("20"),
        used_margin=Decimal("0"),
        max_risk_per_trade_pct=Decimal("1.0"),
        bybit_leverage=5,
        allow_min_lot_override=False,
    )
    assert sizing.approved is False
    assert sizing.effective_min_qty == Decimal("0.002")
    assert sizing.reason_code in {"BELOW_MIN_NOTIONAL", "INSUFFICIENT_MARGIN"}

    # DB integration check: vault_balance = $50,000 MUST NOT allow sizing on $20 active_balance
    db_url = f"sqlite:///{tmp_path / 'bybit_reject.db'}"
    eng = create_db_engine(db_url)
    init_db(eng)
    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    broker = PaperSimBroker(session_factory=SessionCls)

    with SessionCls() as db:
        wallet = broker.get_wallet_account(db, "bybit")
        wallet.active_balance = Decimal("20.000000")
        wallet.vault_balance = Decimal("50000.000000")  # Must be strictly ignored!
        wallet.bybit_leverage = 5
        wallet.allow_min_lot_override = False
        db.commit()

        res = broker.size_and_validate_order(
            db=db,
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("77000"),
            stop_loss=Decimal("76000"),
            max_risk_per_trade_pct=Decimal("1.0"),
        )
        db.commit()

        assert res.approved is False
        assert res.effective_min_qty == Decimal("0.002")
        assert res.reason_code in {"BELOW_MIN_NOTIONAL", "INSUFFICIENT_MARGIN"}

        rejections = db.query(ExecutionRejection).all()
        assert len(rejections) == 1
        rej = rejections[0]
        assert rej.market == "bybit"
        assert rej.symbol == "BTCUSDT"
        assert rej.reason_code in {"BELOW_MIN_NOTIONAL", "INSUFFICIENT_MARGIN"}
        assert rej.required_min_qty == Decimal("0.002000")
        assert rej.available_free_margin == Decimal("20.000000")

        # Also test BELOW_MIN_NOTIONAL explicitly when free margin is sufficient ($500) but 1% risk ($5) sizes < 0.002
        wallet.active_balance = Decimal("500.000000")
        db.commit()
        res_notional = broker.size_and_validate_order(
            db=db,
            symbol="BTCUSDT",
            side="LONG",
            entry_price=Decimal("77000"),
            stop_loss=Decimal("70000"),  # Wide stop -> raw qty ~ 0.0007 < 0.002
            max_risk_per_trade_pct=Decimal("1.0"),
        )
        db.commit()
        assert res_notional.approved is False
        assert res_notional.reason_code == "BELOW_MIN_NOTIONAL"


def test_bybit_live_spec_parser_and_fallbacks() -> None:
    """Verify Bybit V5 linear instruments-info parser & static fallbacks for ETH, SOL, and generic altcoins."""
    eth = get_bybit_contract_spec("ETHUSDT")
    assert eth.min_qty == Decimal("0.01") and eth.min_notional == Decimal("20")
    sol = get_bybit_contract_spec("SOLUSDT")
    assert sol.min_qty == Decimal("0.1") and sol.min_notional == Decimal("5")
    alt = get_bybit_contract_spec("DOGEUSDT")
    assert alt.min_qty == Decimal("0.1") and alt.max_lev == 25 and alt.spread_bps == Decimal("3.0")

    parsed = BybitContractSpecLoader.parse_instruments_payload(
        {
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "fundingInterval": 480,
                        "lotSizeFilter": {
                            "minOrderQty": "0.001",
                            "qtyStep": "0.001",
                            "minNotionalValue": "100",
                        },
                        "priceFilter": {"tickSize": "0.1"},
                        "leverageFilter": {"maxLeverage": "100.00"},
                    }
                ]
            }
        }
    )
    assert "BTCUSDT" in parsed
    assert parsed["BTCUSDT"].funding_interval_hours == 8
    assert parsed["BTCUSDT"].min_notional == Decimal("100")


# ---------------------------------------------------------------------------
# Gate 3: Forex Commissions, XAU_USD $1/$1 Pip Value, & <= 50% Stop-Out Largest Loser First
# ---------------------------------------------------------------------------


def test_forex_commissions_xau_pnl_and_stop_out_largest_loser_first(tmp_path: Path) -> None:
    """Verify 0.01 EUR_USD=$0.07 RT commission, XAU_USD 0.01 lot=$1/$1 move, and <=50% stop-out closes largest loser first."""
    # EUR_USD 0.01 lot Raw ECN commission = $0.035/side, $0.07 round-turn
    assert forex_commission_per_side(Decimal("0.01"), "mt5_raw_ecn") == Decimal("0.035000")
    assert forex_round_turn_commission(Decimal("0.01"), "mt5_raw_ecn") == Decimal("0.070000")
    # Standard STP = $0.00
    assert forex_round_turn_commission(Decimal("0.01"), "mt5_standard_stp") == Decimal("0.000000")

    # XAU_USD 0.01 lot = 1 oz -> $1.00 price move = $1.00 PnL
    xau_pnl = forex_gross_pnl(
        symbol="XAU_USD",
        side="LONG",
        lots=Decimal("0.01"),
        entry_price=Decimal("2650.00"),
        exit_price=Decimal("2651.00"),
    )
    assert xau_pnl == Decimal("1.000000")

    # Unit-level Forex Stop-Out check: 3 open positions, margin level <= 50% -> closes largest loser first
    pos_small_win = ForexPositionSnapshot(
        trade_id=101,
        symbol="EUR_USD",
        side="LONG",
        lots=Decimal("0.10"),
        entry_price=Decimal("1.0800"),
        current_price=Decimal("1.0810"),
        unrealized_pnl=Decimal("10.00"),
        used_margin=Decimal("21.60"),
    )
    pos_med_loss = ForexPositionSnapshot(
        trade_id=102,
        symbol="GBP_USD",
        side="LONG",
        lots=Decimal("0.10"),
        entry_price=Decimal("1.2600"),
        current_price=Decimal("1.2550"),
        unrealized_pnl=Decimal("-50.00"),
        used_margin=Decimal("25.20"),
    )
    pos_big_loss = ForexPositionSnapshot(
        trade_id=103,
        symbol="XAU_USD",
        side="LONG",
        lots=Decimal("0.10"),
        entry_price=Decimal("2650.00"),
        current_price=Decimal("2640.00"),
        unrealized_pnl=Decimal("-100.00"),
        used_margin=Decimal("53.00"),
    )
    # Total used margin = 99.80; equity = 40.00 -> margin_level = 40.08% <= 50%
    # Closing pos_big_loss (margin 53.00) leaves used_margin = 46.80 -> projected margin_level = 85.47% > 50%
    status = forex_evaluate_margin_status(
        forex_equity=Decimal("40.00"),
        open_positions=[pos_small_win, pos_med_loss, pos_big_loss],
        forex_leverage=500,
    )
    assert status.margin_call is True
    assert status.stop_out is True
    assert status.positions_to_close == (103,)

    # DB integration test for Forex STOP_OUT closing largest losing position first
    db_url = f"sqlite:///{tmp_path / 'forex_stopout.db'}"
    eng = create_db_engine(db_url)
    init_db(eng)
    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    broker = PaperSimBroker(session_factory=SessionCls)

    with SessionCls() as db:
        fx_wallet = broker.get_wallet_account(db, "forex")
        fx_wallet.active_balance = Decimal("120.000000")
        fx_wallet.forex_leverage = 500
        db.commit()

        # Position 1: EUR_USD 0.10 lot (10,000 units), margin ~ $21.69
        t1 = broker.place_order(
            symbol="EUR_USD",
            side="LONG",
            qty=float(forex_lots_to_units("EUR_USD", Decimal("0.10"))),
            stop_loss=0.9000,
            take_profit=1.3000,
            strategy_name="trend_rider",
            reason="FX small loser",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("1.08450"),
        )
        # Position 2: XAU_USD 0.10 lot (10 oz), margin ~ $53.00
        t2 = broker.place_order(
            symbol="XAU_USD",
            side="LONG",
            qty=float(forex_lots_to_units("XAU_USD", Decimal("0.10"))),
            stop_loss=2000.0,
            take_profit=3000.0,
            strategy_name="trend_rider",
            reason="FX big loser",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("2650.00"),
        )

        row1 = db.query(Trade).filter(Trade.id == t1["id"]).first()
        row2 = db.query(Trade).filter(Trade.id == t2["id"]).first()
        assert row1 is not None and row2 is not None

        # Make row1 lose -$5 and row2 lose -$85 -> equity = 120 - 90 - fees (~$0.70) ≈ $29.30
        # Total used margin ≈ $74.69 -> margin_level ≈ 39.2% <= 50%
        # Closing row2 (largest loser, margin $53.00) leaves used_margin $21.69 -> level > 100% > 50%
        row1.current_price = 1.08400
        row2.current_price = 2641.50
        db.commit()

        res = broker.apply_periodic_fees_and_risk_checks(db=db)
        db.commit()
        db.refresh(row1)
        db.refresh(row2)

        assert res["stopped_out_ids"] == [row2.id]
        assert row2.status == "CLOSED"
        assert row2.order_exit_type == "STOP_OUT"
        assert row1.status == "OPEN"


# ---------------------------------------------------------------------------
# Gate 4: Strictly Idempotent 8h Bybit Funding & Wednesday 3x NY Swap
# ---------------------------------------------------------------------------


def test_bybit_8h_funding_and_wednesday_3x_ny_swap_idempotency(tmp_path: Path) -> None:
    """Verify 8h Bybit funding & Wednesday 3x NY 17:00 swap apply exact amounts and never double-charge."""
    # --- Bybit 8h Funding (00:00, 08:00, 16:00 UTC) ---
    opened = datetime(2026, 6, 3, 7, 30, tzinfo=timezone.utc)
    after_08 = datetime(2026, 6, 3, 8, 5, tzinfo=timezone.utc)

    first = bybit_calculate_funding(
        side="LONG",
        qty=Decimal("1"),
        mark_price=Decimal("50000"),
        opened_at=opened,
        now_utc=after_08,
        last_funding_applied_at=None,
    )
    assert first.periods_applied == 1
    assert first.fee_amount == Decimal("5.000000")  # 1 * 50000 * 0.0001
    assert first.new_last_funding_applied_at == datetime(2026, 6, 3, 8, 0, tzinfo=timezone.utc)

    # Repeated tick at 08:05 and 15:59 UTC -> 0 periods, 0 fee
    repeat = bybit_calculate_funding(
        side="LONG",
        qty=Decimal("1"),
        mark_price=Decimal("50000"),
        opened_at=opened,
        now_utc=datetime(2026, 6, 3, 15, 59, tzinfo=timezone.utc),
        last_funding_applied_at=first.new_last_funding_applied_at,
    )
    assert repeat.periods_applied == 0
    assert repeat.fee_amount == Decimal("0")

    # Short receives funding (negative fee_amount = credit)
    short_funding = bybit_calculate_funding(
        side="SHORT",
        qty=Decimal("1"),
        mark_price=Decimal("50000"),
        opened_at=opened,
        now_utc=after_08,
        last_funding_applied_at=None,
    )
    assert short_funding.periods_applied == 1
    assert short_funding.fee_amount == Decimal("-5.000000")

    # --- Forex Wednesday 3x Rollover Swap (17:00 America/New_York) ---
    # 2026-06-03 is a Wednesday (EDT = UTC-4, so 17:00 NY = 21:00 UTC)
    wed_open_utc = datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc)  # 10:00 AM NY Wed
    wed_before_rollover = datetime(2026, 6, 3, 20, 59, tzinfo=timezone.utc)  # 16:59 NY Wed
    wed_after_rollover = datetime(2026, 6, 3, 21, 5, tzinfo=timezone.utc)  # 17:05 NY Wed

    pre = forex_calculate_rollover_swap(
        symbol="EUR_USD",
        side="LONG",
        lots=Decimal("1.00"),
        opened_at=wed_open_utc,
        now_utc=wed_before_rollover,
        last_swap_applied_date=None,
    )
    assert pre.rollover_days_applied == 0
    assert pre.fee_amount == Decimal("0")

    wed_swap = forex_calculate_rollover_swap(
        symbol="EUR_USD",
        side="LONG",
        lots=Decimal("1.00"),
        opened_at=wed_open_utc,
        now_utc=wed_after_rollover,
        last_swap_applied_date=None,
    )
    assert wed_swap.rollover_days_applied == 1
    assert wed_swap.total_multiplier == 3
    # EUR_USD swap_long = -6.50 * 1.00 lot * 3x = -$19.50 -> fee_amount = +19.50
    assert wed_swap.fee_amount == Decimal("19.500000")
    assert wed_swap.new_last_swap_applied_date == date(2026, 6, 3)

    # Repeat tick on same evening -> strictly idempotent (0 fee)
    wed_repeat = forex_calculate_rollover_swap(
        symbol="EUR_USD",
        side="LONG",
        lots=Decimal("1.00"),
        opened_at=wed_open_utc,
        now_utc=datetime(2026, 6, 3, 23, 30, tzinfo=timezone.utc),
        last_swap_applied_date=wed_swap.new_last_swap_applied_date,
    )
    assert wed_repeat.rollover_days_applied == 0
    assert wed_repeat.fee_amount == Decimal("0")

    # DB-persisted idempotency test across repeated ticks in PaperSimBroker
    db_url = f"sqlite:///{tmp_path / 'idempotent_fees.db'}"
    eng = create_db_engine(db_url)
    init_db(eng)
    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    broker = PaperSimBroker(session_factory=SessionCls)

    with SessionCls() as db:
        bybit_t = broker.place_order(
            symbol="BTCUSDT",
            side="LONG",
            qty=0.1,
            stop_loss=10000.0,
            take_profit=200000.0,
            strategy_name="trend_rider",
            reason="Funding test",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("50000.0"),
        )
        fx_t = broker.place_order(
            symbol="EUR_USD",
            side="LONG",
            qty=float(forex_lots_to_units("EUR_USD", Decimal("0.10"))),
            stop_loss=0.9000,
            take_profit=1.3000,
            strategy_name="trend_rider",
            reason="Swap test",
            db=db,
            order_entry_type="LIMIT",
            requested_entry_price=Decimal("1.08450"),
        )
        b_row = db.query(Trade).filter(Trade.id == bybit_t["id"]).first()
        f_row = db.query(Trade).filter(Trade.id == fx_t["id"]).first()
        assert b_row is not None and f_row is not None
        b_row.opened_at = opened
        b_row.current_price = 50000.0
        f_row.opened_at = wed_open_utc
        f_row.current_price = 1.08450
        db.commit()

        # Tick 1 at 2026-06-03 21:05 UTC (crosses 08:00 & 16:00 UTC for Bybit, and 17:00 NY Wed 3x for Forex)
        broker.apply_periodic_fees_and_risk_checks(db=db, now_utc=wed_after_rollover)
        db.commit()
        db.refresh(b_row)
        db.refresh(f_row)

        bybit_fee_1 = b_row.funding_or_swap_fee
        forex_fee_1 = f_row.funding_or_swap_fee
        # 2 periods * 0.1 * 50000 * 0.0001 = 1.000000
        assert bybit_fee_1 == Decimal("1.000000")
        # 0.10 lot * 6.50 * 3x = 1.950000
        assert forex_fee_1 == Decimal("1.950000")

        # Tick 2 & 3 at 21:10 and 21:15 UTC -> ZERO additional charges
        broker.apply_periodic_fees_and_risk_checks(
            db=db, now_utc=datetime(2026, 6, 3, 21, 10, tzinfo=timezone.utc)
        )
        broker.apply_periodic_fees_and_risk_checks(
            db=db, now_utc=datetime(2026, 6, 3, 21, 15, tzinfo=timezone.utc)
        )
        db.commit()
        db.refresh(b_row)
        db.refresh(f_row)

        assert b_row.funding_or_swap_fee == bybit_fee_1
        assert f_row.funding_or_swap_fee == forex_fee_1


# ---------------------------------------------------------------------------
# Gate 5: Idempotent Startup Migration on Empty DB & Legacy Existing DB
# ---------------------------------------------------------------------------


def test_startup_migration_empty_db_and_existing_legacy_db(tmp_path: Path) -> None:
    """Verify init_db() runs cleanly and idempotently on both empty DB and existing pre-Phase-1 DB."""
    # 1. Empty DB
    empty_path = tmp_path / "empty_phase1.db"
    empty_engine = create_db_engine(f"sqlite:///{empty_path}")
    init_db(empty_engine)
    init_db(empty_engine)  # Idempotent second run

    insp = inspect(empty_engine)
    tables = set(insp.get_table_names())
    assert {"wallet_accounts", "wallet_transactions", "execution_rejections", "trades"}.issubset(tables)

    trade_cols = {c["name"] for c in insp.get_columns("trades")}
    expected_new_cols = {
        "order_entry_type",
        "order_exit_type",
        "requested_entry_price",
        "filled_entry_price",
        "requested_exit_price",
        "filled_exit_price",
        "initial_margin",
        "maintenance_margin",
        "liquidation_price",
        "gross_pnl",
        "entry_fee",
        "exit_fee",
        "spread_cost",
        "slippage_cost",
        "funding_or_swap_fee",
        "net_pnl",
        "last_funding_applied_at",
        "last_swap_applied_date",
    }
    assert expected_new_cols.issubset(trade_cols)

    EmptySession = sessionmaker(autocommit=False, autoflush=False, bind=empty_engine)
    with EmptySession() as db:
        wallets = {w.market: w for w in db.query(WalletAccount).all()}
        assert set(wallets.keys()) == {"bybit", "forex"}
        assert wallets["bybit"].active_balance == Decimal("10000.000000")
        assert wallets["bybit"].account_profile == "bybit_vip0_linear"
        assert wallets["bybit"].bybit_leverage == 5
        assert wallets["forex"].active_balance == Decimal("10000.000000")
        assert wallets["forex"].account_profile == "mt5_raw_ecn"
        assert wallets["forex"].forex_leverage == 500

    # 2. Existing legacy DB (pre-Phase-1 trades table with existing closed profits, no wallet_accounts)
    legacy_path = tmp_path / "legacy_pre_phase1.db"
    raw_conn = sqlite3.connect(legacy_path)
    try:
        raw_conn.execute(
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
        raw_conn.execute(
            """
            INSERT INTO trades (
                market, symbol, side, strategy_name, status, qty,
                entry_price, current_price, exit_price, stop_loss, take_profit,
                pnl_usd, pnl_pct, entry_reason
            ) VALUES
                ('BYBIT', 'BTCUSDT', 'LONG', 'trend_rider', 'CLOSED', 0.1, 90000, 92500, 92500, 89000, 92500, 250.0, 2.78, 'legacy'),
                ('FOREX', 'EUR_USD', 'SHORT', 'mean_reversion', 'CLOSED', 10000, 1.09, 1.08, 1.08, 1.095, 1.08, 100.0, 0.92, 'legacy')
            """
        )
        raw_conn.commit()
    finally:
        raw_conn.close()

    legacy_engine = create_db_engine(f"sqlite:///{legacy_path}")
    init_db(legacy_engine)
    init_db(legacy_engine)  # Idempotent re-run must not fail or overwrite wallets

    legacy_insp = inspect(legacy_engine)
    legacy_cols = {c["name"] for c in legacy_insp.get_columns("trades")}
    assert expected_new_cols.issubset(legacy_cols)

    LegacySession = sessionmaker(autocommit=False, autoflush=False, bind=legacy_engine)
    with LegacySession() as db:
        # Existing rows preserved intact
        assert db.query(Trade).count() == 2
        wallets = {w.market: w for w in db.query(WalletAccount).all()}
        # Seamlessly seeded from existing Bybit ($10,000 + $250) and Forex ($10,000 + $100) balances!
        assert wallets["bybit"].active_balance == Decimal("10250.000000")
        assert wallets["bybit"].seed_capital == Decimal("10250.000000")
        assert wallets["bybit"].equity_baseline_day == Decimal("10250.000000")
        assert wallets["forex"].active_balance == Decimal("10100.000000")
        assert wallets["forex"].seed_capital == Decimal("10100.000000")
        assert wallets["forex"].equity_baseline_day == Decimal("10100.000000")
        # WalletTransaction table is ready to accept rows
        db.add(
            WalletTransaction(
                market="bybit",
                tx_type="DEPOSIT",
                network="INTERNAL",
                gross_amount=Decimal("50.000000"),
                net_amount=Decimal("50.000000"),
                active_balance_before=Decimal("10250.000000"),
                active_balance_after=Decimal("10300.000000"),
                vault_balance_before=Decimal("0.000000"),
                vault_balance_after=Decimal("0.000000"),
                note="Migration sanity check",
            )
        )
        db.commit()
        assert db.query(WalletTransaction).count() == 1


# ---------------------------------------------------------------------------
# Gate 6: Prompt 2 Pure Math — Min Qty Closing Bypass, Clamped Slippage, Stop-Out Helper & Friction Estimator
# ---------------------------------------------------------------------------


def test_bybit_min_qty_sol_and_closing_bypass_and_clamped_slippage() -> None:
    """Verify BTCUSDT@77k=0.002, SOLUSDT@150=0.1, closing bypass=min_qty, and slippage clamping."""
    assert bybit_effective_min_qty("BTCUSDT", Decimal("77000")) == Decimal("0.002")
    assert bybit_effective_min_qty("BTCUSDT", Decimal("77000"), is_closing=True) == Decimal("0.001")
    assert bybit_effective_min_qty("SOLUSDT", Decimal("150")) == Decimal("0.1")

    # Exact VIP 0 taker (0.055%) vs maker (0.020%) on $1,000 notional
    maker_fill = bybit_simulate_fill(
        symbol="ETHUSDT",
        side="LONG",
        is_entry=True,
        mid_price=Decimal("2000"),
        order_type="LIMIT",
        qty=Decimal("0.5"),
    )
    assert maker_fill.fee == Decimal("0.200000")

    # Extreme volatility Market fill clamps slippage at <= 5.0 bps
    hi_vol_mkt = bybit_simulate_fill(
        symbol="SOLUSDT",
        side="LONG",
        is_entry=True,
        mid_price=Decimal("100"),
        order_type="MARKET",
        qty=Decimal("10"),
        volatility_pct=Decimal("0.50"),
    )
    # 5.0 bps of $100 * 10 = $0.500000
    assert hi_vol_mkt.slippage_cost == Decimal("0.500000")

    # Low volatility Stop-Loss fill clamps slippage at >= 1.5 bps; extreme volatility clamps at <= 8.0 bps
    lo_vol_sl = bybit_simulate_fill(
        symbol="BTCUSDT",
        side="LONG",
        is_entry=False,
        mid_price=Decimal("10000"),
        order_type="STOP_LOSS",
        qty=Decimal("1"),
        volatility_pct=Decimal("0.0001"),
    )
    assert lo_vol_sl.slippage_cost == Decimal("1.500000")

    hi_vol_sl = bybit_simulate_fill(
        symbol="SOLUSDT",
        side="LONG",
        is_entry=False,
        mid_price=Decimal("100"),
        order_type="STOP_LOSS",
        qty=Decimal("10"),
        volatility_pct=Decimal("0.50"),
    )
    assert hi_vol_sl.slippage_cost == Decimal("0.800000")


def test_select_stop_out_targets_and_swap_date_string_idempotency() -> None:
    """Verify select_stop_out_targets picks largest loser first and swap returns/accepts YYYY-MM-DD."""
    positions = [
        ForexPositionSnapshot(
            trade_id=10,
            symbol="EUR_USD",
            side="LONG",
            lots=Decimal("0.01"),
            entry_price=Decimal("1.1000"),
            current_price=Decimal("1.0990"),
            unrealized_pnl=Decimal("-1.00"),
            used_margin=Decimal("2.20"),
        ),
        ForexPositionSnapshot(
            trade_id=20,
            symbol="XAU_USD",
            side="LONG",
            lots=Decimal("0.01"),
            entry_price=Decimal("2500.00"),
            current_price=Decimal("2485.00"),
            unrealized_pnl=Decimal("-15.00"),
            used_margin=Decimal("5.00"),
        ),
        ForexPositionSnapshot(
            trade_id=30,
            symbol="GBP_USD",
            side="SHORT",
            lots=Decimal("0.01"),
            entry_price=Decimal("1.2500"),
            current_price=Decimal("1.2490"),
            unrealized_pnl=Decimal("1.00"),
            used_margin=Decimal("2.50"),
        ),
    ]
    # Total used margin = 9.70; equity = 4.00 -> margin level = 41.24% <= 50%
    # Closing trade 20 (largest loser, margin 5.00) leaves margin 4.70 -> level = 85.11% > 50%
    targets = select_stop_out_targets(positions, equity=Decimal("4.00"), stop_out_pct=Decimal("50"))
    assert [t.trade_id for t in targets] == [20]

    # When healthy (> 50%), returns empty list
    assert select_stop_out_targets(positions, equity=Decimal("20.00"), stop_out_pct=Decimal("50")) == []

    # Swap date string YYYY-MM-DD idempotency on Wednesday (3x multiplier)
    wed_swap = forex_calculate_rollover_swap(
        symbol="EUR_USD",
        side="LONG",
        lots=Decimal("0.01"),
        opened_at=datetime(2026, 6, 3, 18, 0, tzinfo=timezone.utc),
        now_utc=datetime(2026, 6, 3, 21, 5, tzinfo=timezone.utc),
        last_swap_applied_date=None,
    )
    assert wed_swap.total_multiplier == 3
    assert wed_swap.applied_date_str == "2026-06-03"
    assert wed_swap.fee_amount == Decimal("0.195000")

    # Passing YYYY-MM-DD string prevents duplicate application
    dup_swap = forex_calculate_rollover_swap(
        symbol="EUR_USD",
        side="LONG",
        lots=Decimal("0.01"),
        opened_at=datetime(2026, 6, 3, 18, 0, tzinfo=timezone.utc),
        now_utc=datetime(2026, 6, 3, 22, 0, tzinfo=timezone.utc),
        last_swap_applied_date=wed_swap.applied_date_str,
    )
    assert dup_swap.fee_amount == Decimal("0")
    assert dup_swap.rollover_days_applied == 0


def test_estimate_round_turn_friction_bybit_and_forex() -> None:
    """Verify estimate_round_turn_friction returns exact Decimal metrics for Bybit and Forex."""
    bybit_est = estimate_round_turn_friction(
        market="bybit",
        symbol="SOLUSDT",
        side="LONG",
        qty=Decimal("1.0"),
        entry_price=Decimal("150.00"),
        sl_price=Decimal("147.00"),
        tp_price=Decimal("156.00"),
        leverage=5,
        profile="bybit_vip0_linear",
    )
    assert bybit_est.gross_tp_profit == Decimal("6.000000")
    assert bybit_est.gross_sl_loss == Decimal("3.000000")
    assert bybit_est.entry_fee > Decimal("0")
    assert bybit_est.est_exit_fee_tp == Decimal("0.031200")  # 156 * 1 * 0.00020 (maker TP)
    assert bybit_est.est_exit_fee_sl > bybit_est.est_exit_fee_tp  # taker SL > maker TP
    assert bybit_est.total_friction_tp == (
        bybit_est.entry_fee
        + bybit_est.est_exit_fee_tp
        + bybit_est.est_spread_cost
        + bybit_est.est_slippage_cost
    )
    assert bybit_est.net_tp_profit == bybit_est.gross_tp_profit - bybit_est.total_friction_tp
    assert bybit_est.net_sl_loss > bybit_est.gross_sl_loss
    assert Decimal("1.5") < bybit_est.net_rr < Decimal("2.0")
    assert Decimal("0") < bybit_est.fee_drag_pct_of_tp < Decimal("5.0")

    # Forex EUR_USD 0.01 lot ECN: $0.035 entry + $0.035 exit = $0.07 commission
    fx_est = estimate_round_turn_friction(
        market="forex",
        symbol="EUR_USD",
        side="LONG",
        qty=Decimal("0.01"),
        entry_price=Decimal("1.10000"),
        sl_price=Decimal("1.09800"),
        tp_price=Decimal("1.10400"),
        leverage=500,
        profile="mt5_raw_ecn",
    )
    assert fx_est.entry_fee == Decimal("0.035000")
    assert fx_est.est_exit_fee_tp == Decimal("0.035000")
    assert fx_est.est_exit_fee_sl == Decimal("0.035000")
    assert fx_est.gross_tp_profit == Decimal("4.000000")
    assert fx_est.gross_sl_loss == Decimal("2.000000")
    assert fx_est.net_tp_profit < fx_est.gross_tp_profit
    assert fx_est.net_sl_loss > fx_est.gross_sl_loss
    assert fx_est.net_rr > Decimal("1.5")
    assert fx_est.fee_drag_pct_of_tp > Decimal("1.5")

