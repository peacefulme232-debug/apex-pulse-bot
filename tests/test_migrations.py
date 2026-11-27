"""Idempotent database schema & startup migration verification tests."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock

from sqlalchemy import inspect
from sqlalchemy.orm import sessionmaker

from app.database import create_db_engine, init_db, migrate_schema
from app.models import (
    ArmedRetestSetup,
    ExecutionRejection,
    Trade,
    WalletAccount,
    WalletTransaction,
)


EXPECTED_TRADE_EXTENSION_COLS = {
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
    "breakeven_applied",
    "last_funding_applied_at",
    "last_swap_applied_date",
}

EXPECTED_WALLET_ACCOUNT_COLS = {
    "id",
    "market",
    "seed_capital",
    "active_balance",
    "vault_balance",
    "equity_baseline_day",
    "cumulative_net_cashflow_day",
    "account_profile",
    "realism_enabled",
    "forex_leverage",
    "bybit_leverage",
    "allow_min_lot_override",
    "auto_sweep_enabled",
    "auto_sweep_base_capital",
    "auto_sweep_threshold",
    "auto_sweep_network",
    "consecutive_losses",
    "circuit_breaker_until",
    "updated_at",
}

EXPECTED_WALLET_TRANSACTION_COLS = {
    "id",
    "market",
    "tx_type",
    "network",
    "gross_amount",
    "network_fee",
    "fx_spread_fee",
    "net_amount",
    "active_balance_before",
    "active_balance_after",
    "vault_balance_before",
    "vault_balance_after",
    "note",
    "created_at",
}

EXPECTED_EXECUTION_REJECTION_COLS = {
    "id",
    "market",
    "symbol",
    "side",
    "strategy",
    "confluence_score",
    "reason_code",
    "requested_qty",
    "required_min_qty",
    "required_margin",
    "available_free_margin",
    "details",
    "created_at",
}

EXPECTED_ARMED_RETEST_COLS = {
    "id",
    "market",
    "symbol",
    "side",
    "zone_low",
    "zone_high",
    "breakout_candle_range",
    "breakout_atr",
    "bars_remaining",
    "status",
    "created_at",
    "updated_at",
}


def test_fresh_sqlite_migration_idempotent_twice_and_decimal_roundtrip(tmp_path: Path) -> None:
    """Running init_db twice on a fresh SQLite DB succeeds with zero errors and creates all tables/columns."""
    db_path = tmp_path / "fresh_migrations.db"
    eng = create_db_engine(f"sqlite:///{db_path}")

    init_db(eng)
    init_db(eng)  # Second run must be completely idempotent

    insp = inspect(eng)
    tables = set(insp.get_table_names())
    assert {
        "trades",
        "wallet_accounts",
        "wallet_transactions",
        "execution_rejections",
        "armed_retest_setups",
    }.issubset(tables)

    assert EXPECTED_TRADE_EXTENSION_COLS.issubset({c["name"] for c in insp.get_columns("trades")})
    assert EXPECTED_WALLET_ACCOUNT_COLS.issubset({c["name"] for c in insp.get_columns("wallet_accounts")})
    assert EXPECTED_WALLET_TRANSACTION_COLS.issubset({c["name"] for c in insp.get_columns("wallet_transactions")})
    assert EXPECTED_EXECUTION_REJECTION_COLS.issubset({c["name"] for c in insp.get_columns("execution_rejections")})
    assert EXPECTED_ARMED_RETEST_COLS.issubset({c["name"] for c in insp.get_columns("armed_retest_setups")})

    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    with SessionCls() as db:
        # Verify ArmedRetestSetup persistence with exact NUMERIC(18, 8) precision
        setup = ArmedRetestSetup(
            market="bybit",
            symbol="BTCUSDT",
            side="LONG",
            zone_low=Decimal("68123.12345678"),
            zone_high=Decimal("68456.87654321"),
            breakout_candle_range=Decimal("333.75308643"),
            breakout_atr=Decimal("210.00005000"),
        )
        db.add(setup)

        # Verify ExecutionRejection with strategy + confluence_score + nullable numeric cols
        rej = ExecutionRejection(
            market="forex",
            symbol="XAU_USD",
            side="SHORT",
            strategy="breakout_retest",
            confluence_score=Decimal("78.50"),
            reason_code="BELOW_CONFLUENCE_THRESHOLD",
            requested_qty=None,
            required_min_qty=None,
            required_margin=None,
            available_free_margin=Decimal("20.000000"),
            details="Vetoed by confluence gate",
        )
        db.add(rej)
        db.commit()

        loaded_setup = db.query(ArmedRetestSetup).first()
        assert loaded_setup is not None
        assert loaded_setup.bars_remaining == 10
        assert loaded_setup.status == "ARMED"
        assert loaded_setup.zone_low == Decimal("68123.12345678")
        assert loaded_setup.zone_high == Decimal("68456.87654321")
        assert loaded_setup.breakout_candle_range == Decimal("333.75308643")
        assert loaded_setup.breakout_atr == Decimal("210.00005000")

        loaded_rej = db.query(ExecutionRejection).first()
        assert loaded_rej is not None
        assert loaded_rej.strategy == "breakout_retest"
        assert loaded_rej.confluence_score == Decimal("78.50")
        assert loaded_rej.requested_qty is None
        assert loaded_rej.available_free_margin == Decimal("20.000000")


def test_legacy_db_upgrade_adds_new_columns_and_preserves_rows(tmp_path: Path) -> None:
    """Upgrading a legacy DB missing new columns adds them safely without dropping existing rows."""
    legacy_path = tmp_path / "legacy_upgrade.db"
    raw_conn = sqlite3.connect(legacy_path)
    try:
        # Legacy trades table (missing all realism & breakeven_applied columns)
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
                ('BYBIT', 'ETHUSDT', 'LONG', 'trend_rider', 'CLOSED', 1.0, 3000.0, 3150.0, 3150.0, 2950.0, 3150.0, 150.0, 5.0, 'legacy_trade'),
                ('FOREX', 'EUR_USD', 'SHORT', 'mean_reversion', 'CLOSED', 10000.0, 1.10, 1.095, 1.095, 1.105, 1.095, 50.0, 0.45, 'legacy_trade')
            """
        )

        # Legacy wallet_accounts table (missing consecutive_losses & circuit_breaker_until)
        raw_conn.execute(
            """
            CREATE TABLE wallet_accounts (
                id INTEGER PRIMARY KEY,
                market VARCHAR(16) UNIQUE,
                seed_capital NUMERIC(18, 6),
                active_balance NUMERIC(18, 6),
                vault_balance NUMERIC(18, 6),
                equity_baseline_day NUMERIC(18, 6),
                cumulative_net_cashflow_day NUMERIC(18, 6),
                account_profile VARCHAR(32),
                realism_enabled BOOLEAN,
                forex_leverage INTEGER,
                bybit_leverage INTEGER,
                allow_min_lot_override BOOLEAN,
                auto_sweep_enabled BOOLEAN,
                auto_sweep_base_capital NUMERIC(18, 6),
                auto_sweep_threshold NUMERIC(18, 6),
                auto_sweep_network VARCHAR(32),
                updated_at DATETIME
            )
            """
        )
        raw_conn.execute(
            """
            INSERT INTO wallet_accounts (
                market, seed_capital, active_balance, vault_balance,
                equity_baseline_day, cumulative_net_cashflow_day,
                account_profile, realism_enabled, forex_leverage, bybit_leverage,
                allow_min_lot_override, auto_sweep_enabled,
                auto_sweep_base_capital, auto_sweep_threshold, auto_sweep_network
            ) VALUES
                ('bybit', 20.000000, 35.500000, 10.000000, 35.500000, 0.000000,
                 'bybit_vip0_linear', 1, 500, 5, 0, 0, 20.000000, 50.000000, 'INTERNAL')
            """
        )

        # Legacy execution_rejections table (missing strategy & confluence_score)
        raw_conn.execute(
            """
            CREATE TABLE execution_rejections (
                id INTEGER PRIMARY KEY,
                market VARCHAR(16),
                symbol VARCHAR(32),
                side VARCHAR(8),
                reason_code VARCHAR(32),
                requested_qty NUMERIC(18, 6),
                required_min_qty NUMERIC(18, 6),
                required_margin NUMERIC(18, 6),
                available_free_margin NUMERIC(18, 6),
                details TEXT,
                created_at DATETIME
            )
            """
        )
        raw_conn.execute(
            """
            INSERT INTO execution_rejections (
                market, symbol, side, reason_code, requested_qty,
                required_min_qty, required_margin, available_free_margin, details
            ) VALUES
                ('bybit', 'BTCUSDT', 'LONG', 'BELOW_MIN_NOTIONAL', 0.0001, 0.001, 15.0, 20.0, 'legacy_rej')
            """
        )
        raw_conn.commit()
    finally:
        raw_conn.close()

    eng = create_db_engine(f"sqlite:///{legacy_path}")
    init_db(eng)
    init_db(eng)  # Second run must be idempotent

    insp = inspect(eng)
    assert "armed_retest_setups" in set(insp.get_table_names())
    assert EXPECTED_TRADE_EXTENSION_COLS.issubset({c["name"] for c in insp.get_columns("trades")})
    assert EXPECTED_WALLET_ACCOUNT_COLS.issubset({c["name"] for c in insp.get_columns("wallet_accounts")})
    assert EXPECTED_EXECUTION_REJECTION_COLS.issubset({c["name"] for c in insp.get_columns("execution_rejections")})

    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    with SessionCls() as db:
        # Legacy trades survived intact with breakeven_applied defaulting to False
        trades = db.query(Trade).order_by(Trade.id).all()
        assert len(trades) == 2
        assert trades[0].symbol == "ETHUSDT"
        assert trades[0].breakeven_applied is False
        assert trades[1].symbol == "EUR_USD"
        assert trades[1].breakeven_applied is False

        # Existing 'bybit' wallet row survived intact with new columns defaulted; missing 'forex' row was seeded
        wallets = {w.market: w for w in db.query(WalletAccount).all()}
        assert set(wallets.keys()) == {"bybit", "forex"}
        assert wallets["bybit"].active_balance == Decimal("35.500000")
        assert wallets["bybit"].vault_balance == Decimal("10.000000")
        assert wallets["bybit"].consecutive_losses == 0
        assert wallets["bybit"].circuit_breaker_until is None

        # Seeded 'forex' wallet picked up existing $10,000 + $50 legacy trade profit
        assert wallets["forex"].active_balance == Decimal("10050.000000")
        assert wallets["forex"].seed_capital == Decimal("10050.000000")
        assert wallets["forex"].equity_baseline_day == Decimal("10050.000000")
        assert wallets["forex"].account_profile == "mt5_raw_ecn"
        assert wallets["forex"].consecutive_losses == 0

        # Legacy execution_rejections row survived intact
        rejections = db.query(ExecutionRejection).all()
        assert len(rejections) == 1
        assert rejections[0].reason_code == "BELOW_MIN_NOTIONAL"
        assert rejections[0].strategy is None
        assert rejections[0].confluence_score is None


def test_wallet_accounts_seeds_bybit_and_forex_cleanly(tmp_path: Path) -> None:
    """wallet_accounts seeds 'bybit' and 'forex' with exact defaults and preserves edits on re-run."""
    db_path = tmp_path / "seed_wallets.db"
    eng = create_db_engine(f"sqlite:///{db_path}")
    init_db(eng)

    SessionCls = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    with SessionCls() as db:
        wallets = {w.market: w for w in db.query(WalletAccount).all()}
        assert set(wallets.keys()) == {"bybit", "forex"}

        bybit = wallets["bybit"]
        assert bybit.seed_capital == Decimal("10000.000000")
        assert bybit.active_balance == Decimal("10000.000000")
        assert bybit.vault_balance == Decimal("0.000000")
        assert bybit.equity_baseline_day == Decimal("10000.000000")
        assert bybit.cumulative_net_cashflow_day == Decimal("0.000000")
        assert bybit.account_profile == "bybit_vip0_linear"
        assert bybit.realism_enabled is True
        assert bybit.bybit_leverage == 5
        assert bybit.forex_leverage == 500
        assert bybit.allow_min_lot_override is False
        assert bybit.auto_sweep_enabled is False
        assert bybit.auto_sweep_base_capital == Decimal("20.000000")
        assert bybit.auto_sweep_threshold == Decimal("50.000000")
        assert bybit.auto_sweep_network == "INTERNAL"
        assert bybit.consecutive_losses == 0
        assert bybit.circuit_breaker_until is None

        forex = wallets["forex"]
        assert forex.seed_capital == Decimal("10000.000000")
        assert forex.active_balance == Decimal("10000.000000")
        assert forex.account_profile == "mt5_raw_ecn"
        assert forex.consecutive_losses == 0
        assert forex.circuit_breaker_until is None

        # Mutate wallet state and verify a second init_db does NOT overwrite it
        cb_until = datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc)
        bybit.active_balance = Decimal("20.000000")
        bybit.consecutive_losses = 3
        bybit.circuit_breaker_until = cb_until
        db.commit()

    init_db(eng)
    with SessionCls() as db:
        bybit_after = db.query(WalletAccount).filter(WalletAccount.market == "bybit").first()
        assert bybit_after is not None
        assert bybit_after.active_balance == Decimal("20.000000")
        assert bybit_after.consecutive_losses == 3
        assert bybit_after.circuit_breaker_until is not None

        # WalletTransaction accepts exact Decimal ledger rows
        tx = WalletTransaction(
            market="bybit",
            tx_type="DEPOSIT",
            network="INTERNAL",
            gross_amount=Decimal("20.000000"),
            network_fee=Decimal("0.000000"),
            fx_spread_fee=Decimal("0.000000"),
            net_amount=Decimal("20.000000"),
            active_balance_before=Decimal("0.000000"),
            active_balance_after=Decimal("20.000000"),
            vault_balance_before=Decimal("0.000000"),
            vault_balance_after=Decimal("0.000000"),
            note="Initial deposit",
        )
        db.add(tx)
        db.commit()
        assert db.query(WalletTransaction).count() == 1


def test_postgres_migration_uses_add_column_if_not_exists() -> None:
    """Verify PostgreSQL migration path issues idempotent ADD COLUMN IF NOT EXISTS statements."""
    executed_sql: list[str] = []

    mock_conn = MagicMock()

    def _fake_execute(stmt, params=None):
        sql_str = str(stmt)
        executed_sql.append(sql_str)
        result = MagicMock()
        result.fetchone.return_value = (1,)
        return result

    mock_conn.execute.side_effect = _fake_execute
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_conn
    mock_ctx.__exit__.return_value = False

    mock_engine = MagicMock()
    mock_engine.dialect.name = "postgresql"
    mock_engine.begin.return_value = mock_ctx

    migrate_schema(mock_engine)

    alter_stmts = [s for s in executed_sql if s.startswith("ALTER TABLE")]
    assert any("ALTER TABLE trades ADD COLUMN IF NOT EXISTS breakeven_applied BOOLEAN DEFAULT FALSE" in s for s in alter_stmts)
    assert any("ALTER TABLE wallet_accounts ADD COLUMN IF NOT EXISTS consecutive_losses INTEGER NOT NULL DEFAULT 0" in s for s in alter_stmts)
    assert any("ALTER TABLE wallet_accounts ADD COLUMN IF NOT EXISTS circuit_breaker_until TIMESTAMP WITH TIME ZONE" in s for s in alter_stmts)
    assert any("ALTER TABLE execution_rejections ADD COLUMN IF NOT EXISTS strategy VARCHAR(48)" in s for s in alter_stmts)
    assert any("ALTER TABLE execution_rejections ADD COLUMN IF NOT EXISTS confluence_score NUMERIC(6, 2)" in s for s in alter_stmts)
