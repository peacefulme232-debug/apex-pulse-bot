"""Virtual Wallet / Vault Service: exact Decimal accounting, withdrawal fees, margin guards, auto-sweep & cash-flow isolation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import ExecutionRejection, SystemLog, Trade, WalletAccount, WalletTransaction
from app.services.realism_engine import (
    BYBIT_TAKER_FEE_RATE,
    HUNDRED,
    ZERO,
    forex_commission_per_side,
    forex_margin_level_pct,
    forex_required_margin,
    forex_units_to_lots,
    q6,
    to_decimal,
)

# ---------------------------------------------------------------------------
# Exceptions (mapped to HTTP 400 / 409 in FastAPI routes)
# ---------------------------------------------------------------------------


class WalletError(Exception):
    """Base wallet service error with HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class WalletValidationError(WalletError):
    """HTTP 400 validation or insufficient withdrawable balance error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class WalletConflictError(WalletError):
    """HTTP 409 conflict error (e.g., open positions exist during leave-balance)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=409)


# ---------------------------------------------------------------------------
# A. Verified Realistic Withdrawal Fee Schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkFeeSpec:
    """Fee & minimum withdrawal rules for a transfer/withdrawal rail."""

    network: str
    fixed_fee: Decimal
    fx_spread_rate: Decimal  # e.g. Decimal("0.005") = 0.50%
    min_amount: Decimal
    allowed_markets: frozenset[str]


NETWORK_FEE_SCHEDULE: dict[str, NetworkFeeSpec] = {
    "INTERNAL": NetworkFeeSpec(
        network="INTERNAL",
        fixed_fee=Decimal("0.00"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("1.00"),
        allowed_markets=frozenset({"bybit", "forex"}),
    ),
    "APTOS": NetworkFeeSpec(
        network="APTOS",
        fixed_fee=Decimal("0.002"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("1.00"),
        allowed_markets=frozenset({"bybit"}),
    ),
    "POLYGON": NetworkFeeSpec(
        network="POLYGON",
        fixed_fee=Decimal("0.07"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("1.00"),
        allowed_markets=frozenset({"bybit"}),
    ),
    "BEP20": NetworkFeeSpec(
        network="BEP20",
        fixed_fee=Decimal("0.15"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("5.00"),
        allowed_markets=frozenset({"bybit", "forex"}),
    ),
    "ERC20": NetworkFeeSpec(
        network="ERC20",
        fixed_fee=Decimal("0.80"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("6.00"),
        allowed_markets=frozenset({"bybit"}),
    ),
    "TRC20": NetworkFeeSpec(
        network="TRC20",
        fixed_fee=Decimal("1.30"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("10.00"),
        allowed_markets=frozenset({"bybit", "forex"}),
    ),
    "LOCAL_BANK_NGN": NetworkFeeSpec(
        network="LOCAL_BANK_NGN",
        fixed_fee=Decimal("0.50"),
        fx_spread_rate=Decimal("0.005"),  # 0.50% FX spread
        min_amount=Decimal("10.00"),
        allowed_markets=frozenset({"bybit", "forex"}),
    ),
    "BANK_WIRE": NetworkFeeSpec(
        network="BANK_WIRE",
        fixed_fee=Decimal("15.00"),
        fx_spread_rate=Decimal("0"),
        min_amount=Decimal("50.00"),
        allowed_markets=frozenset({"forex"}),
    ),
}

ALLOWED_ACCOUNT_PROFILES = frozenset({"bybit_vip0_linear", "mt5_raw_ecn", "mt5_standard_stp"})
ALLOWED_TRANSFER_TYPES = frozenset(
    {"TRADING_TO_VAULT", "VAULT_TO_TRADING", "TRADING_TO_EXTERNAL", "EXTERNAL_TO_TRADING"}
)


@dataclass(frozen=True)
class FeeBreakdown:
    """Exact Decimal fee calculation result for a transfer or withdrawal."""

    network: str
    gross_amount: Decimal
    network_fee: Decimal
    fx_spread_fee: Decimal
    net_amount: Decimal


def normalize_market(market: str) -> str:
    """Normalize market identifier to lowercase 'bybit' or 'forex'."""
    cleaned = (market or "").strip().lower()
    if cleaned not in {"bybit", "forex"}:
        raise WalletValidationError(f"Unsupported market '{market}'. Must be 'bybit' or 'forex'.")
    return cleaned


def normalize_network(network: str) -> str:
    """Normalize and validate network identifier against the fee schedule."""
    cleaned = (network or "INTERNAL").strip().upper()
    if cleaned not in NETWORK_FEE_SCHEDULE:
        raise WalletValidationError(
            f"Unsupported network '{network}'. Valid options: {', '.join(sorted(NETWORK_FEE_SCHEDULE))}."
        )
    return cleaned


def calculate_withdrawal_fees(
    market: str,
    network: str,
    gross_amount: Decimal | int | float | str,
    is_deposit: bool = False,
) -> FeeBreakdown:
    """Calculate exact Decimal fees for a withdrawal, sweep, or deposit."""
    norm_market = normalize_market(market)
    norm_net = normalize_network(network)
    gross = q6(to_decimal(gross_amount))
    if gross <= ZERO:
        raise WalletValidationError("Amount must be greater than zero.")

    if is_deposit:
        # Deposits into active trading carry $0.00 platform deposit fee
        return FeeBreakdown(
            network=norm_net,
            gross_amount=gross,
            network_fee=q6(ZERO),
            fx_spread_fee=q6(ZERO),
            net_amount=gross,
        )

    spec = NETWORK_FEE_SCHEDULE[norm_net]
    if norm_market not in spec.allowed_markets:
        raise WalletValidationError(
            f"Network '{norm_net}' is not supported for market '{norm_market}'."
        )
    if gross < spec.min_amount:
        raise WalletValidationError(
            f"Amount ${gross:.2f} is below minimum ${spec.min_amount:.2f} for {norm_net}."
        )

    network_fee = q6(spec.fixed_fee)
    fx_spread_fee = q6(gross * spec.fx_spread_rate)
    net_amount = q6(gross - network_fee - fx_spread_fee)
    if net_amount <= ZERO:
        raise WalletValidationError(
            f"Net amount after fees (${net_amount:.6f}) must be greater than zero."
        )

    return FeeBreakdown(
        network=norm_net,
        gross_amount=gross,
        network_fee=network_fee,
        fx_spread_fee=fx_spread_fee,
        net_amount=net_amount,
    )


# ---------------------------------------------------------------------------
# Locking & Margin / Withdrawable Balance Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def lock_wallet_account(db: Session, market: str) -> WalletAccount:
    """Load WalletAccount with SELECT ... FOR UPDATE on PostgreSQL (or normal query on SQLite)."""
    norm_market = normalize_market(market)
    query = db.query(WalletAccount).filter(WalletAccount.market == norm_market)
    bind = db.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        query = query.with_for_update()

    wallet = query.first()
    if wallet is None:
        default_bal = Decimal("10000.000000")
        wallet = WalletAccount(
            market=norm_market,
            seed_capital=default_bal,
            active_balance=default_bal,
            vault_balance=q6(ZERO),
            equity_baseline_day=default_bal,
            cumulative_net_cashflow_day=q6(ZERO),
            account_profile="mt5_raw_ecn" if norm_market == "forex" else "bybit_vip0_linear",
            realism_enabled=True,
            forex_leverage=500,
            bybit_leverage=5,
            allow_min_lot_override=False,
            auto_sweep_enabled=False,
            auto_sweep_base_capital=Decimal("20.000000"),
            auto_sweep_threshold=Decimal("50.000000"),
            auto_sweep_network="INTERNAL",
        )
        db.add(wallet)
        db.flush()
    return wallet


def compute_market_margin_metrics(db: Session, wallet: WalletAccount) -> dict[str, Any]:
    """Compute exact Decimal used_margin, open_fee_reserve, unrealized_pnl, equity, free_margin, margin_level_pct, and withdrawable_active."""
    market_upper = "FOREX" if wallet.market == "forex" else "BYBIT"
    open_trades = (
        db.query(Trade)
        .filter(Trade.status == "OPEN", Trade.market == market_upper)
        .all()
    )

    used_margin = ZERO
    open_fee_reserve = ZERO
    total_unrealized_pnl = ZERO

    for t in open_trades:
        qty_dec = to_decimal(t.qty)
        mark_px = to_decimal(t.current_price if t.current_price else t.entry_price)
        t_margin = to_decimal(t.initial_margin)
        if t_margin <= ZERO and market_upper == "FOREX" and qty_dec > ZERO and mark_px > ZERO:
            lots = forex_units_to_lots(t.symbol, qty_dec)
            t_margin = forex_required_margin(t.symbol, lots, mark_px, int(wallet.forex_leverage or 500))
        used_margin += t_margin

        if wallet.realism_enabled and qty_dec > ZERO and mark_px > ZERO:
            if market_upper == "BYBIT":
                open_fee_reserve += q6(qty_dec * mark_px * BYBIT_TAKER_FEE_RATE)
            else:
                lots = forex_units_to_lots(t.symbol, qty_dec)
                open_fee_reserve += forex_commission_per_side(lots, str(wallet.account_profile or "mt5_raw_ecn"))

        t_net = (
            to_decimal(t.net_pnl)
            if t.filled_entry_price is not None
            else to_decimal(t.pnl_usd)
        )
        t_partial = to_decimal(t.realized_partial_pnl_usd)
        t_funding = to_decimal(t.funding_or_swap_fee)
        total_unrealized_pnl += (t_net - t_partial + t_funding)

    active_bal = q6(to_decimal(wallet.active_balance))
    used_margin = q6(used_margin)
    open_fee_reserve = q6(open_fee_reserve)
    total_unrealized_pnl = q6(total_unrealized_pnl)

    active_equity = q6(active_bal + total_unrealized_pnl)
    free_margin = q6(max(ZERO, active_equity - used_margin))
    margin_level_pct = forex_margin_level_pct(active_equity, used_margin) if used_margin > ZERO else None

    # Strict withdrawable guard:
    # withdrawable_active = max(0, active_balance - used_margin - open_fee_reserve - max(0, -total_unrealized_pnl))
    withdrawable_active = q6(
        max(
            ZERO,
            active_bal - used_margin - open_fee_reserve - max(ZERO, -total_unrealized_pnl),
        )
    )

    return {
        "open_positions_count": len(open_trades),
        "used_margin": used_margin,
        "open_fee_reserve": open_fee_reserve,
        "unrealized_pnl": total_unrealized_pnl,
        "active_equity": active_equity,
        "free_margin": free_margin,
        "margin_level_pct": margin_level_pct,
        "withdrawable_active": withdrawable_active,
    }


def compute_daily_pnl_isolated(
    current_active_equity: Decimal,
    equity_baseline_day: Decimal,
    fallback_capital: Decimal = Decimal("1"),
) -> tuple[Decimal, Decimal]:
    """Compute cash-flow-isolated (daily_pnl_usd, daily_pnl_pct) from equity_baseline_day."""
    daily_pnl_usd = q6(current_active_equity - equity_baseline_day)
    denom = equity_baseline_day if equity_baseline_day > ZERO else max(fallback_capital, Decimal("1"))
    daily_pnl_pct = q6((daily_pnl_usd / denom) * HUNDRED)
    return daily_pnl_usd, daily_pnl_pct


def compute_market_friction_and_reconciliation(
    db: Session,
    wallet: WalletAccount,
) -> tuple[dict[str, Decimal], dict[str, Any]]:
    """Compute lifetime friction breakdown and double-entry reconciliation (drift == 0) for a market."""
    market_upper = "FOREX" if wallet.market == "forex" else "BYBIT"
    trades = db.query(Trade).filter(Trade.market == market_upper).all()
    txs = (
        db.query(WalletTransaction)
        .filter(WalletTransaction.market == wallet.market)
        .order_by(WalletTransaction.id.asc())
        .all()
    )

    entry_fees = ZERO
    exit_fees = ZERO
    spread_cost = ZERO
    slippage_cost = ZERO
    funding_swaps = ZERO
    gross_pnl_total = ZERO
    net_pnl_total = ZERO
    realized_net_pnl = ZERO

    for t in trades:
        entry_fees += to_decimal(t.entry_fee)
        exit_fees += to_decimal(t.exit_fee)
        spread_cost += to_decimal(t.spread_cost)
        slippage_cost += to_decimal(t.slippage_cost)
        funding_swaps += to_decimal(t.funding_or_swap_fee)
        if t.status == "CLOSED":
            t_gross = to_decimal(t.gross_pnl) if t.filled_exit_price is not None else to_decimal(t.pnl_usd)
            t_net = to_decimal(t.net_pnl) if t.filled_exit_price is not None else to_decimal(t.pnl_usd)
            gross_pnl_total += t_gross
            net_pnl_total += t_net
            # Only trades settled through the Phase 1/2 ledger (filled_exit_price set) mutated active_balance post-seed
            if t.filled_exit_price is not None:
                realized_net_pnl += t_net
        else:
            # Partial TP scale-out profits and periodic funding/swap already settled on active_balance
            realized_net_pnl += to_decimal(t.realized_partial_pnl_usd) - to_decimal(t.funding_or_swap_fee)

    withdrawal_fees = ZERO
    external_inflows = ZERO
    external_outflows = ZERO
    internal_transfer_fees = ZERO
    reset_adjustments = ZERO

    for tx in txs:
        tx_fee = to_decimal(tx.network_fee) + to_decimal(tx.fx_spread_fee)
        withdrawal_fees += tx_fee

        ttype = (tx.tx_type or "").upper()
        if ttype == "EXTERNAL_TO_TRADING":
            external_inflows += to_decimal(tx.net_amount)
        elif ttype == "TRADING_TO_EXTERNAL":
            external_outflows += to_decimal(tx.gross_amount)
        elif ttype == "RESET_CHALLENGE":
            delta_active = to_decimal(tx.active_balance_after) - to_decimal(tx.active_balance_before)
            delta_vault = to_decimal(tx.vault_balance_after) - to_decimal(tx.vault_balance_before)
            reset_adjustments += delta_active + delta_vault
        else:
            # TRADING_TO_VAULT, VAULT_TO_TRADING, AUTO_SWEEP, LEAVE_BALANCE_CHALLENGE
            internal_transfer_fees += tx_fee

    total_friction = q6(entry_fees + exit_fees + spread_cost + slippage_cost + funding_swaps + withdrawal_fees)
    friction = {
        "entry_fees": q6(entry_fees),
        "exit_fees": q6(exit_fees),
        "spread_cost": q6(spread_cost),
        "slippage_cost": q6(slippage_cost),
        "funding_swaps": q6(funding_swaps),
        "withdrawal_fees": q6(withdrawal_fees),
        "total_friction": total_friction,
        "gross_pnl_total": q6(gross_pnl_total),
        "net_pnl_total": q6(net_pnl_total),
    }

    expected_total = q6(
        to_decimal(wallet.seed_capital)
        + realized_net_pnl
        + external_inflows
        - external_outflows
        - internal_transfer_fees
        + reset_adjustments
    )
    actual_total = q6(to_decimal(wallet.active_balance) + to_decimal(wallet.vault_balance))
    drift = q6(actual_total - expected_total)
    reconciliation = {
        "reconciled": abs(drift) <= Decimal("0.00001"),
        "drift": drift,
        "expected_total": expected_total,
        "actual_total": actual_total,
    }
    return friction, reconciliation


# ---------------------------------------------------------------------------
# Core Atomic Wallet Mutations
# ---------------------------------------------------------------------------


def _record_transaction(
    db: Session,
    wallet: WalletAccount,
    tx_type: str,
    network: str,
    gross_amount: Decimal,
    network_fee: Decimal,
    fx_spread_fee: Decimal,
    net_amount: Decimal,
    active_before: Decimal,
    active_after: Decimal,
    vault_before: Decimal,
    vault_after: Decimal,
    note: str = "",
) -> WalletTransaction:
    tx = WalletTransaction(
        market=wallet.market,
        tx_type=tx_type,
        network=network,
        gross_amount=q6(gross_amount),
        network_fee=q6(network_fee),
        fx_spread_fee=q6(fx_spread_fee),
        net_amount=q6(net_amount),
        active_balance_before=q6(active_before),
        active_balance_after=q6(active_after),
        vault_balance_before=q6(vault_before),
        vault_balance_after=q6(vault_after),
        note=note,
        created_at=_utcnow(),
    )
    db.add(tx)
    db.flush()
    return tx


def _apply_cashflow_baseline_shift(
    wallet: WalletAccount,
    active_before: Decimal,
    active_after: Decimal,
) -> Decimal:
    """Shift equity_baseline_day and cumulative_net_cashflow_day by delta so cash flows never pollute Today's PnL."""
    delta = q6(active_after - active_before)
    wallet.equity_baseline_day = q6(to_decimal(wallet.equity_baseline_day) + delta)
    wallet.cumulative_net_cashflow_day = q6(to_decimal(wallet.cumulative_net_cashflow_day) + delta)
    wallet.updated_at = _utcnow()
    return delta


def execute_transfer(
    db: Session,
    market: str,
    tx_type: str,
    amount: Decimal | int | float | str,
    network: str = "INTERNAL",
    note: str = "",
    commit: bool = True,
) -> dict[str, Any]:
    """Atomically execute TRADING_TO_VAULT, VAULT_TO_TRADING, TRADING_TO_EXTERNAL, or EXTERNAL_TO_TRADING."""
    norm_market = normalize_market(market)
    norm_type = (tx_type or "").strip().upper()
    if norm_type not in ALLOWED_TRANSFER_TYPES:
        raise WalletValidationError(
            f"Unsupported transfer type '{tx_type}'. Valid: {', '.join(sorted(ALLOWED_TRANSFER_TYPES))}."
        )

    wallet = lock_wallet_account(db, norm_market)
    active_before = q6(to_decimal(wallet.active_balance))
    vault_before = q6(to_decimal(wallet.vault_balance))
    gross = q6(to_decimal(amount))

    if norm_type == "EXTERNAL_TO_TRADING":
        fees = calculate_withdrawal_fees(norm_market, network, gross, is_deposit=True)
        active_after = q6(active_before + fees.net_amount)
        vault_after = vault_before
    elif norm_type == "VAULT_TO_TRADING":
        fees = calculate_withdrawal_fees(norm_market, "INTERNAL", gross, is_deposit=True)
        if gross > vault_before:
            raise WalletValidationError(
                f"Insufficient vault balance on {norm_market}: requested ${gross:.2f}, available ${vault_before:.2f}."
            )
        active_after = q6(active_before + fees.net_amount)
        vault_after = q6(vault_before - gross)
    elif norm_type in {"TRADING_TO_VAULT", "TRADING_TO_EXTERNAL"}:
        fees = calculate_withdrawal_fees(norm_market, network, gross, is_deposit=False)
        metrics = compute_market_margin_metrics(db, wallet)
        withdrawable = metrics["withdrawable_active"]
        if gross > withdrawable:
            raise WalletValidationError(
                f"Withdrawal ${gross:.2f} exceeds withdrawable active balance ${withdrawable:.2f} "
                f"(used margin: ${metrics['used_margin']:.2f})."
            )
        active_after = q6(active_before - gross)
        vault_after = q6(vault_before + fees.net_amount) if norm_type == "TRADING_TO_VAULT" else vault_before
    else:
        raise WalletValidationError(f"Unsupported transfer type '{norm_type}'.")

    wallet.active_balance = active_after
    wallet.vault_balance = vault_after
    _apply_cashflow_baseline_shift(wallet, active_before, active_after)

    tx = _record_transaction(
        db=db,
        wallet=wallet,
        tx_type=norm_type,
        network=fees.network,
        gross_amount=fees.gross_amount,
        network_fee=fees.network_fee,
        fx_spread_fee=fees.fx_spread_fee,
        net_amount=fees.net_amount,
        active_before=active_before,
        active_after=active_after,
        vault_before=vault_before,
        vault_after=vault_after,
        note=note or f"{norm_type} via {fees.network}",
    )

    if commit:
        db.commit()
        db.refresh(wallet)
        db.refresh(tx)

    return {
        "transaction": _serialize_tx(tx),
        "wallet": _serialize_wallet_row(wallet),
    }


def execute_leave_balance(
    db: Session,
    market: str,
    target_active_balance: Decimal | int | float | str,
    network: str = "INTERNAL",
    split_mode: str = "equal",
    note: str = "",
    commit: bool = True,
) -> dict[str, Any]:
    """1-Click 'Withdraw Excess & Leave $X' ($20 Challenge). Refuses with HTTP 409 if open positions exist."""
    raw_market = (market or "all").strip().lower()
    if raw_market not in {"bybit", "forex", "all"}:
        raise WalletValidationError(f"Unsupported market '{market}'. Must be 'bybit', 'forex', or 'all'.")

    target_total = q6(to_decimal(target_active_balance))
    if target_total <= ZERO:
        raise WalletValidationError("target_active_balance must be greater than zero.")

    markets = ["bybit", "forex"] if raw_market == "all" else [raw_market]
    wallets: dict[str, WalletAccount] = {}

    # Lock wallets and enforce open-position conflict guard FIRST across all target markets
    for m in markets:
        w = lock_wallet_account(db, m)
        market_upper = "FOREX" if m == "forex" else "BYBIT"
        open_cnt = (
            db.query(Trade)
            .filter(Trade.status == "OPEN", Trade.market == market_upper)
            .count()
        )
        if open_cnt > 0:
            raise WalletConflictError(
                f"Close open positions on {m} first before running Leave-${target_total:.2f} reset"
            )
        wallets[m] = w

    # Compute per-market target active balance
    targets: dict[str, Decimal] = {}
    if len(markets) == 1:
        targets[markets[0]] = target_total
    else:
        mode = (split_mode or "equal").strip().lower()
        if mode == "bybit_only":
            targets["bybit"] = target_total
            targets["forex"] = q6(ZERO)
        elif mode == "forex_only":
            targets["bybit"] = q6(ZERO)
            targets["forex"] = target_total
        elif mode == "proportional":
            b_bal = to_decimal(wallets["bybit"].active_balance)
            f_bal = to_decimal(wallets["forex"].active_balance)
            tot = b_bal + f_bal
            if tot > ZERO:
                b_target = q6(target_total * (b_bal / tot))
                targets["bybit"] = b_target
                targets["forex"] = q6(target_total - b_target)
            else:
                half = q6(target_total / Decimal("2"))
                targets["bybit"] = half
                targets["forex"] = q6(target_total - half)
        else:
            half = q6(target_total / Decimal("2"))
            targets["bybit"] = half
            targets["forex"] = q6(target_total - half)

    results: list[dict[str, Any]] = []
    for m in markets:
        w = wallets[m]
        active_before = q6(to_decimal(w.active_balance))
        vault_before = q6(to_decimal(w.vault_balance))
        t_bal = targets[m]

        excess = q6(active_before - t_bal)
        if excess <= ZERO:
            if len(markets) == 1:
                raise WalletValidationError(
                    f"Active balance (${active_before:.2f}) on {m} is already at or below target (${t_bal:.2f})."
                )
            continue

        fees = calculate_withdrawal_fees(m, network, excess, is_deposit=False)
        active_after = t_bal
        vault_after = q6(vault_before + fees.net_amount)

        w.active_balance = active_after
        w.vault_balance = vault_after
        _apply_cashflow_baseline_shift(w, active_before, active_after)

        tx = _record_transaction(
            db=db,
            wallet=w,
            tx_type="LEAVE_BALANCE_CHALLENGE",
            network=fees.network,
            gross_amount=fees.gross_amount,
            network_fee=fees.network_fee,
            fx_spread_fee=fees.fx_spread_fee,
            net_amount=fees.net_amount,
            active_before=active_before,
            active_after=active_after,
            vault_before=vault_before,
            vault_after=vault_after,
            note=note or f"1-click Leave ${t_bal:.2f} Challenge via {fees.network}",
        )
        results.append(
            {
                "market": m,
                "transaction": _serialize_tx(tx),
                "wallet": _serialize_wallet_row(w),
            }
        )

    if not results:
        raise WalletValidationError(
            f"No excess balance above target ${target_total:.2f} available to withdraw."
        )

    if commit:
        db.commit()

    return {
        "status": "ok",
        "market": raw_market,
        "target_active_balance": str(target_total),
        "results": results,
    }


def execute_reset_challenge(
    db: Session,
    market: str,
    target_active_balance: Decimal | int | float | str,
    note: str = "",
    commit: bool = True,
) -> dict[str, Any]:
    """Clean-slate challenge reset: active_balance = target, vault_balance = 0, equity_baseline_day = target."""
    raw_market = (market or "all").strip().lower()
    if raw_market not in {"bybit", "forex", "all"}:
        raise WalletValidationError(f"Unsupported market '{market}'. Must be 'bybit', 'forex', or 'all'.")

    target = q6(to_decimal(target_active_balance))
    if target <= ZERO:
        raise WalletValidationError("target_active_balance must be greater than zero.")

    markets = ["bybit", "forex"] if raw_market == "all" else [raw_market]
    wallets: dict[str, WalletAccount] = {}
    for m in markets:
        w = lock_wallet_account(db, m)
        market_upper = "FOREX" if m == "forex" else "BYBIT"
        open_cnt = (
            db.query(Trade)
            .filter(Trade.status == "OPEN", Trade.market == market_upper)
            .count()
        )
        if open_cnt > 0:
            raise WalletConflictError(f"Close open positions on {m} first")
        wallets[m] = w

    results: list[dict[str, Any]] = []
    for m in markets:
        w = wallets[m]
        active_before = q6(to_decimal(w.active_balance))
        vault_before = q6(to_decimal(w.vault_balance))
        active_after = target
        vault_after = q6(ZERO)

        w.active_balance = active_after
        w.vault_balance = vault_after
        w.equity_baseline_day = active_after
        w.cumulative_net_cashflow_day = q6(ZERO)
        w.updated_at = _utcnow()

        tx = _record_transaction(
            db=db,
            wallet=w,
            tx_type="RESET_CHALLENGE",
            network="INTERNAL",
            gross_amount=target,
            network_fee=q6(ZERO),
            fx_spread_fee=q6(ZERO),
            net_amount=target,
            active_before=active_before,
            active_after=active_after,
            vault_before=vault_before,
            vault_after=vault_after,
            note=note or f"Clean-slate challenge reset to ${target:.2f}",
        )
        results.append(
            {
                "market": m,
                "transaction": _serialize_tx(tx),
                "wallet": _serialize_wallet_row(w),
            }
        )

    if commit:
        db.commit()

    return {
        "status": "ok",
        "market": raw_market,
        "target_active_balance": str(target),
        "results": results,
    }


def update_wallet_settings(
    db: Session,
    market: str,
    updates: dict[str, Any],
    commit: bool = True,
) -> dict[str, Any]:
    """Update realism, leverage, min-lot override, and auto-sweep settings for a market wallet."""
    norm_market = normalize_market(market)
    wallet = lock_wallet_account(db, norm_market)

    if "account_profile" in updates and updates["account_profile"] is not None:
        profile = str(updates["account_profile"]).strip().lower()
        if profile not in ALLOWED_ACCOUNT_PROFILES:
            raise WalletValidationError(
                f"Invalid account_profile '{profile}'. Allowed: {', '.join(sorted(ALLOWED_ACCOUNT_PROFILES))}."
            )
        wallet.account_profile = profile

    if "realism_enabled" in updates and updates["realism_enabled"] is not None:
        wallet.realism_enabled = bool(updates["realism_enabled"])

    if "forex_leverage" in updates and updates["forex_leverage"] is not None:
        lev = int(updates["forex_leverage"])
        if lev < 1 or lev > 2000:
            raise WalletValidationError("forex_leverage must be between 1 and 2000.")
        wallet.forex_leverage = lev

    if "bybit_leverage" in updates and updates["bybit_leverage"] is not None:
        lev = int(updates["bybit_leverage"])
        if lev < 1 or lev > 100:
            raise WalletValidationError("bybit_leverage must be between 1 and 100.")
        wallet.bybit_leverage = lev

    if "allow_min_lot_override" in updates and updates["allow_min_lot_override"] is not None:
        wallet.allow_min_lot_override = bool(updates["allow_min_lot_override"])

    if "auto_sweep_enabled" in updates and updates["auto_sweep_enabled"] is not None:
        wallet.auto_sweep_enabled = bool(updates["auto_sweep_enabled"])

    if "auto_sweep_base_capital" in updates and updates["auto_sweep_base_capital"] is not None:
        base = q6(to_decimal(updates["auto_sweep_base_capital"]))
        if base < ZERO:
            raise WalletValidationError("auto_sweep_base_capital cannot be negative.")
        wallet.auto_sweep_base_capital = base

    if "auto_sweep_threshold" in updates and updates["auto_sweep_threshold"] is not None:
        thresh = q6(to_decimal(updates["auto_sweep_threshold"]))
        if thresh <= ZERO:
            raise WalletValidationError("auto_sweep_threshold must be positive.")
        wallet.auto_sweep_threshold = thresh

    if "auto_sweep_network" in updates and updates["auto_sweep_network"] is not None:
        net = normalize_network(str(updates["auto_sweep_network"]))
        if norm_market not in NETWORK_FEE_SCHEDULE[net].allowed_markets:
            raise WalletValidationError(f"auto_sweep_network '{net}' is not allowed for market '{norm_market}'.")
        wallet.auto_sweep_network = net

    wallet.updated_at = _utcnow()
    if commit:
        db.commit()
        db.refresh(wallet)

    return _serialize_wallet_row(wallet)


# ---------------------------------------------------------------------------
# D. Automatic Profit Skimmer (AUTO_SWEEP) Hook
# ---------------------------------------------------------------------------


def maybe_execute_auto_sweep(
    db: Session,
    market: str,
    trade_net_pnl: Decimal | int | float | str,
) -> Optional[WalletTransaction]:
    """Auto-sweep excess profit into vault immediately after a profitable trade closes (same transaction)."""
    if to_decimal(trade_net_pnl) <= ZERO:
        return None

    norm_market = normalize_market(market)
    wallet = lock_wallet_account(db, norm_market)
    if not wallet.auto_sweep_enabled:
        return None

    metrics = compute_market_margin_metrics(db, wallet)
    withdrawable = metrics["withdrawable_active"]
    base_cap = q6(to_decimal(wallet.auto_sweep_base_capital))
    threshold = q6(to_decimal(wallet.auto_sweep_threshold))

    if withdrawable < q6(base_cap + threshold):
        return None

    excess = q6(withdrawable - base_cap)
    network = str(wallet.auto_sweep_network or "INTERNAL").upper()

    try:
        fees = calculate_withdrawal_fees(norm_market, network, excess, is_deposit=False)
    except WalletValidationError as exc:
        db.add(
            SystemLog(
                level="WARN",
                source="AutoSweep",
                message=f"Auto-sweep skipped on {norm_market}: {exc.message}",
            )
        )
        return None

    active_before = q6(to_decimal(wallet.active_balance))
    vault_before = q6(to_decimal(wallet.vault_balance))
    active_after = q6(active_before - fees.gross_amount)
    vault_after = q6(vault_before + fees.net_amount)

    wallet.active_balance = active_after
    wallet.vault_balance = vault_after
    _apply_cashflow_baseline_shift(wallet, active_before, active_after)

    tx = _record_transaction(
        db=db,
        wallet=wallet,
        tx_type="AUTO_SWEEP",
        network=fees.network,
        gross_amount=fees.gross_amount,
        network_fee=fees.network_fee,
        fx_spread_fee=fees.fx_spread_fee,
        net_amount=fees.net_amount,
        active_before=active_before,
        active_after=active_after,
        vault_before=vault_before,
        vault_after=vault_after,
        note=(
            f"Auto-sweep profit skim above ${base_cap:.2f} base + ${threshold:.2f} threshold "
            f"via {fees.network} (fee ${fees.network_fee + fees.fx_spread_fee:.2f})"
        ),
    )
    db.add(
        SystemLog(
            level="INFO",
            source="AutoSweep",
            message=(
                f"AUTO_SWEEP on {norm_market}: swept ${fees.gross_amount:.2f} -> "
                f"+${fees.net_amount:.2f} Vault (fee ${fees.network_fee + fees.fx_spread_fee:.2f} via {fees.network})."
            ),
        )
    )
    return tx


# ---------------------------------------------------------------------------
# Summary, Transactions, & Rejections Read Queries
# ---------------------------------------------------------------------------


def _serialize_tx(tx: WalletTransaction) -> dict[str, Any]:
    return {
        "id": tx.id,
        "market": tx.market,
        "tx_type": tx.tx_type,
        "network": tx.network,
        "gross_amount": float(tx.gross_amount),
        "network_fee": float(tx.network_fee),
        "fx_spread_fee": float(tx.fx_spread_fee),
        "net_amount": float(tx.net_amount),
        "active_balance_before": float(tx.active_balance_before),
        "active_balance_after": float(tx.active_balance_after),
        "vault_balance_before": float(tx.vault_balance_before),
        "vault_balance_after": float(tx.vault_balance_after),
        "note": tx.note,
        "created_at": tx.created_at.isoformat() if tx.created_at else None,
    }


def _serialize_wallet_row(w: WalletAccount) -> dict[str, Any]:
    return {
        "market": w.market,
        "seed_capital": float(w.seed_capital),
        "active_balance": float(w.active_balance),
        "vault_balance": float(w.vault_balance),
        "equity_baseline_day": float(w.equity_baseline_day),
        "cumulative_net_cashflow_day": float(w.cumulative_net_cashflow_day),
        "account_profile": w.account_profile,
        "realism_enabled": bool(w.realism_enabled),
        "forex_leverage": int(w.forex_leverage),
        "bybit_leverage": int(w.bybit_leverage),
        "allow_min_lot_override": bool(w.allow_min_lot_override),
        "auto_sweep_enabled": bool(w.auto_sweep_enabled),
        "auto_sweep_base_capital": float(w.auto_sweep_base_capital),
        "auto_sweep_threshold": float(w.auto_sweep_threshold),
        "auto_sweep_network": w.auto_sweep_network,
        "updated_at": w.updated_at.isoformat() if w.updated_at else None,
    }


def build_wallet_summary(db: Session) -> dict[str, Any]:
    """Build complete per-market + combined wallet summary with friction breakdown & double-entry reconciliation."""
    bybit_w = lock_wallet_account(db, "bybit")
    forex_w = lock_wallet_account(db, "forex")

    markets_out: dict[str, Any] = {}
    comb_active = ZERO
    comb_vault = ZERO
    comb_used_margin = ZERO
    comb_free_margin = ZERO
    comb_withdrawable = ZERO
    comb_active_equity = ZERO
    comb_baseline = ZERO
    comb_drift = ZERO
    comb_reconciled = True
    comb_friction: dict[str, Decimal] = {
        "entry_fees": ZERO,
        "exit_fees": ZERO,
        "spread_cost": ZERO,
        "slippage_cost": ZERO,
        "funding_swaps": ZERO,
        "withdrawal_fees": ZERO,
        "total_friction": ZERO,
        "gross_pnl_total": ZERO,
        "net_pnl_total": ZERO,
    }

    for w in (bybit_w, forex_w):
        m_metrics = compute_market_margin_metrics(db, w)
        friction, recon = compute_market_friction_and_reconciliation(db, w)
        daily_pnl_usd, daily_pnl_pct = compute_daily_pnl_isolated(
            current_active_equity=m_metrics["active_equity"],
            equity_baseline_day=to_decimal(w.equity_baseline_day),
            fallback_capital=to_decimal(w.active_balance),
        )

        markets_out[w.market] = {
            "market": w.market,
            "active_balance": float(w.active_balance),
            "vault_balance": float(w.vault_balance),
            "seed_capital": float(w.seed_capital),
            "equity_baseline_day": float(w.equity_baseline_day),
            "cumulative_net_cashflow_day": float(w.cumulative_net_cashflow_day),
            "active_equity": float(m_metrics["active_equity"]),
            "daily_pnl_usd": float(round(daily_pnl_usd, 2)),
            "daily_pnl_pct": float(round(daily_pnl_pct, 2)),
            "used_margin": float(m_metrics["used_margin"]),
            "free_margin": float(m_metrics["free_margin"]),
            "margin_level_pct": (
                float(round(m_metrics["margin_level_pct"], 2))
                if m_metrics["margin_level_pct"] is not None
                else None
            ),
            "withdrawable_active": float(m_metrics["withdrawable_active"]),
            "open_positions_count": m_metrics["open_positions_count"],
            "lifetime_friction_breakdown": {k: float(v) for k, v in friction.items()},
            "auto_sweep": {
                "enabled": bool(w.auto_sweep_enabled),
                "base_capital": float(w.auto_sweep_base_capital),
                "threshold": float(w.auto_sweep_threshold),
                "network": w.auto_sweep_network,
            },
            "realism_settings": {
                "account_profile": w.account_profile,
                "realism_enabled": bool(w.realism_enabled),
                "forex_leverage": int(w.forex_leverage),
                "bybit_leverage": int(w.bybit_leverage),
                "allow_min_lot_override": bool(w.allow_min_lot_override),
            },
            "reconciliation": {
                "reconciled": bool(recon["reconciled"]),
                "drift": str(recon["drift"]),
            },
        }

        comb_active += to_decimal(w.active_balance)
        comb_vault += to_decimal(w.vault_balance)
        comb_used_margin += m_metrics["used_margin"]
        comb_free_margin += m_metrics["free_margin"]
        comb_withdrawable += m_metrics["withdrawable_active"]
        comb_active_equity += m_metrics["active_equity"]
        comb_baseline += to_decimal(w.equity_baseline_day)
        comb_drift += recon["drift"]
        comb_reconciled = comb_reconciled and bool(recon["reconciled"])
        for k in comb_friction:
            comb_friction[k] = q6(comb_friction[k] + friction[k])

    comb_daily_usd, comb_daily_pct = compute_daily_pnl_isolated(
        current_active_equity=q6(comb_active_equity),
        equity_baseline_day=q6(comb_baseline),
        fallback_capital=q6(comb_active),
    )
    comb_margin_level = (
        float(round(forex_margin_level_pct(comb_active_equity, comb_used_margin), 2))
        if comb_used_margin > ZERO
        else None
    )

    return {
        "markets": markets_out,
        "combined": {
            "active_balance": float(q6(comb_active)),
            "vault_balance": float(q6(comb_vault)),
            "total_holdings": float(q6(comb_active + comb_vault)),
            "active_equity": float(q6(comb_active_equity)),
            "daily_pnl_usd": float(round(comb_daily_usd, 2)),
            "daily_pnl_pct": float(round(comb_daily_pct, 2)),
            "used_margin": float(q6(comb_used_margin)),
            "free_margin": float(q6(comb_free_margin)),
            "margin_level_pct": comb_margin_level,
            "withdrawable_active": float(q6(comb_withdrawable)),
            "lifetime_friction_breakdown": {k: float(v) for k, v in comb_friction.items()},
            "reconciliation": {
                "reconciled": comb_reconciled and (abs(q6(comb_drift)) <= Decimal("0.00001")),
                "drift": str(q6(comb_drift)),
            },
        },
        "fee_schedule": {
            name: {
                "fixed_fee": float(spec.fixed_fee),
                "fx_spread_pct": float(spec.fx_spread_rate * HUNDRED),
                "min_amount": float(spec.min_amount),
                "allowed_markets": sorted(spec.allowed_markets),
            }
            for name, spec in NETWORK_FEE_SCHEDULE.items()
        },
    }


def list_wallet_transactions(
    db: Session,
    market: Optional[str] = None,
    tx_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return paginated wallet transaction ledger."""
    query = db.query(WalletTransaction)
    if market:
        query = query.filter(WalletTransaction.market == normalize_market(market))
    if tx_type:
        query = query.filter(WalletTransaction.tx_type == tx_type.strip().upper())

    total = query.count()
    rows = (
        query.order_by(WalletTransaction.created_at.desc(), WalletTransaction.id.desc())
        .offset(max(0, offset))
        .limit(min(max(1, limit), 200))
        .all()
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_serialize_tx(r) for r in rows],
    }


def list_execution_rejections(
    db: Session,
    market: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return paginated execution_rejections audit entries."""
    query = db.query(ExecutionRejection)
    if market:
        query = query.filter(ExecutionRejection.market == normalize_market(market))

    total = query.count()
    rows = (
        query.order_by(ExecutionRejection.created_at.desc(), ExecutionRejection.id.desc())
        .offset(max(0, offset))
        .limit(min(max(1, limit), 200))
        .all()
    )
    return {
        "total": total,
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": r.id,
                "market": r.market,
                "symbol": r.symbol,
                "side": r.side,
                "strategy": r.strategy,
                "confluence_score": float(r.confluence_score) if r.confluence_score is not None else None,
                "reason_code": r.reason_code,
                "requested_qty": float(r.requested_qty) if r.requested_qty is not None else None,
                "required_min_qty": float(r.required_min_qty) if r.required_min_qty is not None else None,
                "required_margin": float(r.required_margin) if r.required_margin is not None else None,
                "available_free_margin": (
                    float(r.available_free_margin) if r.available_free_margin is not None else None
                ),
                "details": r.details,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }
