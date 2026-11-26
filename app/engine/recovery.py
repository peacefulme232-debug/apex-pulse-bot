"""Crash-recovery and startup state reconciliation engine for APEX-PULSE."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import BotState, SystemLog, Trade

logger = logging.getLogger("apex_pulse.recovery")


def _fetch_remote_open_symbols(broker_or_router: Any, market_label: str, warnings: list[str]) -> Optional[set[str]]:
    """Safely query active non-paper broker open positions without crashing startup."""
    if broker_or_router is None:
        return None

    active_broker = broker_or_router
    # Support BybitRouter and ForexRouter transparently
    if hasattr(broker_or_router, "is_demo_active"):
        if not broker_or_router.is_demo_active():
            return None
        active_broker = broker_or_router.get_broker()
    elif hasattr(broker_or_router, "is_cloud_active"):
        if not broker_or_router.is_cloud_active():
            return None
        active_broker = broker_or_router.get_broker()
    else:
        # Skip PaperSimBroker if passed directly
        if broker_or_router.__class__.__name__ == "PaperSimBroker":
            return None

    if not hasattr(active_broker, "get_open_positions"):
        return None

    try:
        remote_positions = active_broker.get_open_positions() or []
        return {
            str(pos.get("symbol", "")).strip()
            for pos in remote_positions
            if isinstance(pos, dict) and pos.get("symbol")
        }
    except Exception as exc:
        msg = f"{market_label} startup position cross-check failed (non-fatal): {exc}"
        logger.warning(msg)
        warnings.append(msg)
        return None


def reconcile_startup_state(
    db: Session,
    bybit_broker: Any = None,
    forex_broker: Any = None,
) -> dict[str, Any]:
    """Reconcile persisted DB state and open positions once on startup before any tick.

    Guarantees:
    - Strictly preserves persisted kill_switch_active=True across restarts/redeploys.
    - Audits all OPEN trades for valid stop_loss and duplicate symbols.
    - Cross-checks active non-paper brokers when connected without crashing on API errors.
    - Persists an audit summary entry in system_logs.
    """
    warnings: list[str] = []
    broker_mismatches: list[str] = []

    state = db.query(BotState).first()
    kill_switch_preserved = False
    if state is not None and state.kill_switch_active:
        # Never allow a restart or cold start to clear an engaged Kill Switch.
        state.kill_switch_active = True
        state.is_running = False
        kill_switch_preserved = True
        warnings.append(
            f"Persisted Kill Switch lock preserved on startup "
            f"(reason: {state.kill_switch_reason or 'Previously engaged'})."
        )

    open_trades = db.query(Trade).filter(Trade.status == "OPEN").order_by(Trade.id.asc()).all()
    missing_sl_count = 0
    seen_symbols: set[str] = set()
    duplicate_symbols: list[str] = []
    db_bybit_symbols: set[str] = set()
    db_forex_symbols: set[str] = set()

    for trade in open_trades:
        if trade.market == "BYBIT":
            db_bybit_symbols.add(trade.symbol)
        elif trade.market == "FOREX":
            db_forex_symbols.add(trade.symbol)

        if trade.stop_loss is None or float(trade.stop_loss) <= 0.0:
            missing_sl_count += 1
            warnings.append(
                f"OPEN trade #{trade.id} ({trade.symbol} {trade.side}) has invalid/missing stop_loss={trade.stop_loss}."
            )

        if trade.symbol in seen_symbols:
            duplicate_symbols.append(trade.symbol)
            warnings.append(f"Duplicate OPEN DB trade detected on {trade.symbol} (#{trade.id}).")
        else:
            seen_symbols.add(trade.symbol)

    remote_bybit = _fetch_remote_open_symbols(bybit_broker, "Bybit", warnings)
    if remote_bybit is not None:
        only_in_db = sorted(db_bybit_symbols - remote_bybit)
        only_on_broker = sorted(remote_bybit - db_bybit_symbols)
        if only_in_db:
            msg = f"Bybit mismatch: DB OPEN not on broker: {', '.join(only_in_db)}"
            broker_mismatches.append(msg)
            warnings.append(msg)
        if only_on_broker:
            msg = f"Bybit mismatch: Broker OPEN not in DB: {', '.join(only_on_broker)}"
            broker_mismatches.append(msg)
            warnings.append(msg)

    remote_forex = _fetch_remote_open_symbols(forex_broker, "Forex", warnings)
    if remote_forex is not None:
        only_in_db = sorted(db_forex_symbols - remote_forex)
        only_on_broker = sorted(remote_forex - db_forex_symbols)
        if only_in_db:
            msg = f"Forex mismatch: DB OPEN not on broker: {', '.join(only_in_db)}"
            broker_mismatches.append(msg)
            warnings.append(msg)
        if only_on_broker:
            msg = f"Forex mismatch: Broker OPEN not in DB: {', '.join(only_on_broker)}"
            broker_mismatches.append(msg)
            warnings.append(msg)

    if missing_sl_count > 0:
        level = "CRITICAL"
    elif warnings:
        level = "WARN"
    else:
        level = "INFO"

    summary_parts = [
        f"Startup recovery complete: audited {len(open_trades)} OPEN trade(s)",
        f"kill_switch_preserved={kill_switch_preserved}",
        f"missing_sl={missing_sl_count}",
        f"mismatches={len(broker_mismatches)}",
    ]
    if warnings:
        summary_parts.append("Details: " + " | ".join(warnings))

    db.add(
        SystemLog(
            level=level,
            source="StartupRecovery",
            message="; ".join(summary_parts),
        )
    )
    db.commit()

    return {
        "kill_switch_preserved": kill_switch_preserved,
        "open_trades_audited": len(open_trades),
        "missing_sl_count": missing_sl_count,
        "duplicate_symbols": duplicate_symbols,
        "broker_mismatches": broker_mismatches,
        "warnings": warnings,
        "level": level,
    }
