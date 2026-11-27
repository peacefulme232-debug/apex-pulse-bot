"""Smart Post-Entry Trade Manager.

Responsibilities:
1. Breakeven + Fees Shield (+1.0R):
   Once unrealized profit reaches +1.0R (configurable) and `breakeven_applied == False`,
   shifts Stop-Loss to `filled_entry_price + exact_round_turn_fee_offset` (LONG) or
   `filled_entry_price - exact_round_turn_fee_offset` (SHORT) and sets `breakeven_applied = True`.
2. Limit Take-Profit vs Stop-Loss Exit Type Enforcement:
   TP closes with `order_exit_type="TAKE_PROFIT"` (Maker 0.020%, 0 slippage on Bybit)
   while SL closes with `order_exit_type="STOP_LOSS"` (Taker 0.055% + slippage).
3. Consecutive Loss Tracker & Auto-Sweep Trigger:
   Increments `wallet_accounts.consecutive_losses` on any `net_pnl < 0` close
   (triggering 24h `circuit_breaker_until` at >= `max_consecutive_losses`, default 2),
   resets `consecutive_losses = 0` on any `net_pnl > 0` close, and calls
   `maybe_execute_auto_sweep` on profitable closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import SystemLog, Trade
from app.services.confluence_evaluator import BrainConfig
from app.services.realism_engine import (
    ZERO,
    bybit_fee_rate,
    estimate_round_turn_friction,
    forex_units_to_lots,
    to_decimal,
)
from app.services.wallet_service import (
    compute_daily_pnl_isolated,
    lock_wallet_account,
    maybe_execute_auto_sweep,
)


@dataclass(frozen=True)
class BreakevenUpdate:
    """Audit record when the +1.0R Breakeven + Fees Shield shifts a trade's Stop-Loss."""

    trade_id: int
    symbol: str
    side: str
    old_stop_loss: float
    new_stop_loss: float
    filled_entry_price: float
    fee_price_offset: float
    round_turn_fees_usd: float
    unrealized_r: float


@dataclass(frozen=True)
class TradeCloseOutcome:
    """Result of updating consecutive-loss state, circuit breaker, and auto-sweep on close."""

    trade_id: int
    market: str
    net_pnl: float
    consecutive_losses: int
    circuit_breaker_triggered: bool
    circuit_breaker_until: Optional[datetime]
    auto_swept: bool


class TradeManager:
    """Post-entry lifecycle manager for open and newly closed trades."""

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        self.config = config or BrainConfig()

    @staticmethod
    def resolve_exit_order_type(reason: str) -> str:
        """Map an exit reason to its exact realism order_exit_type."""
        text = (reason or "").strip().upper()
        if "TAKE PROFIT" in text or "TP" in text:
            return "TAKE_PROFIT"
        if "STOP LOSS" in text or "SL" in text or "BREAKEVEN" in text:
            return "STOP_LOSS"
        if "LIQUIDATION" in text:
            return "LIQUIDATION"
        if "STOP_OUT" in text or "STOP-OUT" in text:
            return "STOP_OUT"
        return "MARKET"

    def compute_round_turn_fee_offset(
        self,
        trade: Trade,
        *,
        leverage: int,
        account_profile: str,
    ) -> tuple[float, float]:
        """Compute `(fee_price_offset, round_turn_fees_usd)` for Breakeven + Fees protection."""
        norm_market = (trade.market or "bybit").strip().lower()
        filled_entry = float(
            trade.filled_entry_price if trade.filled_entry_price is not None else trade.entry_price
        )
        qty = float(trade.qty)
        if qty <= 1e-12 or filled_entry <= 1e-12:
            return 0.0, 0.0

        if norm_market == "bybit":
            entry_fee_usd = (
                float(trade.entry_fee)
                if trade.entry_fee is not None and float(trade.entry_fee) > 0.0
                else filled_entry * qty * float(bybit_fee_rate(trade.order_entry_type or "MARKET"))
            )
            exit_sl_fee_usd = filled_entry * qty * float(bybit_fee_rate("STOP_LOSS"))
            round_turn_fees_usd = entry_fee_usd + exit_sl_fee_usd
            fee_price_offset = round_turn_fees_usd / qty
            return fee_price_offset, round_turn_fees_usd

        # Forex: convert round-turn commission + spread cost into exact quote-price offset
        lots = forex_units_to_lots(trade.symbol, qty)
        sl_ref = float(trade.stop_loss)
        sl_dist = abs(filled_entry - sl_ref)
        if sl_dist <= 1e-12:
            sl_ref = filled_entry * (0.99 if trade.side.upper() == "LONG" else 1.01)
            sl_dist = abs(filled_entry - sl_ref)
        tp_ref = float(trade.take_profit)
        if abs(tp_ref - filled_entry) <= 1e-12:
            tp_ref = filled_entry * (1.02 if trade.side.upper() == "LONG" else 0.98)

        est = estimate_round_turn_friction(
            market="forex",
            symbol=trade.symbol,
            side=trade.side,
            qty=lots,
            entry_price=filled_entry,
            sl_price=sl_ref,
            tp_price=tp_ref,
            leverage=leverage,
            profile=account_profile,
            entry_order_type=trade.order_entry_type or "MARKET",
            tp_order_type="TAKE_PROFIT",
            sl_order_type="STOP_LOSS",
        )
        round_turn_fees_usd = float(est.entry_fee + est.est_exit_fee_sl + est.est_spread_cost)
        gross_sl_loss = float(est.gross_sl_loss)
        if gross_sl_loss > 1e-12:
            usd_per_price_unit = gross_sl_loss / sl_dist
            fee_price_offset = round_turn_fees_usd / usd_per_price_unit
        else:
            fee_price_offset = round_turn_fees_usd / max(qty, 1e-9)
        return fee_price_offset, round_turn_fees_usd

    def apply_breakeven_shield(
        self,
        db: Session,
        trade: Trade,
        *,
        current_price: Optional[float] = None,
        initial_risk: Optional[float] = None,
        breakeven_r_trigger: Optional[float] = None,
        price_decimals: Optional[int] = None,
    ) -> Optional[BreakevenUpdate]:
        """Shift SL to filled_entry +/- exact round-turn fees at +1.0R (strictly once)."""
        if trade.status != "OPEN" or bool(trade.breakeven_applied):
            return None

        filled_entry = float(
            trade.filled_entry_price if trade.filled_entry_price is not None else trade.entry_price
        )
        old_sl = float(trade.stop_loss)
        risk = float(initial_risk) if initial_risk is not None else abs(filled_entry - old_sl)
        if risk <= 1e-12:
            return None

        live_price = float(current_price if current_price is not None else trade.current_price)
        side = str(trade.side).upper()
        unrealized_r = (
            (live_price - filled_entry) / risk
            if side == "LONG"
            else (filled_entry - live_price) / risk
        )

        trigger_r = (
            float(breakeven_r_trigger)
            if breakeven_r_trigger is not None
            else float(self.config.breakeven_r_trigger)
        )
        if unrealized_r < trigger_r - 1e-9:
            return None

        wallet = lock_wallet_account(db, trade.market)
        norm_market = (trade.market or "bybit").strip().lower()
        leverage = (
            int(wallet.bybit_leverage or 5)
            if norm_market == "bybit"
            else int(wallet.forex_leverage or 500)
        )
        profile = str(
            wallet.account_profile
            or ("bybit_vip0_linear" if norm_market == "bybit" else "mt5_raw_ecn")
        )

        fee_offset, round_turn_fees_usd = self.compute_round_turn_fee_offset(
            trade,
            leverage=leverage,
            account_profile=profile,
        )

        if side == "LONG":
            raw_new_sl = filled_entry + fee_offset
            new_sl = max(old_sl, raw_new_sl)
        else:
            raw_new_sl = filled_entry - fee_offset
            new_sl = min(old_sl, raw_new_sl)

        if price_decimals is not None:
            new_sl = round(new_sl, int(price_decimals))

        trade.stop_loss = new_sl
        trade.breakeven_applied = True
        db.add(
            SystemLog(
                level="INFO",
                source="TradeManager",
                message=(
                    f"+{trigger_r:.1f}R Breakeven+Fees Shield on {trade.symbol} #{trade.id} ({side}): "
                    f"SL shifted {old_sl:.6f} -> {new_sl:.6f} (fee offset={fee_offset:.6f}, "
                    f"round-turn fees=${round_turn_fees_usd:.4f})."
                ),
            )
        )
        db.flush()

        return BreakevenUpdate(
            trade_id=trade.id,
            symbol=trade.symbol,
            side=side,
            old_stop_loss=old_sl,
            new_stop_loss=new_sl,
            filled_entry_price=filled_entry,
            fee_price_offset=fee_offset,
            round_turn_fees_usd=round_turn_fees_usd,
            unrealized_r=unrealized_r,
        )

    def manage_open_trades(
        self,
        db: Session,
        *,
        initial_risks: Optional[dict[int, float]] = None,
        symbol_decimals: Optional[dict[str, int]] = None,
    ) -> list[BreakevenUpdate]:
        """Run the Breakeven + Fees Shield across all currently OPEN trades."""
        updates: list[BreakevenUpdate] = []
        open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
        for trade in open_trades:
            risk = initial_risks.get(trade.id) if initial_risks is not None else None
            decimals = (
                symbol_decimals.get(trade.symbol) if symbol_decimals is not None else None
            )
            upd = self.apply_breakeven_shield(
                db,
                trade,
                initial_risk=risk,
                price_decimals=decimals,
            )
            if upd is not None:
                updates.append(upd)
        return updates

    def record_trade_close(
        self,
        db: Session,
        trade: Trade,
        *,
        now_utc: Optional[datetime] = None,
    ) -> TradeCloseOutcome:
        """Update consecutive_losses, enforce 24h circuit breaker, and run auto-sweep on wins."""
        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        wallet = lock_wallet_account(db, trade.market)
        net_pnl_dec = to_decimal(trade.net_pnl)
        if net_pnl_dec == ZERO and trade.pnl_usd:
            net_pnl_dec = to_decimal(trade.pnl_usd)

        breaker_triggered = False
        auto_swept = False

        if net_pnl_dec < ZERO:
            wallet.consecutive_losses = int(wallet.consecutive_losses or 0) + 1
            if wallet.consecutive_losses >= self.config.max_consecutive_losses:
                wallet.circuit_breaker_until = now + timedelta(hours=24)
                breaker_triggered = True
                db.add(
                    SystemLog(
                        level="WARN",
                        source="TradeManager",
                        message=(
                            f"24h Circuit Breaker engaged on {wallet.market}: "
                            f"{wallet.consecutive_losses} consecutive losing trades."
                        ),
                    )
                )
        elif net_pnl_dec > ZERO:
            wallet.consecutive_losses = 0
            sweep_tx = maybe_execute_auto_sweep(
                db=db,
                market=wallet.market,
                trade_net_pnl=net_pnl_dec,
            )
            auto_swept = sweep_tx is not None

        # Also check daily trading loss % from equity_baseline_day (vault-withdrawal isolated)
        _daily_usd, daily_pnl_pct_dec = compute_daily_pnl_isolated(
            current_active_equity=to_decimal(wallet.active_balance),
            equity_baseline_day=to_decimal(wallet.equity_baseline_day),
            fallback_capital=to_decimal(wallet.active_balance),
        )
        if float(daily_pnl_pct_dec) <= -abs(self.config.max_daily_loss_pct):
            wallet.circuit_breaker_until = now + timedelta(hours=24)
            breaker_triggered = True
            db.add(
                SystemLog(
                    level="WARN",
                    source="TradeManager",
                    message=(
                        f"24h Circuit Breaker engaged on {wallet.market}: "
                        f"daily trading loss {float(daily_pnl_pct_dec):.2f}% <= "
                        f"-{self.config.max_daily_loss_pct:.2f}%."
                    ),
                )
            )

        db.flush()
        return TradeCloseOutcome(
            trade_id=trade.id,
            market=wallet.market,
            net_pnl=float(net_pnl_dec),
            consecutive_losses=int(wallet.consecutive_losses or 0),
            circuit_breaker_triggered=breaker_triggered,
            circuit_breaker_until=wallet.circuit_breaker_until,
            auto_swept=auto_swept,
        )
