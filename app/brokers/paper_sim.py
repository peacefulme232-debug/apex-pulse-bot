"""Realistic PaperSimBroker for Bybit Crypto + Forex simulation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

import pandas as pd
from sqlalchemy.orm import Session

from app.brokers.base import BaseBroker
from app.config import settings
from app.database import SessionLocal
from app.models import (
    BotState,
    EquitySnapshot,
    ExecutionRejection,
    StrategyConfig,
    SystemLog,
    Trade,
    WalletAccount,
)
from app.services.realism_engine import (
    ONE,
    ZERO,
    OrderSizingResult,
    ForexPositionSnapshot,
    apply_negative_balance_protection,
    bybit_calculate_funding,
    bybit_calculate_margin_and_liquidation,
    bybit_should_liquidate,
    bybit_simulate_fill,
    bybit_size_and_validate_order,
    calculate_net_pnl,
    ensure_utc,
    forex_calculate_rollover_swap,
    forex_evaluate_margin_status,
    forex_required_margin,
    forex_simulate_fill,
    forex_size_and_validate_order,
    forex_units_to_lots,
    q6,
    q8,
    to_decimal,
)

INITIAL_BYBIT_BALANCE = 10000.00
INITIAL_FOREX_BALANCE = 10000.00

SYMBOL_SPECS: dict[str, dict[str, Any]] = {
    "BTCUSDT": {"market": "BYBIT", "base_price": 94500.0, "volatility": 0.0018, "decimals": 2, "qty": 0.04},
    "ETHUSDT": {"market": "BYBIT", "base_price": 3320.0, "volatility": 0.0022, "decimals": 2, "qty": 1.2},
    "EUR_USD": {"market": "FOREX", "base_price": 1.0845, "volatility": 0.0005, "decimals": 5, "qty": 10000.0},
    "XAU_USD": {"market": "FOREX", "base_price": 2652.0, "volatility": 0.0012, "decimals": 2, "qty": 1.5},
}

DEFAULT_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "trend_rider",
        "display_name": "Trend Rider",
        "description": "EMA ribbon momentum follower capturing sustained directional moves.",
        "target_regime": "TRENDING",
        "enabled": True,
        "total_trades": 14,
        "win_rate": 64.3,
        "net_pnl_usd": 312.40,
    },
    {
        "name": "mean_reversion",
        "display_name": "Mean Reversion",
        "description": "RSI + Bollinger band fade engine tuned for range-bound liquidity zones.",
        "target_regime": "RANGE_BOUND",
        "enabled": True,
        "total_trades": 11,
        "win_rate": 72.7,
        "net_pnl_usd": 184.85,
    },
    {
        "name": "volatility_breakout",
        "display_name": "Volatility Breakout",
        "description": "ATR expansion breakout detector for London/NY session & crypto impulses.",
        "target_regime": "HIGH_VOLATILITY",
        "enabled": True,
        "total_trades": 8,
        "win_rate": 50.0,
        "net_pnl_usd": 96.20,
    },
    {
        "name": "safety_standby",
        "display_name": "Safety Standby",
        "description": "Capital preservation mode; halts new entries during erratic whipsaws.",
        "target_regime": "DEFENSIVE",
        "enabled": False,
        "total_trades": 0,
        "win_rate": 0.0,
        "net_pnl_usd": 0.0,
    },
]


class PaperSimBroker(BaseBroker):
    """Simulated multi-market broker backed by SQLAlchemy persistence.

    Ledger Convention (enforced consistently to guarantee zero double-counting):
    1. Entry (`place_order`): Persists `entry_fee`, `spread_cost`, `slippage_cost`,
       `initial_margin`, `maintenance_margin`, and `liquidation_price` on the Trade row.
       Does NOT debit `entry_fee` from `wallet.active_balance` at open; `entry_fee` is
       settled once at close inside `net_pnl`.
    2. Periodic Funding / Swap (`apply_periodic_fees_and_risk_checks`): Immediately debits
       or credits `wallet.active_balance` (with negative-balance protection) whenever an
       8h Bybit funding boundary or 17:00 America/New_York Forex rollover is crossed, and
       accumulates the amount into `trade.funding_or_swap_fee`.
    3. Partial Scale-Out: Credits `partial_pnl` to `wallet.active_balance` and accumulates
       it into `trade.realized_partial_pnl_usd`.
    4. Exit (`close_position`): Computes `net_pnl = gross_pnl - entry_fee - exit_fee - funding_or_swap_fee`
       and settles only the remaining unsettled cash delta
       `realized_delta = net_pnl - realized_partial_pnl_usd + funding_or_swap_fee` onto
       `wallet.active_balance`. Across any complete open -> funding/swap -> close cycle,
       `active_balance_after - active_balance_before == net_pnl` to 6 decimal places.
    """

    def __init__(self, session_factory: Callable[[], Session] = SessionLocal) -> None:
        self._session_factory = session_factory
        self._prices: dict[str, float] = {
            sym: float(spec["base_price"]) for sym, spec in SYMBOL_SPECS.items()
        }
        self._candles: dict[str, list[dict[str, Any]]] = {}
        self._max_history_bars = 320
        for sym in SYMBOL_SPECS:
            self._init_symbol_candles(sym, bars=260)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _infer_market(symbol: str) -> str:
        if symbol in SYMBOL_SPECS:
            return str(SYMBOL_SPECS[symbol]["market"])
        return "FOREX" if "_" in symbol else "BYBIT"

    @staticmethod
    def _wallet_market_key(market: str) -> str:
        return "forex" if (market or "").strip().lower() == "forex" else "bybit"

    def get_wallet_account(self, db: Session, market: str) -> WalletAccount:
        """Return the WalletAccount row for market ('bybit' | 'forex'), creating it if absent."""
        key = self._wallet_market_key(market)
        wallet = db.query(WalletAccount).filter(WalletAccount.market == key).first()
        if wallet is None:
            default_bal = (
                Decimal(str(INITIAL_FOREX_BALANCE))
                if key == "forex"
                else Decimal(str(INITIAL_BYBIT_BALANCE))
            )
            wallet = WalletAccount(
                market=key,
                seed_capital=q6(default_bal),
                active_balance=q6(default_bal),
                vault_balance=q6(ZERO),
                equity_baseline_day=q6(default_bal),
                cumulative_net_cashflow_day=q6(ZERO),
                account_profile="mt5_raw_ecn" if key == "forex" else "bybit_vip0_linear",
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

    def get_used_margin(self, db: Session, market: str) -> Decimal:
        """Return total initial margin locked in OPEN trades for the specified market."""
        market_upper = "FOREX" if self._wallet_market_key(market) == "forex" else "BYBIT"
        open_trades = (
            db.query(Trade)
            .filter(Trade.status == "OPEN", Trade.market == market_upper)
            .all()
        )
        return q6(sum((to_decimal(t.initial_margin) for t in open_trades), ZERO))

    def log_execution_rejection(
        self,
        db: Session,
        market: str,
        symbol: str,
        side: str,
        reason_code: str,
        requested_qty: Optional[Decimal | int | float | str] = ZERO,
        required_min_qty: Optional[Decimal | int | float | str] = ZERO,
        required_margin: Optional[Decimal | int | float | str] = ZERO,
        available_free_margin: Optional[Decimal | int | float | str] = ZERO,
        details: str = "",
        strategy: Optional[str] = None,
        confluence_score: Optional[Decimal | int | float | str] = None,
    ) -> ExecutionRejection:
        """Persist an ExecutionRejection audit record using exact Decimal columns."""
        score_dec = (
            to_decimal(confluence_score).quantize(Decimal("0.01"))
            if confluence_score is not None
            else None
        )
        row = ExecutionRejection(
            market=self._wallet_market_key(market),
            symbol=symbol,
            side=(side or "").strip().upper(),
            strategy=strategy,
            confluence_score=score_dec,
            reason_code=reason_code,
            requested_qty=q8(to_decimal(requested_qty if requested_qty is not None else ZERO)),
            required_min_qty=q8(to_decimal(required_min_qty if required_min_qty is not None else ZERO)),
            required_margin=q6(to_decimal(required_margin if required_margin is not None else ZERO)),
            available_free_margin=q6(to_decimal(available_free_margin if available_free_margin is not None else ZERO)),
            details=details,
            created_at=self._utcnow(),
        )
        db.add(row)
        db.flush()
        return row

    def size_and_validate_order(
        self,
        db: Session,
        symbol: str,
        side: str,
        entry_price: Decimal | int | float | str,
        stop_loss: Decimal | int | float | str,
        max_risk_per_trade_pct: Decimal | int | float | str,
        order_entry_type: str = "MARKET",
        volatility_pct: Optional[Decimal | int | float | str] = None,
        strategy: Optional[str] = None,
        confluence_score: Optional[Decimal | int | float | str] = None,
    ) -> OrderSizingResult:
        """Size and validate a proposed trade using ONLY wallet_accounts.active_balance (never vault_balance)."""
        market = self._infer_market(symbol)
        wallet = self.get_wallet_account(db, market)
        used_margin = self.get_used_margin(db, market)
        spec = SYMBOL_SPECS.get(symbol, {"volatility": 0.0015})
        vol = to_decimal(volatility_pct) if volatility_pct is not None else to_decimal(spec.get("volatility", 0.0015))

        if market == "BYBIT":
            result = bybit_size_and_validate_order(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                active_balance=to_decimal(wallet.active_balance),
                used_margin=used_margin,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                bybit_leverage=int(wallet.bybit_leverage or 5),
                order_entry_type=order_entry_type,
                allow_min_lot_override=bool(wallet.allow_min_lot_override),
                volatility_pct=vol,
            )
        else:
            result = forex_size_and_validate_order(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                active_balance=to_decimal(wallet.active_balance),
                used_margin=used_margin,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                forex_leverage=int(wallet.forex_leverage or 500),
                account_profile=str(wallet.account_profile or "mt5_raw_ecn"),
                order_entry_type=order_entry_type,
                allow_min_lot_override=bool(wallet.allow_min_lot_override),
                volatility_pct=vol,
            )

        if not result.approved and result.reason_code:
            self.log_execution_rejection(
                db=db,
                market=market,
                symbol=symbol,
                side=side,
                reason_code=result.reason_code,
                requested_qty=result.qty,
                required_min_qty=result.effective_min_qty,
                required_margin=result.required_margin,
                available_free_margin=result.available_free_margin,
                details=result.details,
                strategy=strategy,
                confluence_score=confluence_score,
            )
        return result

    @staticmethod
    def _calc_pnl(side: str, entry_price: float, current_price: float, qty: float) -> tuple[float, float]:
        direction = 1.0 if side.upper() == "LONG" else -1.0
        pnl_usd = (current_price - entry_price) * qty * direction
        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0 * direction if entry_price else 0.0
        return round(pnl_usd, 2), round(pnl_pct, 2)

    @staticmethod
    def _infer_exit_type(reason: str, explicit_exit_type: Optional[str] = None) -> str:
        if explicit_exit_type:
            return explicit_exit_type.strip().upper()
        lower = (reason or "").strip().lower()
        if "liquidation" in lower:
            return "LIQUIDATION"
        if "stop_out" in lower or "stop-out" in lower or "stop out" in lower:
            return "STOP_OUT"
        if "stop loss" in lower or "stop_loss" in lower:
            return "STOP_LOSS"
        if "take profit" in lower or "take_profit" in lower:
            return "TAKE_PROFIT"
        if "limit" in lower:
            return "LIMIT"
        return "MARKET"

    @staticmethod
    def _trade_to_dict(trade: Trade) -> dict[str, Any]:
        return {
            "id": trade.id,
            "market": trade.market,
            "symbol": trade.symbol,
            "side": trade.side,
            "strategy_name": trade.strategy_name,
            "status": trade.status,
            "qty": trade.qty,
            "entry_price": trade.entry_price,
            "current_price": trade.current_price,
            "exit_price": trade.exit_price,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "pnl_usd": round(trade.pnl_usd, 2),
            "pnl_pct": round(trade.pnl_pct, 2),
            "entry_reason": trade.entry_reason,
            "initial_qty": trade.initial_qty if trade.initial_qty is not None else trade.qty,
            "partial_tp_taken": bool(trade.partial_tp_taken),
            "realized_partial_pnl_usd": round(float(trade.realized_partial_pnl_usd or 0.0), 2),
            "order_entry_type": trade.order_entry_type or "MARKET",
            "order_exit_type": trade.order_exit_type,
            "requested_entry_price": float(trade.requested_entry_price) if trade.requested_entry_price is not None else None,
            "filled_entry_price": float(trade.filled_entry_price) if trade.filled_entry_price is not None else None,
            "requested_exit_price": float(trade.requested_exit_price) if trade.requested_exit_price is not None else None,
            "filled_exit_price": float(trade.filled_exit_price) if trade.filled_exit_price is not None else None,
            "initial_margin": float(trade.initial_margin or ZERO),
            "maintenance_margin": float(trade.maintenance_margin or ZERO),
            "liquidation_price": float(trade.liquidation_price) if trade.liquidation_price is not None else None,
            "gross_pnl": float(trade.gross_pnl or ZERO),
            "entry_fee": float(trade.entry_fee or ZERO),
            "exit_fee": float(trade.exit_fee or ZERO),
            "spread_cost": float(trade.spread_cost or ZERO),
            "slippage_cost": float(trade.slippage_cost or ZERO),
            "funding_or_swap_fee": float(trade.funding_or_swap_fee or ZERO),
            "net_pnl": float(trade.net_pnl or ZERO),
            "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
            "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
        }

    def seed_demo_data(self, db: Session) -> None:
        """Seed initial strategies, bot state, equity curve, demo trades, and logs if empty."""
        now = self._utcnow()

        if db.query(BotState).count() == 0:
            db.add(
                BotState(
                    trading_mode=settings.TRADING_MODE,
                    is_running=True,
                    kill_switch_active=False,
                    kill_switch_reason=None,
                    active_regime_bybit="TRENDING_UP",
                    active_regime_forex="RANGE_BOUND",
                    last_heartbeat=now,
                )
            )

        if db.query(StrategyConfig).count() == 0:
            for item in DEFAULT_STRATEGIES:
                db.add(StrategyConfig(**item))

        if db.query(Trade).count() == 0:
            demo_trades = [
                Trade(
                    market="BYBIT",
                    symbol="BTCUSDT",
                    side="LONG",
                    strategy_name="trend_rider",
                    status="OPEN",
                    qty=0.04,
                    entry_price=94120.00,
                    current_price=94580.00,
                    stop_loss=93350.00,
                    take_profit=95900.00,
                    pnl_usd=18.40,
                    pnl_pct=0.49,
                    entry_reason="EMA 20/50 bullish cross with rising spot volume on 15m.",
                    opened_at=now - timedelta(minutes=42),
                ),
                Trade(
                    market="FOREX",
                    symbol="EUR_USD",
                    side="SHORT",
                    strategy_name="mean_reversion",
                    status="OPEN",
                    qty=10000.0,
                    entry_price=1.08620,
                    current_price=1.08460,
                    stop_loss=1.08900,
                    take_profit=1.08150,
                    pnl_usd=16.00,
                    pnl_pct=0.15,
                    entry_reason="Upper Bollinger rejection + RSI(14) divergence at London resistance.",
                    opened_at=now - timedelta(minutes=28),
                ),
                Trade(
                    market="BYBIT",
                    symbol="ETHUSDT",
                    side="LONG",
                    strategy_name="volatility_breakout",
                    status="CLOSED",
                    qty=1.2,
                    entry_price=3245.00,
                    current_price=3318.00,
                    exit_price=3318.00,
                    stop_loss=3205.00,
                    take_profit=3318.00,
                    pnl_usd=87.60,
                    pnl_pct=2.25,
                    entry_reason="ATR(14) range expansion above 4h consolidation ceiling.",
                    opened_at=now - timedelta(hours=4),
                    closed_at=now - timedelta(hours=1, minutes=20),
                ),
                Trade(
                    market="FOREX",
                    symbol="XAU_USD",
                    side="LONG",
                    strategy_name="trend_rider",
                    status="CLOSED",
                    qty=1.5,
                    entry_price=2632.50,
                    current_price=2651.20,
                    exit_price=2651.20,
                    stop_loss=2620.00,
                    take_profit=2651.20,
                    pnl_usd=28.05,
                    pnl_pct=0.71,
                    entry_reason="NY open momentum continuation above VWAP support.",
                    opened_at=now - timedelta(hours=6),
                    closed_at=now - timedelta(hours=2, minutes=45),
                ),
            ]
            db.add_all(demo_trades)

        if db.query(EquitySnapshot).count() == 0:
            points = [
                (6, 10000.0, 10000.0, 0.0, 0.0),
                (5, 10042.5, 10012.0, 54.5, 0.27),
                (4, 10028.0, 10035.0, 63.0, 0.32),
                (3, 10095.4, 10028.0, 123.4, 0.62),
                (2, 10148.2, 10052.1, 200.3, 1.00),
                (1, 10186.0, 10068.5, 254.5, 1.27),
                (0, 10206.0, 10084.0, 290.0, 1.45),
            ]
            for hours_ago, bybit_eq, forex_eq, d_usd, d_pct in points:
                db.add(
                    EquitySnapshot(
                        timestamp=now - timedelta(hours=hours_ago),
                        total_equity=round(bybit_eq + forex_eq, 2),
                        bybit_equity=round(bybit_eq, 2),
                        forex_equity=round(forex_eq, 2),
                        daily_pnl_usd=d_usd,
                        daily_pnl_pct=d_pct,
                    )
                )

        if db.query(SystemLog).count() == 0:
            db.add_all(
                [
                    SystemLog(
                        timestamp=now - timedelta(minutes=5),
                        level="INFO",
                        source="PaperSimBroker",
                        message="Initialized Paper Trading simulation ($10,000 Bybit + $10,000 Forex).",
                    ),
                    SystemLog(
                        timestamp=now - timedelta(minutes=2),
                        level="INFO",
                        source="BotRunner",
                        message="Market regimes synced: BYBIT=TRENDING_UP, FOREX=RANGE_BOUND.",
                    ),
                ]
            )

        db.commit()

    def get_balance(self, db: Session | None = None) -> dict[str, float]:
        """Compute current Bybit, Forex, and Total equity from wallet active_balance + open unrealized PnL."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            bybit_wallet = self.get_wallet_account(session, "bybit")
            forex_wallet = self.get_wallet_account(session, "forex")

            open_trades = session.query(Trade).filter(Trade.status == "OPEN").all()
            bybit_open_pnl = ZERO
            forex_open_pnl = ZERO
            for t in open_trades:
                # Unrealized net PnL not yet credited to active_balance
                t_net = (
                    to_decimal(t.net_pnl)
                    if t.filled_entry_price is not None
                    else to_decimal(t.pnl_usd)
                )
                t_partial = to_decimal(t.realized_partial_pnl_usd)
                t_funding = to_decimal(t.funding_or_swap_fee)
                unrealized = t_net - t_partial + t_funding
                if (t.market or "").upper() == "BYBIT":
                    bybit_open_pnl += unrealized
                else:
                    forex_open_pnl += unrealized

            bybit_equity_dec = q6(to_decimal(bybit_wallet.active_balance) + bybit_open_pnl)
            forex_equity_dec = q6(to_decimal(forex_wallet.active_balance) + forex_open_pnl)
            total_equity_dec = q6(bybit_equity_dec + forex_equity_dec)
            return {
                "bybit_equity": float(round(bybit_equity_dec, 2)),
                "forex_equity": float(round(forex_equity_dec, 2)),
                "total_equity": float(round(total_equity_dec, 2)),
            }
        finally:
            if owns_session:
                session.close()

    def _init_symbol_candles(self, symbol: str, bars: int = 260) -> None:
        """Initialize a realistic historical OHLCV buffer anchored to the symbol's current price."""
        spec = SYMBOL_SPECS.get(symbol, {"base_price": 100.0, "volatility": 0.0015, "decimals": 2})
        target_price = self._prices.get(symbol, float(spec["base_price"]))
        vol = float(spec["volatility"])
        decimals = int(spec["decimals"])

        # Generate a realistic mean-reverting path ending cleanly at target_price
        raw_closes: list[float] = [target_price]
        for _ in range(bars - 1):
            drift = random.uniform(-vol, vol)
            next_p = max(raw_closes[-1] * (1.0 + drift), 1e-6)
            raw_closes.append(next_p)

        scale = target_price / raw_closes[-1]
        closes = [round(max(p * scale, 1e-6), decimals) for p in raw_closes]
        closes[-1] = round(target_price, decimals)

        candles: list[dict[str, Any]] = []
        now = self._utcnow()
        for idx, close_p in enumerate(closes):
            open_p = closes[idx - 1] if idx > 0 else close_p
            high_p = round(max(open_p, close_p) * (1.0 + random.uniform(0.0001, vol * 0.6)), decimals)
            low_p = round(
                max(min(open_p, close_p) * (1.0 - random.uniform(0.0001, vol * 0.6)), 1e-6),
                decimals,
            )
            candles.append(
                {
                    "timestamp": now - timedelta(minutes=15 * (bars - idx)),
                    "open": float(open_p),
                    "high": float(max(high_p, open_p, close_p)),
                    "low": float(min(low_p, open_p, close_p)),
                    "close": float(close_p),
                    "volume": round(random.uniform(50.0, 500.0), 2),
                }
            )

        self._candles[symbol] = candles
        self._prices[symbol] = float(candles[-1]["close"])

    def _evolve_symbol_candle(self, symbol: str, new_close: float) -> None:
        """Append a clean newly evolved candle to the symbol's OHLCV buffer."""
        if symbol not in self._candles or not self._candles[symbol]:
            self._init_symbol_candles(symbol, bars=260)

        spec = SYMBOL_SPECS.get(symbol, {"base_price": 100.0, "volatility": 0.0015, "decimals": 2})
        vol = float(spec["volatility"])
        decimals = int(spec["decimals"])

        history = self._candles[symbol]
        prev = history[-1]
        open_p = float(prev["close"])
        close_p = round(max(float(new_close), 1e-6), decimals)
        high_p = round(max(open_p, close_p) * (1.0 + random.uniform(0.0001, vol * 0.5)), decimals)
        low_p = round(
            max(min(open_p, close_p) * (1.0 - random.uniform(0.0001, vol * 0.5)), 1e-6),
            decimals,
        )
        next_ts = prev["timestamp"] + timedelta(minutes=15)

        history.append(
            {
                "timestamp": next_ts,
                "open": float(open_p),
                "high": float(max(high_p, open_p, close_p)),
                "low": float(min(low_p, open_p, close_p)),
                "close": float(close_p),
                "volume": round(random.uniform(50.0, 500.0), 2),
            }
        )
        if len(history) > self._max_history_bars:
            del history[:-self._max_history_bars]

    def get_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 250) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame (timestamp, open, high, low, close, volume)."""
        if symbol not in self._candles or len(self._candles[symbol]) < limit:
            self._init_symbol_candles(symbol, bars=max(limit + 10, 260))

        sliced = self._candles[symbol][-limit:]
        return pd.DataFrame(
            sliced,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        ).reset_index(drop=True)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str,
        reason: str,
        db: Session | None = None,
        order_entry_type: str = "MARKET",
        requested_entry_price: Optional[float | Decimal] = None,
        volatility_pct: Optional[float | Decimal] = None,
    ) -> dict[str, Any]:
        """Persist a new OPEN paper trade with exact Decimal realism fills, fees, and margin."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            market = self._infer_market(symbol)
            spec = SYMBOL_SPECS.get(symbol, {"base_price": 100.0, "volatility": 0.0015, "decimals": 2})
            wallet = self.get_wallet_account(session, market)
            mid_price = (
                to_decimal(requested_entry_price)
                if requested_entry_price is not None
                else to_decimal(self._prices.get(symbol, float(spec["base_price"])))
            )
            qty_dec = to_decimal(qty)
            vol = (
                to_decimal(volatility_pct)
                if volatility_pct is not None
                else to_decimal(spec.get("volatility", 0.0015))
            )
            entry_type = (order_entry_type or "MARKET").strip().upper()

            if wallet.realism_enabled:
                if market == "BYBIT":
                    fill = bybit_simulate_fill(
                        symbol=symbol,
                        side=side,
                        is_entry=True,
                        mid_price=mid_price,
                        order_type=entry_type,
                        qty=qty_dec,
                        volatility_pct=vol,
                    )
                    metrics = bybit_calculate_margin_and_liquidation(
                        side=side,
                        entry_price=fill.filled_price,
                        qty=qty_dec,
                        leverage=int(wallet.bybit_leverage or 5),
                        mark_price=fill.filled_price,
                        entry_order_type=entry_type,
                        symbol=symbol,
                    )
                    initial_margin = metrics.initial_margin
                    maintenance_margin = metrics.maintenance_margin
                    liquidation_price: Optional[Decimal] = metrics.liquidation_price
                    required_cash = metrics.required_cash
                else:
                    lots = forex_units_to_lots(symbol, qty_dec)
                    fill = forex_simulate_fill(
                        symbol=symbol,
                        side=side,
                        is_entry=True,
                        mid_price=mid_price,
                        order_type=entry_type,
                        lots=lots,
                        account_profile=str(wallet.account_profile or "mt5_raw_ecn"),
                        volatility_pct=vol,
                    )
                    initial_margin = forex_required_margin(
                        symbol=symbol,
                        lots=lots,
                        fill_price=fill.filled_price,
                        forex_leverage=int(wallet.forex_leverage or 500),
                    )
                    maintenance_margin = q6(initial_margin * Decimal("0.5"))
                    liquidation_price = None
                    required_cash = initial_margin

                used_margin = self.get_used_margin(session, market)
                free_margin = q6(max(ZERO, to_decimal(wallet.active_balance) - used_margin))
                if free_margin < required_cash:
                    details = (
                        f"{symbol} requires ${required_cash:.2f} margin/cash "
                        f"> available free margin ${free_margin:.2f}."
                    )
                    self.log_execution_rejection(
                        db=session,
                        market=market,
                        symbol=symbol,
                        side=side,
                        reason_code="INSUFFICIENT_MARGIN",
                        requested_qty=qty_dec,
                        required_min_qty=qty_dec,
                        required_margin=required_cash,
                        available_free_margin=free_margin,
                        details=details,
                        strategy=strategy_name,
                    )
                    session.commit()
                    raise ValueError(f"INSUFFICIENT_MARGIN: {details}")

                req_entry_px = fill.requested_price
                filled_entry_px = fill.filled_price
                entry_fee = fill.fee
                spread_cost = fill.spread_cost
                slippage_cost = fill.slippage_cost
            else:
                req_entry_px = q8(mid_price)
                filled_entry_px = q8(mid_price)
                initial_margin = q6(ZERO)
                maintenance_margin = q6(ZERO)
                liquidation_price = None
                entry_fee = q6(ZERO)
                spread_cost = q6(ZERO)
                slippage_cost = q6(ZERO)

            net_pnl = calculate_net_pnl(ZERO, entry_fee, ZERO, ZERO)
            entry_price_float = float(filled_entry_px)
            trade = Trade(
                market=market,
                symbol=symbol,
                side=side.upper(),
                strategy_name=strategy_name,
                status="OPEN",
                qty=float(qty_dec),
                initial_qty=float(qty_dec),
                partial_tp_taken=False,
                realized_partial_pnl_usd=0.0,
                entry_price=entry_price_float,
                current_price=float(mid_price),
                stop_loss=stop_loss,
                take_profit=take_profit,
                pnl_usd=float(round(net_pnl, 2)),
                pnl_pct=0.0,
                entry_reason=reason,
                opened_at=self._utcnow(),
                order_entry_type=entry_type,
                order_exit_type=None,
                requested_entry_price=req_entry_px,
                filled_entry_price=filled_entry_px,
                requested_exit_price=None,
                filled_exit_price=None,
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                liquidation_price=liquidation_price,
                gross_pnl=q6(ZERO),
                entry_fee=entry_fee,
                exit_fee=q6(ZERO),
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
                funding_or_swap_fee=q6(ZERO),
                net_pnl=net_pnl,
            )
            session.add(trade)
            session.add(
                SystemLog(
                    level="INFO",
                    source="PaperSimBroker",
                    message=f"Opened {trade.side} {trade.symbol} ({trade.strategy_name}) @ {entry_price_float}.",
                )
            )
            session.commit()
            session.refresh(trade)
            return self._trade_to_dict(trade)
        finally:
            if owns_session:
                session.close()

    def close_position(
        self,
        trade_id: int,
        reason: str,
        db: Session | None = None,
        order_exit_type: Optional[str] = None,
        requested_exit_price: Optional[float | Decimal] = None,
    ) -> dict[str, Any]:
        """Close an OPEN paper trade with exact Decimal realism exit fees, net PnL, and wallet update."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")
            if trade.status == "CLOSED":
                return self._trade_to_dict(trade)

            wallet = self.get_wallet_account(session, trade.market)
            exit_type = self._infer_exit_type(reason, order_exit_type)
            qty_dec = to_decimal(trade.qty)
            entry_px = (
                to_decimal(trade.filled_entry_price)
                if trade.filled_entry_price is not None
                else to_decimal(trade.entry_price)
            )
            req_exit_px = (
                to_decimal(requested_exit_price)
                if requested_exit_price is not None
                else to_decimal(trade.current_price if trade.current_price else trade.entry_price)
            )
            spec = SYMBOL_SPECS.get(trade.symbol, {"volatility": 0.0015})
            vol = to_decimal(spec.get("volatility", 0.0015))

            if wallet.realism_enabled:
                if trade.market == "BYBIT":
                    if exit_type == "LIQUIDATION":
                        metrics = bybit_calculate_margin_and_liquidation(
                            side=trade.side,
                            entry_price=entry_px,
                            qty=qty_dec,
                            leverage=int(wallet.bybit_leverage or 5),
                            mark_price=req_exit_px,
                            entry_order_type=trade.order_entry_type or "MARKET",
                            symbol=trade.symbol,
                        )
                        filled_exit_px = metrics.bankruptcy_price
                        exit_fee = q6(ZERO)  # 0% extra liquidation penalty fee
                        exit_spread = q6(ZERO)
                        exit_slip = q6(ZERO)
                    else:
                        fill = bybit_simulate_fill(
                            symbol=trade.symbol,
                            side=trade.side,
                            is_entry=False,
                            mid_price=req_exit_px,
                            order_type=exit_type,
                            qty=qty_dec,
                            volatility_pct=vol,
                        )
                        filled_exit_px = fill.filled_price
                        exit_fee = fill.fee
                        exit_spread = fill.spread_cost
                        exit_slip = fill.slippage_cost
                else:
                    lots = forex_units_to_lots(trade.symbol, qty_dec)
                    fill = forex_simulate_fill(
                        symbol=trade.symbol,
                        side=trade.side,
                        is_entry=False,
                        mid_price=req_exit_px,
                        order_type=exit_type,
                        lots=lots,
                        account_profile=str(wallet.account_profile or "mt5_raw_ecn"),
                        volatility_pct=vol,
                    )
                    filled_exit_px = fill.filled_price
                    exit_fee = fill.fee
                    exit_spread = fill.spread_cost
                    exit_slip = fill.slippage_cost
            else:
                filled_exit_px = q8(req_exit_px)
                exit_fee = q6(ZERO)
                exit_spread = q6(ZERO)
                exit_slip = q6(ZERO)

            direction = ONE if trade.side.upper() == "LONG" else -ONE
            remaining_gross = q6((filled_exit_px - entry_px) * qty_dec * direction)
            partial_realized = to_decimal(trade.realized_partial_pnl_usd)
            gross_pnl = q6(partial_realized + remaining_gross)

            entry_fee = to_decimal(trade.entry_fee)
            funding_or_swap = to_decimal(trade.funding_or_swap_fee)
            net_pnl = calculate_net_pnl(gross_pnl, entry_fee, exit_fee, funding_or_swap)

            pnl_pct = (
                float(((filled_exit_px - entry_px) / entry_px) * Decimal("100") * direction)
                if entry_px > ZERO
                else 0.0
            )
            pnl_usd_float = float(round(net_pnl, 2))

            trade.status = "CLOSED"
            trade.order_exit_type = exit_type
            trade.requested_exit_price = q8(req_exit_px)
            trade.filled_exit_price = q8(filled_exit_px)
            trade.exit_price = float(filled_exit_px)
            trade.gross_pnl = gross_pnl
            trade.exit_fee = exit_fee
            trade.spread_cost = q6(to_decimal(trade.spread_cost) + exit_spread)
            trade.slippage_cost = q6(to_decimal(trade.slippage_cost) + exit_slip)
            trade.net_pnl = net_pnl
            trade.pnl_usd = pnl_usd_float
            trade.pnl_pct = round(pnl_pct, 2)
            trade.closed_at = self._utcnow()

            # Credit remaining unsettled net PnL (excluding partial PnL & funding/swap already applied to active_balance)
            realized_delta = net_pnl - partial_realized + funding_or_swap
            wallet.active_balance = apply_negative_balance_protection(wallet.active_balance, realized_delta)

            strat = session.query(StrategyConfig).filter(StrategyConfig.name == trade.strategy_name).first()
            if strat:
                prev_wins = round((strat.win_rate / 100.0) * strat.total_trades)
                strat.total_trades += 1
                new_wins = prev_wins + (1 if pnl_usd_float > 0 else 0)
                strat.win_rate = round((new_wins / strat.total_trades) * 100.0, 1)
                strat.net_pnl_usd = round(strat.net_pnl_usd + pnl_usd_float, 2)

            session.add(
                SystemLog(
                    level="WARN" if exit_type in {"LIQUIDATION", "STOP_OUT"} else "INFO",
                    source="PaperSimBroker",
                    message=(
                        f"Closed {trade.symbol} #{trade.id} ({reason} [{exit_type}]) "
                        f"Net PnL: ${pnl_usd_float:+.2f} ({trade.pnl_pct:+.2f}%)."
                    ),
                )
            )
            session.flush()
            if net_pnl > ZERO:
                from app.services.wallet_service import maybe_execute_auto_sweep

                maybe_execute_auto_sweep(db=session, market=trade.market, trade_net_pnl=net_pnl)

            session.commit()
            session.refresh(trade)
            return self._trade_to_dict(trade)
        finally:
            if owns_session:
                session.close()

    def get_open_positions(self, db: Session | None = None) -> list[dict[str, Any]]:
        """Return all currently OPEN trades."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            trades = (
                session.query(Trade)
                .filter(Trade.status == "OPEN")
                .order_by(Trade.opened_at.desc())
                .all()
            )
            return [self._trade_to_dict(t) for t in trades]
        finally:
            if owns_session:
                session.close()

    def apply_periodic_fees_and_risk_checks(
        self,
        db: Session,
        now_utc: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Mark open trades to current_price, apply idempotent 8h funding / NY swap, Bybit liquidation, SL/TP, and Forex stop-out."""
        now = ensure_utc(now_utc) if now_utc is not None else self._utcnow()
        bybit_wallet = self.get_wallet_account(db, "bybit")
        forex_wallet = self.get_wallet_account(db, "forex")

        liquidated_ids: list[int] = []
        stopped_out_ids: list[int] = []

        open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
        for trade in open_trades:
            wallet = bybit_wallet if trade.market == "BYBIT" else forex_wallet
            live_px = to_decimal(trade.current_price if trade.current_price else trade.entry_price)
            entry_px = (
                to_decimal(trade.filled_entry_price)
                if trade.filled_entry_price is not None
                else to_decimal(trade.entry_price)
            )
            qty_dec = to_decimal(trade.qty)
            direction = ONE if trade.side.upper() == "LONG" else -ONE

            if wallet.realism_enabled:
                if trade.market == "BYBIT":
                    funding_res = bybit_calculate_funding(
                        side=trade.side,
                        qty=qty_dec,
                        mark_price=live_px,
                        opened_at=trade.opened_at or now,
                        now_utc=now,
                        last_funding_applied_at=trade.last_funding_applied_at,
                    )
                    if funding_res.periods_applied > 0:
                        trade.funding_or_swap_fee = q6(
                            to_decimal(trade.funding_or_swap_fee) + funding_res.fee_amount
                        )
                        trade.last_funding_applied_at = funding_res.new_last_funding_applied_at
                        wallet.active_balance = apply_negative_balance_protection(
                            wallet.active_balance, -funding_res.fee_amount
                        )

                    # Recompute maintenance margin at current mark price
                    if qty_dec > ZERO and entry_px > ZERO:
                        metrics = bybit_calculate_margin_and_liquidation(
                            side=trade.side,
                            entry_price=entry_px,
                            qty=qty_dec,
                            leverage=int(wallet.bybit_leverage or 5),
                            mark_price=live_px,
                            entry_order_type=trade.order_entry_type or "MARKET",
                            symbol=trade.symbol,
                        )
                        trade.maintenance_margin = metrics.maintenance_margin
                        if trade.liquidation_price is None:
                            trade.liquidation_price = metrics.liquidation_price
                else:
                    lots = forex_units_to_lots(trade.symbol, qty_dec)
                    swap_res = forex_calculate_rollover_swap(
                        symbol=trade.symbol,
                        side=trade.side,
                        lots=lots,
                        opened_at=trade.opened_at or now,
                        now_utc=now,
                        last_swap_applied_date=trade.last_swap_applied_date,
                    )
                    if swap_res.rollover_days_applied > 0:
                        trade.funding_or_swap_fee = q6(
                            to_decimal(trade.funding_or_swap_fee) + swap_res.fee_amount
                        )
                        trade.last_swap_applied_date = swap_res.new_last_swap_applied_date
                        wallet.active_balance = apply_negative_balance_protection(
                            wallet.active_balance, -swap_res.fee_amount
                        )

            remaining_gross = q6((live_px - entry_px) * qty_dec * direction)
            partial_realized = to_decimal(trade.realized_partial_pnl_usd)
            gross_pnl = q6(partial_realized + remaining_gross)
            net_pnl = calculate_net_pnl(
                gross_pnl,
                trade.entry_fee,
                trade.exit_fee,
                trade.funding_or_swap_fee,
            )
            pnl_pct = (
                float(((live_px - entry_px) / entry_px) * Decimal("100") * direction)
                if entry_px > ZERO
                else 0.0
            )
            trade.gross_pnl = gross_pnl
            trade.net_pnl = net_pnl
            trade.pnl_usd = float(round(net_pnl, 2))
            trade.pnl_pct = round(pnl_pct, 2)

            # Bybit Liquidation Check FIRST
            if (
                trade.market == "BYBIT"
                and wallet.realism_enabled
                and bybit_should_liquidate(trade.side, live_px, trade.liquidation_price)
            ):
                self.close_position(
                    trade.id,
                    "Bybit UTA Liquidation triggered",
                    db=db,
                    order_exit_type="LIQUIDATION",
                    requested_exit_price=live_px,
                )
                liquidated_ids.append(trade.id)
                continue

            live_float = float(live_px)
            hit_sl = (trade.side == "LONG" and live_float <= trade.stop_loss) or (
                trade.side == "SHORT" and live_float >= trade.stop_loss
            )
            hit_tp = (trade.side == "LONG" and live_float >= trade.take_profit) or (
                trade.side == "SHORT" and live_float <= trade.take_profit
            )
            if hit_sl:
                self.close_position(
                    trade.id,
                    "Stop Loss triggered",
                    db=db,
                    order_exit_type="STOP_LOSS",
                    requested_exit_price=to_decimal(trade.stop_loss),
                )
            elif hit_tp:
                self.close_position(
                    trade.id,
                    "Take Profit reached",
                    db=db,
                    order_exit_type="TAKE_PROFIT",
                    requested_exit_price=to_decimal(trade.take_profit),
                )

        # Forex Margin Call (<= 100%) & Stop-Out (<= 50%) check
        margin_call = False
        if forex_wallet.realism_enabled:
            open_fx = (
                db.query(Trade)
                .filter(Trade.status == "OPEN", Trade.market == "FOREX")
                .all()
            )
            if open_fx:
                snapshots: list[ForexPositionSnapshot] = []
                total_fx_unrealized = ZERO
                for t in open_fx:
                    lots = forex_units_to_lots(t.symbol, t.qty)
                    t_entry = (
                        to_decimal(t.filled_entry_price)
                        if t.filled_entry_price is not None
                        else to_decimal(t.entry_price)
                    )
                    t_curr = to_decimal(t.current_price if t.current_price else t.entry_price)
                    used_m = (
                        to_decimal(t.initial_margin)
                        if to_decimal(t.initial_margin) > ZERO
                        else forex_required_margin(t.symbol, lots, t_entry, int(forex_wallet.forex_leverage or 500))
                    )
                    t_unreal = (
                        to_decimal(t.net_pnl)
                        - to_decimal(t.realized_partial_pnl_usd)
                        + to_decimal(t.funding_or_swap_fee)
                    )
                    total_fx_unrealized += t_unreal
                    snapshots.append(
                        ForexPositionSnapshot(
                            trade_id=t.id,
                            symbol=t.symbol,
                            side=t.side,
                            lots=lots,
                            entry_price=t_entry,
                            current_price=t_curr,
                            unrealized_pnl=t_unreal,
                            used_margin=used_m,
                        )
                    )
                fx_equity = q6(to_decimal(forex_wallet.active_balance) + total_fx_unrealized)
                status = forex_evaluate_margin_status(
                    forex_equity=fx_equity,
                    open_positions=snapshots,
                    forex_leverage=int(forex_wallet.forex_leverage or 500),
                )
                margin_call = status.margin_call
                if status.margin_call and not status.stop_out:
                    db.add(
                        SystemLog(
                            level="WARN",
                            source="ForexMarginGuard",
                            message=f"Forex Margin Call: margin level {status.margin_level_pct:.2f}% <= 100%.",
                        )
                    )
                if status.stop_out:
                    for close_id in status.positions_to_close:
                        self.close_position(
                            close_id,
                            f"Forex STOP_OUT triggered (margin level {status.margin_level_pct:.2f}% <= 50%)",
                            db=db,
                            order_exit_type="STOP_OUT",
                        )
                        stopped_out_ids.append(close_id)

        return {
            "liquidated_ids": liquidated_ids,
            "stopped_out_ids": stopped_out_ids,
            "forex_margin_call": margin_call,
        }

    def tick_market(self, db: Session, now_utc: Optional[datetime] = None) -> dict[str, float]:
        """Advance simulated prices, mark open trades to market, apply funding/swap, and trigger exits."""
        for symbol, spec in SYMBOL_SPECS.items():
            vol = float(spec["volatility"])
            decimals = int(spec["decimals"])
            current = self._prices[symbol]
            step = random.uniform(-vol, vol)
            new_price = round(max(current * (1.0 + step), 1e-6), decimals)
            self._prices[symbol] = new_price
            self._evolve_symbol_candle(symbol, new_price)

        open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
        for trade in open_trades:
            spec = SYMBOL_SPECS.get(trade.symbol, {"decimals": 2})
            decimals = int(spec["decimals"])
            live_price = round(self._prices.get(trade.symbol, trade.current_price), decimals)
            trade.current_price = live_price

        self.apply_periodic_fees_and_risk_checks(db=db, now_utc=now_utc)

        balances = self.get_balance(db=db)
        bybit_wallet = self.get_wallet_account(db, "bybit")
        forex_wallet = self.get_wallet_account(db, "forex")
        from app.services.wallet_service import compute_daily_pnl_isolated

        total_baseline = q6(
            to_decimal(bybit_wallet.equity_baseline_day) + to_decimal(forex_wallet.equity_baseline_day)
        )
        total_active = q6(
            to_decimal(bybit_wallet.active_balance) + to_decimal(forex_wallet.active_balance)
        )
        daily_pnl_usd_dec, daily_pnl_pct_dec = compute_daily_pnl_isolated(
            current_active_equity=to_decimal(balances["total_equity"]),
            equity_baseline_day=total_baseline,
            fallback_capital=total_active,
        )
        daily_pnl_usd = float(round(daily_pnl_usd_dec, 2))
        daily_pnl_pct = float(round(daily_pnl_pct_dec, 2))

        db.add(
            EquitySnapshot(
                timestamp=ensure_utc(now_utc) if now_utc is not None else self._utcnow(),
                total_equity=balances["total_equity"],
                bybit_equity=balances["bybit_equity"],
                forex_equity=balances["forex_equity"],
                daily_pnl_usd=daily_pnl_usd,
                daily_pnl_pct=daily_pnl_pct,
            )
        )
        db.commit()
        return balances
