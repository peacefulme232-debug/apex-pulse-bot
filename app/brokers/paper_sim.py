"""Realistic PaperSimBroker for Bybit Crypto + Forex simulation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.brokers.base import BaseBroker
from app.config import settings
from app.database import SessionLocal
from app.models import BotState, EquitySnapshot, StrategyConfig, SystemLog, Trade

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
    """Simulated multi-market broker backed by SQLAlchemy persistence."""

    def __init__(self, session_factory: Callable[[], Session] = SessionLocal) -> None:
        self._session_factory = session_factory
        self._prices: dict[str, float] = {
            sym: spec["base_price"] for sym, spec in SYMBOL_SPECS.items()
        }

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _infer_market(symbol: str) -> str:
        if symbol in SYMBOL_SPECS:
            return str(SYMBOL_SPECS[symbol]["market"])
        return "FOREX" if "_" in symbol else "BYBIT"

    @staticmethod
    def _calc_pnl(side: str, entry_price: float, current_price: float, qty: float) -> tuple[float, float]:
        direction = 1.0 if side.upper() == "LONG" else -1.0
        pnl_usd = (current_price - entry_price) * qty * direction
        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0 * direction if entry_price else 0.0
        return round(pnl_usd, 2), round(pnl_pct, 2)

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
        """Compute current Bybit, Forex, and Total equity from initial balances + realized/unrealized PnL."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            trades = session.query(Trade).all()
            bybit_pnl = sum(t.pnl_usd for t in trades if t.market == "BYBIT")
            forex_pnl = sum(t.pnl_usd for t in trades if t.market == "FOREX")
            bybit_equity = round(INITIAL_BYBIT_BALANCE + bybit_pnl, 2)
            forex_equity = round(INITIAL_FOREX_BALANCE + forex_pnl, 2)
            total_equity = round(bybit_equity + forex_equity, 2)
            return {
                "bybit_equity": bybit_equity,
                "forex_equity": forex_equity,
                "total_equity": total_equity,
            }
        finally:
            if owns_session:
                session.close()

    def get_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 50) -> list[dict[str, Any]]:
        """Generate realistic simulated OHLCV candles around the current price."""
        spec = SYMBOL_SPECS.get(symbol, {"base_price": 100.0, "volatility": 0.0015, "decimals": 2})
        price = self._prices.get(symbol, float(spec["base_price"]))
        vol = float(spec["volatility"])
        decimals = int(spec["decimals"])

        candles: list[dict[str, Any]] = []
        now = self._utcnow()
        cursor = price * (1.0 - vol * 3)
        for idx in range(limit):
            open_p = round(cursor, decimals)
            drift = random.uniform(-vol, vol * 1.05)
            close_p = round(open_p * (1.0 + drift), decimals)
            high_p = round(max(open_p, close_p) * (1.0 + random.uniform(0.0, vol * 0.6)), decimals)
            low_p = round(min(open_p, close_p) * (1.0 - random.uniform(0.0, vol * 0.6)), decimals)
            candles.append(
                {
                    "timestamp": (now - timedelta(minutes=15 * (limit - idx))).isoformat(),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": round(random.uniform(50.0, 500.0), 2),
                }
            )
            cursor = close_p
        return candles

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
    ) -> dict[str, Any]:
        """Persist a new OPEN paper trade."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            market = self._infer_market(symbol)
            spec = SYMBOL_SPECS.get(symbol, {"base_price": 100.0, "decimals": 2})
            entry_price = round(self._prices.get(symbol, float(spec["base_price"])), int(spec["decimals"]))
            trade = Trade(
                market=market,
                symbol=symbol,
                side=side.upper(),
                strategy_name=strategy_name,
                status="OPEN",
                qty=qty,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                pnl_usd=0.0,
                pnl_pct=0.0,
                entry_reason=reason,
                opened_at=self._utcnow(),
            )
            session.add(trade)
            session.add(
                SystemLog(
                    level="INFO",
                    source="PaperSimBroker",
                    message=f"Opened {trade.side} {trade.symbol} ({trade.strategy_name}) @ {entry_price}.",
                )
            )
            session.commit()
            session.refresh(trade)
            return self._trade_to_dict(trade)
        finally:
            if owns_session:
                session.close()

    def close_position(self, trade_id: int, reason: str, db: Session | None = None) -> dict[str, Any]:
        """Close an OPEN paper trade and update strategy telemetry."""
        owns_session = db is None
        session = db or self._session_factory()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade is None:
                raise ValueError(f"Trade {trade_id} not found")
            if trade.status == "CLOSED":
                return self._trade_to_dict(trade)

            exit_price = trade.current_price or trade.entry_price
            pnl_usd, pnl_pct = self._calc_pnl(trade.side, trade.entry_price, exit_price, trade.qty)
            trade.status = "CLOSED"
            trade.exit_price = exit_price
            trade.pnl_usd = pnl_usd
            trade.pnl_pct = pnl_pct
            trade.closed_at = self._utcnow()

            strat = session.query(StrategyConfig).filter(StrategyConfig.name == trade.strategy_name).first()
            if strat:
                prev_wins = round((strat.win_rate / 100.0) * strat.total_trades)
                strat.total_trades += 1
                new_wins = prev_wins + (1 if pnl_usd > 0 else 0)
                strat.win_rate = round((new_wins / strat.total_trades) * 100.0, 1)
                strat.net_pnl_usd = round(strat.net_pnl_usd + pnl_usd, 2)

            session.add(
                SystemLog(
                    level="INFO",
                    source="PaperSimBroker",
                    message=f"Closed {trade.symbol} #{trade.id} ({reason}) PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%).",
                )
            )
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

    def tick_market(self, db: Session) -> dict[str, float]:
        """Advance simulated prices, mark open trades to market, and trigger SL/TP exits."""
        for symbol, spec in SYMBOL_SPECS.items():
            vol = float(spec["volatility"])
            decimals = int(spec["decimals"])
            current = self._prices[symbol]
            step = random.uniform(-vol, vol)
            self._prices[symbol] = round(current * (1.0 + step), decimals)

        open_trades = db.query(Trade).filter(Trade.status == "OPEN").all()
        for trade in open_trades:
            spec = SYMBOL_SPECS.get(trade.symbol, {"decimals": 2})
            decimals = int(spec["decimals"])
            live_price = round(self._prices.get(trade.symbol, trade.current_price), decimals)
            trade.current_price = live_price
            pnl_usd, pnl_pct = self._calc_pnl(trade.side, trade.entry_price, live_price, trade.qty)
            trade.pnl_usd = pnl_usd
            trade.pnl_pct = pnl_pct

            hit_sl = (trade.side == "LONG" and live_price <= trade.stop_loss) or (
                trade.side == "SHORT" and live_price >= trade.stop_loss
            )
            hit_tp = (trade.side == "LONG" and live_price >= trade.take_profit) or (
                trade.side == "SHORT" and live_price <= trade.take_profit
            )
            if hit_sl:
                self.close_position(trade.id, "Stop Loss triggered", db=db)
            elif hit_tp:
                self.close_position(trade.id, "Take Profit reached", db=db)

        balances = self.get_balance(db=db)
        starting_equity = INITIAL_BYBIT_BALANCE + INITIAL_FOREX_BALANCE
        daily_pnl_usd = round(balances["total_equity"] - starting_equity, 2)
        daily_pnl_pct = round((daily_pnl_usd / starting_equity) * 100.0, 2)

        db.add(
            EquitySnapshot(
                timestamp=self._utcnow(),
                total_equity=balances["total_equity"],
                bybit_equity=balances["bybit_equity"],
                forex_equity=balances["forex_equity"],
                daily_pnl_usd=daily_pnl_usd,
                daily_pnl_pct=daily_pnl_pct,
            )
        )
        db.commit()
        return balances
