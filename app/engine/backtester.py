"""Zero-Lookahead Quantitative Backtesting Engine with pessimistic intrabar execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.engine.confluence import ConfluenceFilter
from app.engine.indicators import MIN_CANDLES, add_all_indicators
from app.engine.risk_manager import RiskManager
from app.engine.strategies import StrategyDispatcher

DEFAULT_ENABLED_STRATEGIES: set[str] = {
    "trend_rider",
    "mean_reversion",
    "volatility_breakout",
}


@dataclass
class BacktestResult:
    """JSON-serializable backtest telemetry and performance report."""

    symbol: str
    interval: str
    data_source: str
    candles_tested: int
    starting_balance: float
    ending_balance: float
    net_pnl_usd: float
    net_pnl_pct: float
    buy_and_hold_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    break_even_win_rate_pct: float
    profit_factor: float
    expectancy_usd: float
    max_drawdown_pct: float
    best_trade_usd: float
    worst_trade_usd: float
    partial_tp_hits: int
    circuit_breaker_trips: int
    per_strategy_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    recent_trades: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result into a plain JSON-compatible dict."""
        return asdict(self)


@dataclass
class _OpenPosition:
    side: str  # LONG | SHORT
    strategy_name: str
    regime: str
    entry_bar_idx: int
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    initial_risk: float
    initial_qty: float
    qty: float
    entry_fee_usd: float
    entry_reason: str
    partial_tp_taken: bool = False
    realized_partial_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0


def _to_utc_datetime(val: Any) -> datetime:
    if isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
    elif isinstance(val, datetime):
        dt = val
    else:
        dt = pd.to_datetime(val, utc=True).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_completed_4h_candles(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Aggregate OHLCV into strictly completed 4H UTC candles and their close-completion epochs.

    A 4H candle covering [bucket_start, bucket_start + 4h) is only visible to bar `i`
    once `bucket_end <= bar_i_timestamp` (zero future leak).
    """
    if df.empty or "timestamp" not in df.columns:
        empty = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        return empty, np.array([], dtype=np.int64)

    work = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    ts = pd.to_datetime(work["timestamp"], utc=True)
    work["bucket"] = ts.dt.floor("4h")

    grouped = (
        work.groupby("bucket", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .rename(columns={"bucket": "timestamp"})
    )

    four_hours_ns = np.int64(4 * 3600 * 1_000_000_000)
    bucket_end_ns = pd.Index(grouped["timestamp"]).asi8 + four_hours_ns
    return grouped, bucket_end_ns


def _downsample_equity_curve(points: list[dict[str, Any]], max_points: int = 200) -> list[dict[str, Any]]:
    """Downsample equity curve to <= max_points while always preserving first and last point."""
    n = len(points)
    if n <= max_points:
        return points
    indices = np.linspace(0, n - 1, num=max_points, dtype=int)
    seen: set[int] = set()
    sampled: list[dict[str, Any]] = []
    for idx in indices:
        i = int(idx)
        if i not in seen:
            seen.add(i)
            sampled.append(points[i])
    return sampled


class Backtester:
    """Fast O(N), strictly zero-lookahead backtesting engine with full risk and partial TP parity."""

    def __init__(
        self,
        dispatcher: Optional[StrategyDispatcher] = None,
        confluence_filter: Optional[ConfluenceFilter] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.dispatcher = dispatcher or StrategyDispatcher()
        self.confluence_filter = confluence_filter or ConfluenceFilter()
        self.risk_manager = risk_manager or RiskManager()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str = "1h",
        starting_balance: float = 10000.0,
        fee_pct: float = 0.0010,
        slippage_pct: float = 0.0005,
        enable_partial_tp: bool = True,
        enable_confluence: bool = True,
        enabled_strategies: set[str] | None = None,
        data_source: str = "HISTORICAL",
    ) -> BacktestResult:
        """Execute a strictly causal bar-by-bar backtest on chronological OHLCV data."""
        start_bal = float(starting_balance)
        active_strategies = (
            set(enabled_strategies)
            if enabled_strategies is not None
            else set(DEFAULT_ENABLED_STRATEGIES)
        )

        if df is None or len(df) < MIN_CANDLES + 2:
            return BacktestResult(
                symbol=symbol,
                interval=interval,
                data_source=data_source,
                candles_tested=0 if df is None else len(df),
                starting_balance=round(start_bal, 2),
                ending_balance=round(start_bal, 2),
                net_pnl_usd=0.0,
                net_pnl_pct=0.0,
                buy_and_hold_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                break_even_win_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_usd=0.0,
                max_drawdown_pct=0.0,
                best_trade_usd=0.0,
                worst_trade_usd=0.0,
                partial_tp_hits=0,
                circuit_breaker_trips=0,
                per_strategy_breakdown={},
                equity_curve=[],
                recent_trades=[],
            )

        work_df = df.reset_index(drop=True).copy()
        if "timestamp" not in work_df.columns:
            base_ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
            work_df["timestamp"] = [base_ts + pd.Timedelta(hours=i) for i in range(len(work_df))]
        else:
            work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)

        # Precompute strictly causal indicators ONCE on the full dataset
        enriched = add_all_indicators(work_df)
        htf_df, htf_end_ns = _build_completed_4h_candles(work_df)

        timestamps_ns = pd.Index(work_df["timestamp"]).asi8
        opens = work_df["open"].astype(float).to_numpy()
        highs = work_df["high"].astype(float).to_numpy()
        lows = work_df["low"].astype(float).to_numpy()
        closes = work_df["close"].astype(float).to_numpy()

        n_bars = len(work_df)
        warmup = MIN_CANDLES
        first_eval_close = float(closes[warmup]) if closes[warmup] > 0 else 1.0

        cash_balance = start_bal
        open_pos: Optional[_OpenPosition] = None
        closed_trades: list[dict[str, Any]] = []
        partial_tp_hits = 0
        circuit_breaker_trips = 0

        last_loss_bar_idx: Optional[int] = None
        current_utc_day: Optional[date] = None
        day_start_equity = start_bal
        day_circuit_tripped = False

        equity_points: list[dict[str, Any]] = []
        peak_equity = start_bal
        max_drawdown_pct = 0.0

        def _mark_to_market(pos: Optional[_OpenPosition], price: float) -> float:
            if pos is None:
                return cash_balance
            direction = 1.0 if pos.side == "LONG" else -1.0
            unrealized = (price - pos.entry_price) * pos.qty * direction
            return cash_balance + unrealized

        def _record_close(
            pos: _OpenPosition,
            bar_idx: int,
            exit_ts: datetime,
            exit_price: float,
            exit_reason: str,
        ) -> None:
            nonlocal cash_balance, open_pos, last_loss_bar_idx
            direction = 1.0 if pos.side == "LONG" else -1.0
            gross_rem = (exit_price - pos.entry_price) * pos.qty * direction
            exit_fee = exit_price * pos.qty * fee_pct
            net_rem = gross_rem - exit_fee
            cash_balance += net_rem
            total_fees = pos.total_fees_usd + exit_fee
            # entry_fee was already deducted from cash_balance on entry; include it in trade net PnL
            total_trade_pnl = pos.realized_partial_pnl_usd + net_rem - pos.entry_fee_usd
            notional = pos.entry_price * pos.initial_qty
            pnl_pct = (total_trade_pnl / notional) * 100.0 if notional > 1e-12 else 0.0

            if total_trade_pnl < 0.0:
                last_loss_bar_idx = bar_idx

            closed_trades.append(
                {
                    "symbol": symbol,
                    "side": pos.side,
                    "strategy_name": pos.strategy_name,
                    "regime": pos.regime,
                    "entry_time": pos.entry_time.isoformat(),
                    "exit_time": exit_ts.isoformat(),
                    "entry_price": round(pos.entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "stop_loss": round(pos.stop_loss, 6),
                    "take_profit": round(pos.take_profit, 6),
                    "initial_qty": round(pos.initial_qty, 6),
                    "partial_tp_taken": pos.partial_tp_taken,
                    "fees_usd": round(total_fees, 4),
                    "pnl_usd": round(total_trade_pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": exit_reason,
                    "entry_reason": pos.entry_reason,
                }
            )
            open_pos = None

        def _process_open_bar(bar_idx: int) -> None:
            """Evaluate intrabar SL/TP (pessimistic tie-breaker), +1.0R partial scale-out, and trailing SL."""
            nonlocal cash_balance, open_pos, partial_tp_hits
            if open_pos is None:
                return

            bar_high = float(highs[bar_idx])
            bar_low = float(lows[bar_idx])
            bar_close = float(closes[bar_idx])
            bar_ts = _to_utc_datetime(work_df.iloc[bar_idx]["timestamp"])

            # 1. Check intrabar High/Low against SL and TP.
            # PESSIMISTIC TIE-BREAKER: If BOTH SL and TP hit in the same candle, ALWAYS assume SL hit first!
            if open_pos.side == "LONG":
                hit_sl = bar_low <= open_pos.stop_loss
                hit_tp = bar_high >= open_pos.take_profit
            else:
                hit_sl = bar_high >= open_pos.stop_loss
                hit_tp = bar_low <= open_pos.take_profit

            if hit_sl:
                reason = "Stop Loss (Breakeven Runner)" if open_pos.partial_tp_taken else "Stop Loss"
                _record_close(open_pos, bar_idx, bar_ts, open_pos.stop_loss, reason)
                return

            if hit_tp:
                _record_close(open_pos, bar_idx, bar_ts, open_pos.take_profit, "Take Profit")
                return

            # 2. +1.0R Partial Take-Profit Scale-Out (only when SL was NOT hit on this bar)
            if enable_partial_tp and not open_pos.partial_tp_taken:
                if open_pos.side == "LONG":
                    target_1r = open_pos.entry_price + open_pos.initial_risk
                    reached_1r = bar_high >= target_1r
                else:
                    target_1r = open_pos.entry_price - open_pos.initial_risk
                    reached_1r = bar_low <= target_1r

                if reached_1r:
                    scale_dec = self.risk_manager.evaluate_partial_scale_out(
                        side=open_pos.side,
                        entry_price=open_pos.entry_price,
                        current_price=target_1r,
                        current_sl=open_pos.stop_loss,
                        initial_risk=open_pos.initial_risk,
                        current_qty=open_pos.qty,
                        partial_already_taken=False,
                    )
                    if scale_dec.should_scale_out:
                        direction = 1.0 if open_pos.side == "LONG" else -1.0
                        gross_partial = (target_1r - open_pos.entry_price) * scale_dec.close_qty * direction
                        partial_fee = target_1r * scale_dec.close_qty * fee_pct
                        net_partial = gross_partial - partial_fee

                        cash_balance += net_partial
                        open_pos.realized_partial_pnl_usd += net_partial
                        open_pos.total_fees_usd += partial_fee
                        open_pos.qty = scale_dec.remaining_qty
                        open_pos.stop_loss = scale_dec.new_stop_loss
                        open_pos.partial_tp_taken = True
                        partial_tp_hits += 1

            # 3. Trailing SL on remaining open size
            new_sl = self.risk_manager.compute_trailing_sl(
                side=open_pos.side,
                entry_price=open_pos.entry_price,
                current_price=bar_close,
                current_sl=open_pos.stop_loss,
                initial_risk=open_pos.initial_risk,
            )
            open_pos.stop_loss = new_sl

        # Main strictly causal bar-by-bar walk from warmup (205) to len(df) - 2
        for i in range(warmup, n_bars - 1):
            bar_ts = _to_utc_datetime(work_df.iloc[i]["timestamp"])
            bar_day = bar_ts.date()

            # Reset UTC daily starting equity at UTC midnight BEFORE processing bar i exits
            if current_utc_day is None or bar_day != current_utc_day:
                current_utc_day = bar_day
                day_start_equity = _mark_to_market(open_pos, float(opens[i]))
                day_circuit_tripped = False

            # Step A: Process open position on bar i (if opened before bar i, or on bar i)
            if open_pos is not None:
                _process_open_bar(i)

            # Step B: Mark-to-market equity and UTC daily drawdown circuit breaker
            current_equity = _mark_to_market(open_pos, float(closes[i]))

            daily_pnl_pct = (
                ((current_equity - day_start_equity) / day_start_equity) * 100.0
                if day_start_equity > 1e-12
                else 0.0
            )
            if daily_pnl_pct <= -abs(self.risk_manager.max_daily_loss_pct):
                if not day_circuit_tripped:
                    circuit_breaker_trips += 1
                    day_circuit_tripped = True

            if current_equity > peak_equity:
                peak_equity = current_equity
            dd_pct = ((peak_equity - current_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd_pct > max_drawdown_pct:
                max_drawdown_pct = dd_pct

            benchmark_equity = start_bal * (float(closes[i]) / first_eval_close)
            equity_points.append(
                {
                    "timestamp": bar_ts.isoformat(),
                    "equity": round(current_equity, 2),
                    "benchmark_equity": round(benchmark_equity, 2),
                }
            )

            # Step C: Evaluate signal on bar i ONLY if flat and circuit breaker not tripped
            if open_pos is not None or day_circuit_tripped:
                continue

            # Bounded window ending strictly at bar i + dummy copy of bar i as forming iloc[-1].
            # Bar i+1 NEVER enters eval_df!
            start_w = max(0, i - 210)
            window = enriched.iloc[start_w : i + 1]
            eval_df = pd.concat([window, enriched.iloc[i : i + 1]], ignore_index=True)

            regime_result, signal = self.dispatcher.evaluate(eval_df, active_strategies)
            if signal.action not in {"BUY", "SELL"}:
                continue

            # Optional MTF/VWAP Confluence Gate using strictly completed 4H candles
            if enable_confluence:
                htf_eval_df: Optional[pd.DataFrame] = None
                if len(htf_end_ns) > 0:
                    completed_count = int(np.searchsorted(htf_end_ns, timestamps_ns[i], side="right"))
                    if completed_count >= 55:
                        htf_start = max(0, completed_count - 70)
                        htf_slice = htf_df.iloc[htf_start:completed_count]
                        htf_eval_df = pd.concat(
                            [htf_slice, htf_df.iloc[completed_count - 1 : completed_count]],
                            ignore_index=True,
                        )
                conf_result = self.confluence_filter.evaluate(signal, eval_df, htf_df=htf_eval_df)
                if not conf_result.passed:
                    continue

            candles_since_loss = (
                (i - last_loss_bar_idx) if last_loss_bar_idx is not None else None
            )
            decision = self.risk_manager.approve_trade(
                side=signal.action,
                symbol=symbol,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                equity=current_equity,
                daily_pnl_pct=daily_pnl_pct,
                open_positions=[],
                candles_since_last_loss=candles_since_loss,
            )
            if decision.trigger_kill_switch:
                if not day_circuit_tripped:
                    circuit_breaker_trips += 1
                    day_circuit_tripped = True
                continue
            if not decision.approved or decision.qty <= 0.0:
                continue

            # Execute entry at OPEN of bar i + 1 with slippage and entry fee
            next_open = float(opens[i + 1])
            next_ts = _to_utc_datetime(work_df.iloc[i + 1]["timestamp"])
            order_side = "LONG" if signal.action == "BUY" else "SHORT"
            stop_dist = abs(float(signal.entry_price) - float(signal.stop_loss))
            reward_dist = abs(float(signal.take_profit) - float(signal.entry_price))

            if order_side == "LONG":
                fill_price = next_open * (1.0 + slippage_pct)
                exec_sl = fill_price - stop_dist
                exec_tp = fill_price + reward_dist
            else:
                fill_price = next_open * (1.0 - slippage_pct)
                exec_sl = fill_price + stop_dist
                exec_tp = fill_price - reward_dist

            if fill_price <= 0.0 or exec_sl <= 0.0 or exec_tp <= 0.0:
                continue

            entry_fee = fill_price * decision.qty * fee_pct
            cash_balance -= entry_fee

            open_pos = _OpenPosition(
                side=order_side,
                strategy_name=signal.strategy_name,
                regime=regime_result.regime.value,
                entry_bar_idx=i + 1,
                entry_time=next_ts,
                entry_price=fill_price,
                stop_loss=exec_sl,
                take_profit=exec_tp,
                initial_risk=stop_dist,
                initial_qty=decision.qty,
                qty=decision.qty,
                entry_fee_usd=entry_fee,
                entry_reason=signal.reason,
                total_fees_usd=entry_fee,
            )

        # Process the final bar (n_bars - 1) if a position is still open
        last_idx = n_bars - 1
        last_ts = _to_utc_datetime(work_df.iloc[last_idx]["timestamp"])
        if open_pos is not None:
            _process_open_bar(last_idx)
        if open_pos is not None:
            _record_close(open_pos, last_idx, last_ts, float(closes[last_idx]), "End of Backtest")

        final_equity = cash_balance
        if final_equity > peak_equity:
            peak_equity = final_equity
        final_dd = ((peak_equity - final_equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if final_dd > max_drawdown_pct:
            max_drawdown_pct = final_dd

        benchmark_final = start_bal * (float(closes[last_idx]) / first_eval_close)
        equity_points.append(
            {
                "timestamp": last_ts.isoformat(),
                "equity": round(final_equity, 2),
                "benchmark_equity": round(benchmark_final, 2),
            }
        )

        # Aggregate metrics
        total_trades = len(closed_trades)
        pnls = [float(t["pnl_usd"]) for t in closed_trades]
        wins = [p for p in pnls if p > 0.0]
        losses = [p for p in pnls if p < 0.0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate_pct = round((winning_trades / total_trades) * 100.0, 2) if total_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss > 1e-12:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0.0:
            profit_factor = round(gross_profit, 2)
        else:
            profit_factor = 0.0

        avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
        if avg_win > 1e-12 and avg_loss > 1e-12:
            avg_rr = avg_win / avg_loss
            break_even_win_rate_pct = round((1.0 / (1.0 + avg_rr)) * 100.0, 2)
        elif avg_win > 1e-12 and avg_loss == 0.0:
            break_even_win_rate_pct = 0.0
        else:
            break_even_win_rate_pct = 50.0 if total_trades > 0 else 0.0

        net_pnl_usd = round(final_equity - start_bal, 2)
        net_pnl_pct = round((net_pnl_usd / start_bal) * 100.0, 2) if start_bal > 0 else 0.0
        buy_and_hold_pct = (
            round(((float(closes[last_idx]) - first_eval_close) / first_eval_close) * 100.0, 2)
            if first_eval_close > 0
            else 0.0
        )
        expectancy_usd = round(sum(pnls) / total_trades, 2) if total_trades > 0 else 0.0
        best_trade_usd = round(max(pnls), 2) if pnls else 0.0
        worst_trade_usd = round(min(pnls), 2) if pnls else 0.0

        # Per-strategy breakdown
        per_strategy: dict[str, dict[str, Any]] = {}
        for t in closed_trades:
            sname = t["strategy_name"]
            bucket = per_strategy.setdefault(
                sname,
                {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "win_rate": 0.0},
            )
            bucket["trades"] += 1
            if t["pnl_usd"] > 0:
                bucket["wins"] += 1
            elif t["pnl_usd"] < 0:
                bucket["losses"] += 1
            bucket["net_pnl"] = round(bucket["net_pnl"] + float(t["pnl_usd"]), 2)

        for bucket in per_strategy.values():
            bucket["win_rate"] = (
                round((bucket["wins"] / bucket["trades"]) * 100.0, 2)
                if bucket["trades"] > 0
                else 0.0
            )

        return BacktestResult(
            symbol=symbol,
            interval=interval,
            data_source=data_source,
            candles_tested=n_bars,
            starting_balance=round(start_bal, 2),
            ending_balance=round(final_equity, 2),
            net_pnl_usd=net_pnl_usd,
            net_pnl_pct=net_pnl_pct,
            buy_and_hold_pct=buy_and_hold_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate_pct,
            break_even_win_rate_pct=break_even_win_rate_pct,
            profit_factor=profit_factor,
            expectancy_usd=expectancy_usd,
            max_drawdown_pct=round(max_drawdown_pct, 2),
            best_trade_usd=best_trade_usd,
            worst_trade_usd=worst_trade_usd,
            partial_tp_hits=partial_tp_hits,
            circuit_breaker_trips=circuit_breaker_trips,
            per_strategy_breakdown=per_strategy,
            equity_curve=_downsample_equity_curve(equity_points, max_points=200),
            recent_trades=closed_trades[-50:],
        )
