"""Zero-Lookahead Walk-Forward Out-of-Sample Validator with Real Exchange Friction.

Runs the exact live Strategy Brain:
  RegimeRouter -> 3 Synced Strategies -> 0-100 ConfluenceEvaluator ->
  Realism Engine Sizing & Fill Simulation -> Smart TradeManager (BE+Fees & Circuit Breaker)

Guarantees:
1. Rolling Fold Split (3 or 5 sequential folds: 60% warm-up/IS + 40% strictly unseen OOS).
2. Strict Zero-Lookahead: at unseen candle `t`, ONLY closed candles `[:t+1]` are visible.
3. Conservative Worst-Case Intrabar Tie-Breaker: if both SL and TP lie within the same
   bar's `[low, high]`, SL is assumed to have hit first.
4. Exact Decimal accounting via `app/services/realism_engine.py` (Bybit VIP 0 / MT5 Raw ECN
   fees, spread, slippage, 8h funding / NY rollover swaps, liquidation / stop-out, min qty).
5. Strict Verdict: `PASS_READY_FOR_PAPER` ONLY IF OOS `net_pnl > 0`, `profit_factor >= 1.35`,
   `max_drawdown_pct <= 15.0`, and `trades_executed >= 5`; else `FAIL_DO_NOT_TRADE_LIVE`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
from typing import Any, Literal, Optional, Sequence

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database import Base
from app.models import Trade
from app.services.confluence_evaluator import BrainConfig, ConfluenceEvaluator
from app.services.indicators import Candle, filter_closed_candles
from app.services.realism_engine import (
    HUNDRED,
    ONE,
    ZERO,
    apply_negative_balance_protection,
    bybit_calculate_funding,
    bybit_calculate_margin_and_liquidation,
    bybit_should_liquidate,
    bybit_simulate_fill,
    bybit_size_and_validate_order,
    forex_calculate_rollover_swap,
    forex_evaluate_margin_status,
    forex_gross_pnl,
    forex_required_margin,
    forex_simulate_fill,
    forex_size_and_validate_order,
    forex_units_to_lots,
    q6,
    to_decimal,
)
from app.services.regime_router import RegimeRouter, RegimeState
from app.services.strategies import (
    BreakoutRetestStrategy,
    CandidateSignal,
    EmaPullbackStrategy,
    RsiDivergenceSniperStrategy,
)
from app.services.trade_manager import TradeManager
from app.services.wallet_service import lock_wallet_account


ALL_STRATEGIES: tuple[str, ...] = (
    "EMA_50_PULLBACK",
    "BREAKOUT_RETEST",
    "RSI_DIVERGENCE_SNIPER",
)


@dataclass
class _SimPosition:
    """Internal open simulated position tracked bar-by-bar with Decimal precision."""

    trade_id: int
    symbol: str
    market: str
    strategy: str
    side: Literal["LONG", "SHORT"]
    qty: Decimal  # base units for bybit, lots for forex
    units: Decimal  # base units for bybit, currency units for forex
    filled_entry: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    initial_risk: Decimal
    initial_margin: Decimal
    liquidation_price: Optional[Decimal]
    entry_fee: Decimal
    entry_spread_cost: Decimal
    entry_slippage_cost: Decimal
    funding_or_swap_fee: Decimal
    opened_at: datetime
    last_funding_applied_at: Optional[datetime] = None
    last_swap_applied_date: Optional[Any] = None
    breakeven_applied: bool = False


@dataclass(frozen=True)
class FoldMetrics:
    """Out-of-sample metrics for a single walk-forward fold."""

    fold_index: int
    is_start_index: int
    is_end_index: int
    oos_start_index: int
    oos_end_index: int
    starting_capital: float
    ending_capital: float
    gross_pnl: float
    total_fees_paid: float
    net_pnl: float
    win_rate: float
    profit_factor: float
    expectancy_per_trade: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades_executed: int
    signals_vetoed_count: int
    veto_breakdown: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "is_start_index": self.is_start_index,
            "is_end_index": self.is_end_index,
            "oos_start_index": self.oos_start_index,
            "oos_end_index": self.oos_end_index,
            "starting_capital": round(self.starting_capital, 6),
            "ending_capital": round(self.ending_capital, 6),
            "gross_pnl": round(self.gross_pnl, 6),
            "total_fees_paid": round(self.total_fees_paid, 6),
            "net_pnl": round(self.net_pnl, 6),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy_per_trade": round(self.expectancy_per_trade, 6),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "trades_executed": self.trades_executed,
            "signals_vetoed_count": self.signals_vetoed_count,
            "veto_breakdown": dict(self.veto_breakdown),
        }


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate & per-fold Out-of-Sample Walk-Forward validation report."""

    symbol: str
    market: str
    folds_count: int
    realism_enabled: bool
    verdict: Literal["PASS_READY_FOR_PAPER", "FAIL_DO_NOT_TRADE_LIVE"]
    failing_criteria: list[str]
    starting_capital: float
    ending_capital: float
    gross_pnl: float
    total_fees_paid: float
    net_pnl: float
    win_rate: float
    profit_factor: float
    expectancy_per_trade: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades_executed: int
    signals_vetoed_count: int
    veto_breakdown: dict[str, int]
    folds: list[FoldMetrics]
    equity_curve: list[dict[str, Any]]
    data_source: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "folds_count": self.folds_count,
            "realism_enabled": self.realism_enabled,
            "verdict": self.verdict,
            "failing_criteria": list(self.failing_criteria),
            "starting_capital": round(self.starting_capital, 6),
            "ending_capital": round(self.ending_capital, 6),
            "gross_pnl": round(self.gross_pnl, 6),
            "total_fees_paid": round(self.total_fees_paid, 6),
            "net_pnl": round(self.net_pnl, 6),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy_per_trade": round(self.expectancy_per_trade, 6),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "trades_executed": self.trades_executed,
            "signals_vetoed_count": self.signals_vetoed_count,
            "veto_breakdown": dict(self.veto_breakdown),
            "folds": [f.to_dict() for f in self.folds],
            "equity_curve": list(self.equity_curve),
            "data_source": self.data_source,
        }


def _parse_candle_time(candle: Candle, bar_index: int, base_time: datetime) -> datetime:
    ts = candle.timestamp
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        # Support epoch seconds or milliseconds
        sec = float(ts) / 1000.0 if float(ts) > 1e12 else float(ts)
        return datetime.fromtimestamp(sec, tz=timezone.utc)
    if isinstance(ts, str) and ts.strip():
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return base_time + timedelta(hours=bar_index)


def _resample_causal(candles: Sequence[Candle], group_size: int) -> list[Candle]:
    """Strictly causal non-overlapping OHLCV aggregation (drops incomplete trailing block)."""
    n = len(candles)
    full_groups = n // group_size
    out: list[Candle] = []
    for g in range(full_groups):
        chunk = candles[g * group_size : (g + 1) * group_size]
        out.append(
            Candle(
                open=chunk[0].open,
                high=max(c.high for c in chunk),
                low=min(c.low for c in chunk),
                close=chunk[-1].close,
                volume=sum(c.volume for c in chunk),
                timestamp=chunk[-1].timestamp,
                is_closed=True,
            )
        )
    return out


def _compute_stats(
    trade_net_pnls: Sequence[float],
    equity_points: Sequence[float],
    starting_capital: float,
) -> tuple[float, float, float, float, float]:
    """Return `(win_rate_pct, profit_factor, expectancy, max_drawdown_pct, sharpe_ratio)`."""
    n_trades = len(trade_net_pnls)
    if n_trades == 0:
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0
    else:
        wins = [p for p in trade_net_pnls if p > 0.0]
        losses = [p for p in trade_net_pnls if p < 0.0]
        win_rate = (len(wins) / n_trades) * 100.0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss <= 1e-12:
            profit_factor = 99.99 if gross_win > 0.0 else 0.0
        else:
            profit_factor = gross_win / gross_loss
        expectancy = sum(trade_net_pnls) / n_trades

    peak = starting_capital
    max_dd_pct = 0.0
    for eq in equity_points:
        if eq > peak:
            peak = eq
        if peak > 1e-12:
            dd = ((peak - eq) / peak) * 100.0
            if dd > max_dd_pct:
                max_dd_pct = dd

    # Bar-to-bar return Sharpe ratio (annualized on hourly bars)
    returns: list[float] = []
    for i in range(1, len(equity_points)):
        prev_eq = equity_points[i - 1]
        if prev_eq > 1e-12:
            returns.append((equity_points[i] - prev_eq) / prev_eq)

    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r)
        sharpe = (mean_r / std_r) * math.sqrt(24.0 * 365.0) if std_r > 1e-12 else 0.0
    else:
        sharpe = 0.0

    return win_rate, profit_factor, expectancy, max_dd_pct, sharpe


def compute_rolling_folds(total_bars: int, folds: int = 3) -> list[tuple[int, int, int, int]]:
    """Compute sequential `(is_start, is_end, oos_start, oos_end)` tuples with 60/40 IS/OOS split."""
    if folds not in (3, 5):
        raise ValueError(f"folds must be 3 or 5, got {folds}")
    if total_bars < 60:
        raise ValueError(f"Need at least 60 closed candles for walk-forward, got {total_bars}")

    window_size = int(total_bars / (0.6 + 0.4 * folds))
    is_len = max(20, int(round(window_size * 0.6)))
    oos_len = max(5, int((total_bars - is_len) // folds))

    if is_len + oos_len * folds > total_bars:
        oos_len = max(1, (total_bars - is_len) // folds)

    result: list[tuple[int, int, int, int]] = []
    for f in range(folds):
        oos_start = is_len + f * oos_len
        oos_end = total_bars if f == folds - 1 else (oos_start + oos_len)
        is_start = max(0, oos_start - is_len)
        is_end = oos_start
        if oos_end > oos_start:
            result.append((is_start, is_end, oos_start, oos_end))
    return result


class WalkForwardEngine:
    """Deterministic zero-lookahead Walk-Forward Out-of-Sample Validator."""

    def __init__(
        self,
        *,
        regime_router: Optional[RegimeRouter] = None,
        ema_pullback: Optional[EmaPullbackStrategy] = None,
        breakout_retest: Optional[BreakoutRetestStrategy] = None,
        rsi_divergence_sniper: Optional[RsiDivergenceSniperStrategy] = None,
    ) -> None:
        self.regime_router = regime_router or RegimeRouter()
        self.ema_pullback = ema_pullback or EmaPullbackStrategy()
        self.breakout_retest = breakout_retest or BreakoutRetestStrategy()
        self.rsi_divergence_sniper = rsi_divergence_sniper or RsiDivergenceSniperStrategy()

    def run(
        self,
        *,
        candles_h1: Any,
        symbol: str,
        market: str = "bybit",
        starting_capital: Decimal | float | str = Decimal("20.00"),
        strategies: Optional[Sequence[str]] = None,
        folds: int = 3,
        leverage: Optional[int] = None,
        profile: Optional[str] = None,
        realism_enabled: bool = True,
        min_confluence_score: int = 75,
        max_fee_drag_pct: float = 15.0,
        max_risk_per_trade_pct: float = 1.0,
        allow_min_lot_override: bool = False,
        candles_h4: Optional[Any] = None,
        candles_d1: Optional[Any] = None,
        data_source: str = "synthetic",
        forced_signals: Optional[dict[int, CandidateSignal]] = None,
    ) -> WalkForwardResult:
        """Execute strict bar-by-bar Walk-Forward validation across N rolling OOS folds."""
        closed_h1 = filter_closed_candles(candles_h1)
        fold_ranges = compute_rolling_folds(len(closed_h1), folds=folds)

        norm_market = (market or "bybit").strip().lower()
        if norm_market not in ("bybit", "forex"):
            raise ValueError(f"Unsupported market '{market}'. Expected 'bybit' or 'forex'.")

        resolved_leverage = (
            int(leverage)
            if leverage is not None
            else (5 if norm_market == "bybit" else 500)
        )
        resolved_profile = (
            profile
            if profile is not None
            else ("bybit_vip0_linear" if norm_market == "bybit" else "mt5_raw_ecn")
        )
        enabled_set = {
            s.strip().upper()
            for s in (strategies if strategies is not None else ALL_STRATEGIES)
        }

        brain_cfg = BrainConfig(
            ema_50_pullback_enabled=("EMA_50_PULLBACK" in enabled_set),
            breakout_retest_enabled=("BREAKOUT_RETEST" in enabled_set),
            rsi_divergence_sniper_enabled=("RSI_DIVERGENCE_SNIPER" in enabled_set),
            min_confluence_score=int(min_confluence_score),
            max_fee_drag_pct=float(max_fee_drag_pct),
        )
        evaluator = ConfluenceEvaluator(
            config=brain_cfg,
            divergence_detector=self.rsi_divergence_sniper,
        )
        trade_manager = TradeManager(config=brain_cfg)

        # Isolated ephemeral SQLite DB so stateful Breakout+Retest, TradeManager, and
        # ConfluenceEvaluator execute identically to live production without side effects.
        mem_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=mem_engine)
        MemSession = sessionmaker(autocommit=False, autoflush=False, bind=mem_engine)
        db: Session = MemSession()

        try:
            start_cap_dec = q6(to_decimal(starting_capital))
            balance = start_cap_dec
            wallet = lock_wallet_account(db, norm_market)
            wallet.seed_capital = start_cap_dec
            wallet.active_balance = start_cap_dec
            wallet.equity_baseline_day = start_cap_dec
            wallet.bybit_leverage = resolved_leverage if norm_market == "bybit" else 5
            wallet.forex_leverage = resolved_leverage if norm_market == "forex" else 500
            wallet.account_profile = resolved_profile
            wallet.realism_enabled = bool(realism_enabled)
            wallet.consecutive_losses = 0
            wallet.circuit_breaker_until = None
            db.commit()

            base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
            pre_h4 = filter_closed_candles(candles_h4) if candles_h4 is not None else None
            pre_d1 = filter_closed_candles(candles_d1) if candles_d1 is not None else None

            agg_gross_pnl = ZERO
            agg_total_fees = ZERO
            agg_net_pnl = ZERO
            agg_trade_pnls: list[float] = []
            agg_equity_curve: list[dict[str, Any]] = []
            agg_equity_values: list[float] = [float(balance)]
            agg_vetoes: Counter[str] = Counter()
            agg_vetoed_count = 0
            fold_results: list[FoldMetrics] = []

            for fold_idx, (is_start, is_end, oos_start, oos_end) in enumerate(fold_ranges, start=1):
                fold_start_bal = balance
                fold_gross_pnl = ZERO
                fold_fees_paid = ZERO
                fold_net_pnl = ZERO
                fold_trade_pnls: list[float] = []
                fold_equity_vals: list[float] = [float(balance)]
                fold_vetoes: Counter[str] = Counter()
                fold_vetoed_count = 0

                open_pos: Optional[_SimPosition] = None
                last_sl_bar: Optional[int] = None

                for t in range(oos_start, oos_end):
                    bar = closed_h1[t]
                    bar_time = _parse_candle_time(bar, t, base_time)
                    bar_high = to_decimal(bar.high)
                    bar_low = to_decimal(bar.low)
                    bar_close = to_decimal(bar.close)

                    # -------------------------------------------------------
                    # 1. Manage existing open position on bar `t`
                    # -------------------------------------------------------
                    if open_pos is not None:
                        closed_trade_pnl = self._process_open_position_bar(
                            db=db,
                            pos=open_pos,
                            bar_high=bar_high,
                            bar_low=bar_low,
                            bar_close=bar_close,
                            bar_time=bar_time,
                            balance=balance,
                            leverage=resolved_leverage,
                            profile=resolved_profile,
                            realism_enabled=realism_enabled,
                            trade_manager=trade_manager,
                        )
                        if closed_trade_pnl is not None:
                            gross_p, fees_p, net_p, exit_type = closed_trade_pnl
                            balance = apply_negative_balance_protection(balance, net_p)
                            wallet.active_balance = balance
                            fold_gross_pnl = q6(fold_gross_pnl + gross_p)
                            fold_fees_paid = q6(fold_fees_paid + fees_p)
                            fold_net_pnl = q6(fold_net_pnl + net_p)
                            fold_trade_pnls.append(float(net_p))
                            agg_gross_pnl = q6(agg_gross_pnl + gross_p)
                            agg_total_fees = q6(agg_total_fees + fees_p)
                            agg_net_pnl = q6(agg_net_pnl + net_p)
                            agg_trade_pnls.append(float(net_p))
                            if exit_type == "STOP_LOSS":
                                last_sl_bar = t
                            open_pos = None

                    # -------------------------------------------------------
                    # 2. Evaluate Strategy Brain strictly on closed `[:t+1]`
                    # -------------------------------------------------------
                    if open_pos is None and balance > ZERO:
                        h1_window = closed_h1[max(0, t - 260) : t + 1]
                        if pre_h4 is not None:
                            h4_window = [
                                c
                                for idx_h4, c in enumerate(pre_h4)
                                if _parse_candle_time(c, idx_h4 * 4, base_time) <= bar_time
                            ][-260:]
                        else:
                            h4_window = _resample_causal(closed_h1[: t + 1], 4)[-260:]

                        if pre_d1 is not None:
                            d1_window: Optional[list[Candle]] = [
                                c
                                for idx_d1, c in enumerate(pre_d1)
                                if _parse_candle_time(c, idx_d1 * 24, base_time) <= bar_time
                            ][-260:]
                        else:
                            d1_cand = _resample_causal(closed_h1[: t + 1], 24)
                            d1_window = d1_cand[-260:] if len(d1_cand) >= 50 else None

                        candidate: Optional[CandidateSignal] = None
                        regime_decision = self.regime_router.classify(
                            symbol=symbol,
                            h1_candles=h1_window,
                            h4_candles=h4_window if h4_window else h1_window,
                            d1_candles=d1_window,
                        )

                        if forced_signals is not None and t in forced_signals:
                            candidate = forced_signals[t]
                        else:
                            if regime_decision.state == RegimeState.DEAD_CHOP:
                                fold_vetoes["DEAD_CHOP"] += 1
                                agg_vetoes["DEAD_CHOP"] += 1
                            candidate = self._evaluate_strategies(
                                db=db,
                                symbol=symbol,
                                market=norm_market,
                                regime_decision=regime_decision,
                                brain_cfg=brain_cfg,
                                h1_candles=h1_window,
                                h4_candles=h4_window if h4_window else h1_window,
                                d1_candles=d1_window,
                            )

                        if candidate is not None:
                            bars_since_sl = (t - last_sl_bar) if last_sl_bar is not None else None
                            eval_res = evaluator.evaluate(
                                candidate,
                                db=db,
                                h1_candles=h1_window,
                                h4_candles=h4_window if h4_window else h1_window,
                                d1_candles=d1_window,
                                regime_decision=regime_decision,
                                active_balance=balance,
                                current_active_equity=balance,
                                leverage=resolved_leverage,
                                account_profile=resolved_profile,
                                bars_since_last_sl=bars_since_sl,
                                now_utc=bar_time,
                                log_rejection=False,
                            )

                            # When realism_enabled=False, ignore fee-drag veto so zero-fee comparison works
                            effective_vetoes = list(eval_res.vetoes)
                            if not realism_enabled:
                                effective_vetoes = [
                                    v for v in effective_vetoes if v != "FEE_DRAG_TOO_HIGH"
                                ]

                            if effective_vetoes:
                                primary = effective_vetoes[0]
                                fold_vetoed_count += 1
                                agg_vetoed_count += 1
                                fold_vetoes[primary] += 1
                                agg_vetoes[primary] += 1
                            else:
                                pos_or_reason = self._open_position(
                                    db=db,
                                    signal=candidate,
                                    balance=balance,
                                    bar_time=bar_time,
                                    leverage=resolved_leverage,
                                    profile=resolved_profile,
                                    max_risk_per_trade_pct=max_risk_per_trade_pct,
                                    allow_min_lot_override=allow_min_lot_override,
                                    realism_enabled=realism_enabled,
                                )
                                if isinstance(pos_or_reason, str):
                                    fold_vetoed_count += 1
                                    agg_vetoed_count += 1
                                    fold_vetoes[pos_or_reason] += 1
                                    agg_vetoes[pos_or_reason] += 1
                                else:
                                    open_pos = pos_or_reason

                    # Track mark-to-market equity point
                    mtm_equity = float(balance)
                    if open_pos is not None:
                        mtm_equity += float(
                            self._calc_gross_pnl(open_pos, bar_close)
                            - open_pos.entry_fee
                            - open_pos.funding_or_swap_fee
                        )
                    mtm_equity = max(0.0, mtm_equity)
                    fold_equity_vals.append(mtm_equity)
                    agg_equity_values.append(mtm_equity)
                    agg_equity_curve.append(
                        {
                            "bar_index": t,
                            "fold": fold_idx,
                            "timestamp": bar_time.isoformat(),
                            "equity": round(mtm_equity, 6),
                        }
                    )

                # Close any remaining open position at end of fold OOS window (mark-to-market)
                if open_pos is not None:
                    last_bar = closed_h1[oos_end - 1]
                    last_time = _parse_candle_time(last_bar, oos_end - 1, base_time)
                    gross_p, fees_p, net_p = self._close_position(
                        db=db,
                        pos=open_pos,
                        exit_price=to_decimal(last_bar.close),
                        exit_order_type="MARKET",
                        bar_time=last_time,
                        profile=resolved_profile,
                        realism_enabled=realism_enabled,
                        trade_manager=trade_manager,
                    )
                    balance = apply_negative_balance_protection(balance, net_p)
                    wallet.active_balance = balance
                    fold_gross_pnl = q6(fold_gross_pnl + gross_p)
                    fold_fees_paid = q6(fold_fees_paid + fees_p)
                    fold_net_pnl = q6(fold_net_pnl + net_p)
                    fold_trade_pnls.append(float(net_p))
                    agg_gross_pnl = q6(agg_gross_pnl + gross_p)
                    agg_total_fees = q6(agg_total_fees + fees_p)
                    agg_net_pnl = q6(agg_net_pnl + net_p)
                    agg_trade_pnls.append(float(net_p))
                    fold_equity_vals.append(float(balance))
                    agg_equity_values.append(float(balance))
                    open_pos = None

                f_wr, f_pf, f_exp, f_dd, f_sharpe = _compute_stats(
                    fold_trade_pnls,
                    fold_equity_vals,
                    float(fold_start_bal),
                )
                fold_results.append(
                    FoldMetrics(
                        fold_index=fold_idx,
                        is_start_index=is_start,
                        is_end_index=is_end,
                        oos_start_index=oos_start,
                        oos_end_index=oos_end,
                        starting_capital=float(fold_start_bal),
                        ending_capital=float(balance),
                        gross_pnl=float(fold_gross_pnl),
                        total_fees_paid=float(fold_fees_paid),
                        net_pnl=float(fold_net_pnl),
                        win_rate=f_wr,
                        profit_factor=f_pf,
                        expectancy_per_trade=f_exp,
                        max_drawdown_pct=f_dd,
                        sharpe_ratio=f_sharpe,
                        trades_executed=len(fold_trade_pnls),
                        signals_vetoed_count=fold_vetoed_count,
                        veto_breakdown=dict(fold_vetoes),
                    )
                )

            agg_wr, agg_pf, agg_exp, agg_dd, agg_sharpe = _compute_stats(
                agg_trade_pnls,
                agg_equity_values,
                float(start_cap_dec),
            )
            trades_executed = len(agg_trade_pnls)

            failing_criteria: list[str] = []
            if float(agg_net_pnl) <= 0.0:
                failing_criteria.append(f"net_pnl ({float(agg_net_pnl):.4f}) <= 0")
            if agg_pf < 1.35:
                failing_criteria.append(f"profit_factor ({agg_pf:.2f}) < 1.35")
            if agg_dd > 15.0:
                failing_criteria.append(f"max_drawdown_pct ({agg_dd:.2f}%) > 15.0%")
            if trades_executed < 5:
                failing_criteria.append(f"trades_executed ({trades_executed}) < 5")

            verdict: Literal["PASS_READY_FOR_PAPER", "FAIL_DO_NOT_TRADE_LIVE"] = (
                "PASS_READY_FOR_PAPER" if not failing_criteria else "FAIL_DO_NOT_TRADE_LIVE"
            )

            return WalkForwardResult(
                symbol=symbol,
                market=norm_market,
                folds_count=len(fold_results),
                realism_enabled=realism_enabled,
                verdict=verdict,
                failing_criteria=failing_criteria,
                starting_capital=float(start_cap_dec),
                ending_capital=float(balance),
                gross_pnl=float(agg_gross_pnl),
                total_fees_paid=float(agg_total_fees),
                net_pnl=float(agg_net_pnl),
                win_rate=agg_wr,
                profit_factor=agg_pf,
                expectancy_per_trade=agg_exp,
                max_drawdown_pct=agg_dd,
                sharpe_ratio=agg_sharpe,
                trades_executed=trades_executed,
                signals_vetoed_count=agg_vetoed_count,
                veto_breakdown=dict(agg_vetoes),
                folds=fold_results,
                equity_curve=agg_equity_curve,
                data_source=data_source,
            )
        finally:
            db.close()
            mem_engine.dispose()

    # ------------------------------------------------------------------
    # Internal Simulation Helpers
    # ------------------------------------------------------------------

    def _evaluate_strategies(
        self,
        *,
        db: Session,
        symbol: str,
        market: str,
        regime_decision: Any,
        brain_cfg: BrainConfig,
        h1_candles: Sequence[Candle],
        h4_candles: Sequence[Candle],
        d1_candles: Optional[Sequence[Candle]],
    ) -> Optional[CandidateSignal]:
        if brain_cfg.is_strategy_enabled("BREAKOUT_RETEST"):
            armed = self.breakout_retest.get_armed_setup(db, symbol=symbol, market=market)
            if armed is not None:
                sig = self.breakout_retest.evaluate(
                    db=db, symbol=symbol, market=market, candles=h1_candles
                )
                if sig is not None:
                    return sig

        for strat_name in regime_decision.allowed_strategies:
            if not brain_cfg.is_strategy_enabled(strat_name):
                continue
            if strat_name == "EMA_50_PULLBACK":
                sig = self.ema_pullback.evaluate(
                    symbol=symbol,
                    market=market,
                    regime=regime_decision,
                    h1_candles=h1_candles,
                    h4_candles=h4_candles,
                )
            elif strat_name == "BREAKOUT_RETEST":
                sig = self.breakout_retest.evaluate(
                    db=db, symbol=symbol, market=market, candles=h1_candles
                )
            elif strat_name == "RSI_DIVERGENCE_SNIPER":
                htf = d1_candles if d1_candles is not None else h4_candles
                sig = self.rsi_divergence_sniper.evaluate_sniper_entry(
                    candles_htf=htf,
                    candles_exec=h1_candles,
                    symbol=symbol,
                    market=market,
                )
            else:
                sig = None
            if sig is not None:
                return sig
        return None

    def _open_position(
        self,
        *,
        db: Session,
        signal: CandidateSignal,
        balance: Decimal,
        bar_time: datetime,
        leverage: int,
        profile: str,
        max_risk_per_trade_pct: float,
        allow_min_lot_override: bool,
        realism_enabled: bool,
    ) -> _SimPosition | str:
        norm_market = signal.market.strip().lower()
        side = signal.side.upper()
        entry_mid = to_decimal(signal.entry_price)
        sl_px = to_decimal(signal.sl_price)
        tp_px = to_decimal(signal.tp_price)

        if realism_enabled:
            if norm_market == "bybit":
                sizing = bybit_size_and_validate_order(
                    symbol=signal.symbol,
                    side=side,
                    entry_price=entry_mid,
                    stop_loss=sl_px,
                    active_balance=balance,
                    used_margin=ZERO,
                    max_risk_per_trade_pct=max_risk_per_trade_pct,
                    bybit_leverage=leverage,
                    order_entry_type=signal.order_entry_type or "MARKET",
                    allow_min_lot_override=allow_min_lot_override,
                )
                if not sizing.approved:
                    return str(sizing.reason_code or "SIZING_REJECTED")
                qty = sizing.qty
                units = sizing.qty
                fill = bybit_simulate_fill(
                    symbol=signal.symbol,
                    side=side,
                    is_entry=True,
                    mid_price=entry_mid,
                    order_type=signal.order_entry_type or "MARKET",
                    qty=qty,
                )
                margin_info = bybit_calculate_margin_and_liquidation(
                    symbol=signal.symbol,
                    side=side,
                    qty=qty,
                    entry_price=fill.filled_price,
                    leverage=leverage,
                    entry_order_type=signal.order_entry_type or "MARKET",
                )
                filled_entry = fill.filled_price
                entry_fee = fill.fee
                spread_cost = fill.spread_cost
                slippage_cost = fill.slippage_cost
                init_margin = margin_info.initial_margin
                liq_px: Optional[Decimal] = margin_info.liquidation_price
            else:
                sizing = forex_size_and_validate_order(
                    symbol=signal.symbol,
                    side=side,
                    entry_price=entry_mid,
                    stop_loss=sl_px,
                    active_balance=balance,
                    used_margin=ZERO,
                    max_risk_per_trade_pct=max_risk_per_trade_pct,
                    forex_leverage=leverage,
                    account_profile=profile,
                    order_entry_type=signal.order_entry_type or "MARKET",
                    allow_min_lot_override=allow_min_lot_override,
                )
                if not sizing.approved:
                    return str(sizing.reason_code or "SIZING_REJECTED")
                lots = sizing.lots if sizing.lots is not None else forex_units_to_lots(signal.symbol, sizing.qty)
                qty = lots
                units = sizing.qty
                fill = forex_simulate_fill(
                    symbol=signal.symbol,
                    side=side,
                    is_entry=True,
                    mid_price=entry_mid,
                    order_type=signal.order_entry_type or "MARKET",
                    lots=lots,
                    account_profile=profile,
                )
                filled_entry = fill.filled_price
                entry_fee = fill.fee
                spread_cost = fill.spread_cost
                slippage_cost = fill.slippage_cost
                init_margin = forex_required_margin(signal.symbol, lots, filled_entry, leverage)
                liq_px = None
        else:
            # Zero-friction baseline sizing
            risk_usd = balance * (to_decimal(max_risk_per_trade_pct) / HUNDRED)
            sl_dist = abs(entry_mid - sl_px)
            if sl_dist <= Decimal("1e-12"):
                return "INVALID_SL_DISTANCE"
            if norm_market == "bybit":
                qty = q6(max(risk_usd / sl_dist, Decimal("0.001")))
                units = qty
                init_margin = q6((qty * entry_mid) / to_decimal(max(leverage, 1)))
            else:
                lots = q6(max(Decimal("0.01"), risk_usd / max(sl_dist * Decimal("100000"), Decimal("1"))))
                qty = lots
                units = q6(lots * Decimal("100000"))
                init_margin = forex_required_margin(signal.symbol, lots, entry_mid, leverage)
            filled_entry = entry_mid
            entry_fee = ZERO
            spread_cost = ZERO
            slippage_cost = ZERO
            liq_px = None

        trade_row = Trade(
            market=norm_market.upper(),
            symbol=signal.symbol,
            side=side,
            strategy_name=signal.strategy,
            status="OPEN",
            qty=float(units),
            entry_price=float(filled_entry),
            filled_entry_price=filled_entry,
            current_price=float(filled_entry),
            stop_loss=float(sl_px),
            take_profit=float(tp_px),
            order_entry_type=signal.order_entry_type or "MARKET",
            entry_fee=entry_fee,
            initial_margin=init_margin,
            liquidation_price=liq_px,
            breakeven_applied=False,
            opened_at=bar_time,
        )
        db.add(trade_row)
        db.flush()

        return _SimPosition(
            trade_id=trade_row.id,
            symbol=signal.symbol,
            market=norm_market,
            strategy=signal.strategy,
            side=side,  # type: ignore[arg-type]
            qty=qty,
            units=units,
            filled_entry=filled_entry,
            stop_loss=sl_px,
            take_profit=tp_px,
            initial_risk=abs(filled_entry - sl_px),
            initial_margin=init_margin,
            liquidation_price=liq_px,
            entry_fee=entry_fee,
            entry_spread_cost=spread_cost,
            entry_slippage_cost=slippage_cost,
            funding_or_swap_fee=ZERO,
            opened_at=bar_time,
        )

    @staticmethod
    def _calc_gross_pnl(pos: _SimPosition, price: Decimal) -> Decimal:
        direction = ONE if pos.side == "LONG" else -ONE
        if pos.market == "bybit":
            return q6((price - pos.filled_entry) * pos.qty * direction)
        return q6(forex_gross_pnl(pos.symbol, pos.side, pos.qty, pos.filled_entry, price))

    def _process_open_position_bar(
        self,
        *,
        db: Session,
        pos: _SimPosition,
        bar_high: Decimal,
        bar_low: Decimal,
        bar_close: Decimal,
        bar_time: datetime,
        balance: Decimal,
        leverage: int,
        profile: str,
        realism_enabled: bool,
        trade_manager: TradeManager,
    ) -> Optional[tuple[Decimal, Decimal, Decimal, str]]:
        """Process funding/swap, liquidation/stop-out, conservative SL-before-TP, and BE shield."""
        if realism_enabled:
            if pos.market == "bybit":
                fund = bybit_calculate_funding(
                    side=pos.side,
                    qty=pos.qty,
                    mark_price=bar_close,
                    opened_at=pos.opened_at,
                    now_utc=bar_time,
                    last_funding_applied_at=pos.last_funding_applied_at,
                )
                if fund.periods_applied > 0:
                    pos.funding_or_swap_fee = q6(pos.funding_or_swap_fee + fund.fee_amount)
                    pos.last_funding_applied_at = fund.new_last_funding_applied_at
            else:
                swap = forex_calculate_rollover_swap(
                    symbol=pos.symbol,
                    side=pos.side,
                    lots=pos.qty,
                    opened_at=pos.opened_at,
                    now_utc=bar_time,
                    last_swap_applied_date=pos.last_swap_applied_date,
                )
                if swap.days_applied > 0:
                    pos.funding_or_swap_fee = q6(pos.funding_or_swap_fee + swap.fee_amount)
                    pos.last_swap_applied_date = swap.new_last_swap_applied_date

            # Liquidation / Stop-Out check FIRST
            worst_px = bar_low if pos.side == "LONG" else bar_high
            if pos.market == "bybit" and bybit_should_liquidate(pos.side, worst_px, pos.liquidation_price):
                liq_exit = pos.liquidation_price if pos.liquidation_price is not None else worst_px
                g, f, n = self._close_position(
                    db=db,
                    pos=pos,
                    exit_price=liq_exit,
                    exit_order_type="LIQUIDATION",
                    bar_time=bar_time,
                    profile=profile,
                    realism_enabled=True,
                    trade_manager=trade_manager,
                )
                return g, f, n, "LIQUIDATION"

            if pos.market == "forex":
                worst_unreal = self._calc_gross_pnl(pos, worst_px) - pos.entry_fee - pos.funding_or_swap_fee
                status = forex_evaluate_margin_status(
                    active_balance=balance,
                    total_unrealized_pnl=worst_unreal,
                    used_margin=pos.initial_margin,
                )
                if status.stop_out_triggered:
                    g, f, n = self._close_position(
                        db=db,
                        pos=pos,
                        exit_price=worst_px,
                        exit_order_type="STOP_OUT",
                        bar_time=bar_time,
                        profile=profile,
                        realism_enabled=True,
                        trade_manager=trade_manager,
                    )
                    return g, f, n, "STOP_OUT"

        # Conservative Worst-Case Tie-Break: check SL BEFORE TP when both sit inside [low, high]
        hit_sl = (pos.side == "LONG" and bar_low <= pos.stop_loss) or (
            pos.side == "SHORT" and bar_high >= pos.stop_loss
        )
        hit_tp = (pos.side == "LONG" and bar_high >= pos.take_profit) or (
            pos.side == "SHORT" and bar_low <= pos.take_profit
        )

        if hit_sl:
            g, f, n = self._close_position(
                db=db,
                pos=pos,
                exit_price=pos.stop_loss,
                exit_order_type="STOP_LOSS",
                bar_time=bar_time,
                profile=profile,
                realism_enabled=realism_enabled,
                trade_manager=trade_manager,
            )
            return g, f, n, "STOP_LOSS"

        if hit_tp:
            g, f, n = self._close_position(
                db=db,
                pos=pos,
                exit_price=pos.take_profit,
                exit_order_type="TAKE_PROFIT",
                bar_time=bar_time,
                profile=profile,
                realism_enabled=realism_enabled,
                trade_manager=trade_manager,
            )
            return g, f, n, "TAKE_PROFIT"

        # Surviving bar: check +1.0R Breakeven + Fees Shield via TradeManager
        if not pos.breakeven_applied:
            trade_row = db.query(Trade).filter(Trade.id == pos.trade_id).first()
            if trade_row is not None:
                trade_row.current_price = float(bar_close)
                be_upd = trade_manager.apply_breakeven_shield(
                    db=db,
                    trade=trade_row,
                    current_price=float(bar_close),
                    initial_risk=float(pos.initial_risk),
                )
                if be_upd is not None:
                    pos.stop_loss = to_decimal(be_upd.new_stop_loss)
                    pos.breakeven_applied = True

        return None

    def _close_position(
        self,
        *,
        db: Session,
        pos: _SimPosition,
        exit_price: Decimal,
        exit_order_type: str,
        bar_time: datetime,
        profile: str,
        realism_enabled: bool,
        trade_manager: TradeManager,
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Close simulated position, compute exact gross/fees/net PnL, and notify TradeManager."""
        if realism_enabled:
            if pos.market == "bybit":
                fill = bybit_simulate_fill(
                    symbol=pos.symbol,
                    side=pos.side,
                    is_entry=False,
                    mid_price=exit_price,
                    order_type=exit_order_type,
                    qty=pos.qty,
                )
            else:
                fill = forex_simulate_fill(
                    symbol=pos.symbol,
                    side=pos.side,
                    is_entry=False,
                    mid_price=exit_price,
                    order_type=exit_order_type,
                    lots=pos.qty,
                    account_profile=profile,
                )
            filled_exit = fill.filled_price
            exit_fee = fill.fee
            exit_spread_cost = fill.spread_cost
            exit_slippage_cost = fill.slippage_cost
        else:
            filled_exit = exit_price
            exit_fee = ZERO
            exit_spread_cost = ZERO
            exit_slippage_cost = ZERO

        # Gross PnL measured on requested mid prices vs filled exit so friction is cleanly itemized
        gross_pnl = self._calc_gross_pnl(pos, exit_price)
        filled_gross = self._calc_gross_pnl(pos, filled_exit)
        execution_drag = q6(max(ZERO, gross_pnl - filled_gross))
        explicit_fees = q6(pos.entry_fee + exit_fee + pos.funding_or_swap_fee)
        total_fees_paid = q6(
            explicit_fees + max(execution_drag, pos.entry_spread_cost + exit_spread_cost + pos.entry_slippage_cost + exit_slippage_cost)
        )
        net_pnl = q6(filled_gross - explicit_fees)

        trade_row = db.query(Trade).filter(Trade.id == pos.trade_id).first()
        if trade_row is not None:
            trade_row.status = "CLOSED"
            trade_row.exit_price = float(filled_exit)
            trade_row.filled_exit_price = filled_exit
            trade_row.order_exit_type = exit_order_type
            trade_row.gross_pnl = gross_pnl
            trade_row.exit_fee = exit_fee
            trade_row.funding_or_swap_fee = pos.funding_or_swap_fee
            trade_row.net_pnl = net_pnl
            trade_row.pnl_usd = float(net_pnl)
            trade_row.closed_at = bar_time
            db.flush()
            trade_manager.record_trade_close(db, trade_row, now_utc=bar_time)

        return gross_pnl, total_fees_paid, net_pnl
