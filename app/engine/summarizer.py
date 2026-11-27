"""Plain-English Performance, Accuracy, Edge Math, and Risk Shield Summarizer Engine."""

from __future__ import annotations

from typing import Any, Iterable, Optional

from sqlalchemy.orm import Session

from app.brokers.paper_sim import INITIAL_BYBIT_BALANCE, INITIAL_FOREX_BALANCE
from app.config import settings
from app.models import BotState, StrategyConfig, SystemLog, Trade
from app.services.realism_engine import to_decimal

STRATEGY_REGIME_FALLBACK: dict[str, str] = {
    "trend_rider": "TRENDING",
    "range_mean_reversion": "RANGE_BOUND",
    "volatility_breakout": "HIGH_VOLATILITY",
    "safety_standby": "DEFENSIVE",
    "EMA_50_PULLBACK": "TRENDING_BULL",
    "BREAKOUT_RETEST": "BREAKOUT_EXPANSION",
    "RSI_DIVERGENCE_SNIPER": "REVERSAL_EXHAUSTION",
}

BRAIN_STRATEGIES: tuple[tuple[str, str, str], ...] = (
    ("EMA_50_PULLBACK", "EMA 50 Pullback", "TRENDING_BULL"),
    ("BREAKOUT_RETEST", "Breakout + Retest", "BREAKOUT_EXPANSION"),
    ("RSI_DIVERGENCE_SNIPER", "RSI Divergence Sniper", "REVERSAL_EXHAUSTION"),
)


class BotSummarizer:
    """Computes quantitative edge metrics and translates them into beginner-friendly plain English."""

    @staticmethod
    def _compute_trade_gross_fees_net(trades: Iterable[Any]) -> tuple[float, float, float]:
        """Return (gross_pnl_usd, fees_paid_usd, net_pnl_usd) across closed trades."""
        gross_total = 0.0
        fees_total = 0.0
        net_total = 0.0
        for t in trades:
            if hasattr(t, "filled_exit_price"):
                entry_fee = float(to_decimal(getattr(t, "entry_fee", 0)))
                exit_fee = float(to_decimal(getattr(t, "exit_fee", 0)))
                spread_cost = float(to_decimal(getattr(t, "spread_cost", 0)))
                slippage_cost = float(to_decimal(getattr(t, "slippage_cost", 0)))
                funding_swap = float(to_decimal(getattr(t, "funding_or_swap_fee", 0)))
                trade_fees = entry_fee + exit_fee + spread_cost + slippage_cost + funding_swap
                if t.filled_exit_price is not None:
                    t_gross = float(to_decimal(getattr(t, "gross_pnl", 0)))
                    t_net = float(to_decimal(getattr(t, "net_pnl", 0)))
                else:
                    t_net = float(getattr(t, "pnl_usd", 0.0) or 0.0)
                    t_gross = t_net + trade_fees
            else:
                trade_fees = float(t.get("fees_paid_usd", 0.0) or 0.0)
                t_net = float(t.get("pnl_usd", 0.0) or 0.0)
                t_gross = float(t.get("gross_pnl_usd", t_net + trade_fees) or 0.0)
            gross_total += t_gross
            fees_total += trade_fees
            net_total += t_net
        return round(gross_total, 2), round(fees_total, 2), round(net_total, 2)

    @staticmethod
    def compute_core_metrics(
        closed_trades: Iterable[Any],
        starting_equity: float = INITIAL_BYBIT_BALANCE + INITIAL_FOREX_BALANCE,
    ) -> dict[str, Any]:
        """Compute accuracy, break-even win rate, edge margin, profit factor, expectancy, drawdown, and grade."""
        trades_list = list(closed_trades)
        total_closed = len(trades_list)

        if total_closed == 0:
            return {
                "total_closed_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "break_even_win_rate_pct": 0.0,
                "edge_margin_pct": 0.0,
                "profit_factor": 0.0,
                "expectancy_usd": 0.0,
                "avg_win_usd": 0.0,
                "avg_loss_usd": 0.0,
                "avg_rr_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "net_pnl_usd": 0.0,
                "health_grade": "CALIBRATING",
            }

        pnls = [
            float(t.pnl_usd if hasattr(t, "pnl_usd") else t.get("pnl_usd", 0.0))
            for t in trades_list
        ]

        wins = [p for p in pnls if p > 0.0]
        losses = [abs(p) for p in pnls if p < 0.0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_closed
        loss_rate = losing_trades / total_closed
        win_rate_pct = win_rate * 100.0

        gross_profit = sum(wins)
        gross_loss = sum(losses)
        net_pnl_usd = sum(pnls)

        avg_win_usd = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss_usd = (gross_loss / losing_trades) if losing_trades > 0 else 0.0

        if gross_loss > 0.0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0.0:
            profit_factor = min(99.99, gross_profit)
        else:
            profit_factor = 0.0

        if avg_loss_usd > 0.0:
            avg_rr_ratio = avg_win_usd / avg_loss_usd
        elif avg_win_usd > 0.0:
            avg_rr_ratio = 2.0
        else:
            avg_rr_ratio = 0.0

        break_even_win_rate_pct = (
            (100.0 / (1.0 + avg_rr_ratio)) if avg_rr_ratio > 0.0 else 50.0
        )
        edge_margin_pct = win_rate_pct - break_even_win_rate_pct
        expectancy_usd = (win_rate * avg_win_usd) - (loss_rate * avg_loss_usd)

        # Peak-to-trough drawdown across closed trades
        equity = float(starting_equity) if starting_equity > 0 else 10000.0
        peak = equity
        max_drawdown_pct = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            if peak > 0.0:
                dd = ((peak - equity) / peak) * 100.0
                if dd > max_drawdown_pct:
                    max_drawdown_pct = dd

        health_grade = BotSummarizer._assign_health_grade(
            total_closed=total_closed,
            profit_factor=profit_factor,
            edge_margin_pct=edge_margin_pct,
            expectancy_usd=expectancy_usd,
            max_drawdown_pct=max_drawdown_pct,
        )

        return {
            "total_closed_trades": total_closed,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate_pct, 2),
            "break_even_win_rate_pct": round(break_even_win_rate_pct, 2),
            "edge_margin_pct": round(edge_margin_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy_usd": round(expectancy_usd, 2),
            "avg_win_usd": round(avg_win_usd, 2),
            "avg_loss_usd": round(avg_loss_usd, 2),
            "avg_rr_ratio": round(avg_rr_ratio, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "net_pnl_usd": round(net_pnl_usd, 2),
            "health_grade": health_grade,
        }

    @staticmethod
    def _assign_health_grade(
        total_closed: int,
        profit_factor: float,
        edge_margin_pct: float,
        expectancy_usd: float,
        max_drawdown_pct: float,
    ) -> str:
        if total_closed < 5:
            return "CALIBRATING"
        if profit_factor >= 2.0 and edge_margin_pct >= 15.0 and expectancy_usd > 0.0 and max_drawdown_pct < 5.0:
            return "A+"
        if profit_factor >= 1.5 and edge_margin_pct >= 8.0 and expectancy_usd > 0.0:
            return "A"
        if profit_factor >= 1.2 and edge_margin_pct >= 0.0 and expectancy_usd > 0.0:
            return "B"
        if profit_factor >= 0.95:
            return "C"
        return "D"

    def build_summary(
        self,
        db: Session,
        market: str = "ALL",
        balances: Optional[dict[str, float]] = None,
        forex_session: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate complete API summary payload for ALL, BYBIT, or FOREX."""
        market_upper = (market or "ALL").strip().upper()
        if market_upper == "BYBIT":
            starting_equity = INITIAL_BYBIT_BALANCE
            selected_equity = float((balances or {}).get("bybit_equity", INITIAL_BYBIT_BALANCE))
        elif market_upper == "FOREX":
            starting_equity = INITIAL_FOREX_BALANCE
            selected_equity = float((balances or {}).get("forex_equity", INITIAL_FOREX_BALANCE))
        else:
            market_upper = "ALL"
            starting_equity = INITIAL_BYBIT_BALANCE + INITIAL_FOREX_BALANCE
            selected_equity = float((balances or {}).get("total_equity", starting_equity))

        closed_q = db.query(Trade).filter(Trade.status == "CLOSED")
        if market_upper != "ALL":
            closed_q = closed_q.filter(Trade.market == market_upper)
        closed_trades = closed_q.order_by(Trade.closed_at.asc(), Trade.id.asc()).all()

        core = self.compute_core_metrics(closed_trades, starting_equity=starting_equity)

        # Strategy rankings & regime breakdown
        strategy_configs = db.query(StrategyConfig).order_by(StrategyConfig.id.asc()).all()
        config_by_name = {s.name: s for s in strategy_configs}

        trades_by_strat: dict[str, list[Trade]] = {s.name: [] for s in strategy_configs}
        for t in closed_trades:
            trades_by_strat.setdefault(t.strategy_name, []).append(t)

        strategy_rankings: list[dict[str, Any]] = []
        regime_buckets: dict[str, dict[str, Any]] = {}

        for name, strat_trades in trades_by_strat.items():
            cfg = config_by_name.get(name)
            display_name = cfg.display_name if cfg else name.replace("_", " ").title()
            target_regime = cfg.target_regime if cfg else STRATEGY_REGIME_FALLBACK.get(name, "GENERAL")
            enabled = bool(cfg.enabled) if cfg else True

            s_metrics = self.compute_core_metrics(strat_trades, starting_equity=starting_equity)
            s_gross, s_fees, s_net = self._compute_trade_gross_fees_net(strat_trades)
            strategy_rankings.append(
                {
                    "name": name,
                    "display_name": display_name,
                    "target_regime": target_regime,
                    "enabled": enabled,
                    "total_trades": s_metrics["total_closed_trades"],
                    "winning_trades": s_metrics["winning_trades"],
                    "losing_trades": s_metrics["losing_trades"],
                    "win_rate_pct": s_metrics["win_rate_pct"],
                    "gross_pnl_usd": s_gross,
                    "fees_paid_usd": s_fees,
                    "net_pnl_usd": s_net,
                    "profit_factor": s_metrics["profit_factor"],
                }
            )

            bucket = regime_buckets.setdefault(
                target_regime,
                {"regime": target_regime, "total_trades": 0, "winning_trades": 0, "net_pnl_usd": 0.0},
            )
            bucket["total_trades"] += s_metrics["total_closed_trades"]
            bucket["winning_trades"] += s_metrics["winning_trades"]
            bucket["net_pnl_usd"] = round(bucket["net_pnl_usd"] + s_metrics["net_pnl_usd"], 2)

        # Always provide the 3 canonical Strategy Brain strategies with Trade Count, Win Rate, Gross PnL, Fees Paid, Net PnL
        brain_strategy_performance: list[dict[str, Any]] = []
        for strat_key, label, default_regime in BRAIN_STRATEGIES:
            strat_trades = trades_by_strat.get(strat_key, [])
            cfg = config_by_name.get(strat_key)
            s_metrics = self.compute_core_metrics(strat_trades, starting_equity=starting_equity)
            s_gross, s_fees, s_net = self._compute_trade_gross_fees_net(strat_trades)
            brain_strategy_performance.append(
                {
                    "name": strat_key,
                    "display_name": label,
                    "target_regime": cfg.target_regime if cfg else default_regime,
                    "enabled": bool(cfg.enabled) if cfg else True,
                    "total_trades": s_metrics["total_closed_trades"],
                    "winning_trades": s_metrics["winning_trades"],
                    "losing_trades": s_metrics["losing_trades"],
                    "win_rate_pct": s_metrics["win_rate_pct"],
                    "gross_pnl_usd": s_gross,
                    "fees_paid_usd": s_fees,
                    "net_pnl_usd": s_net,
                    "profit_factor": s_metrics["profit_factor"],
                }
            )

        strategy_rankings.sort(
            key=lambda item: (item["net_pnl_usd"], item["win_rate_pct"], item["total_trades"]),
            reverse=True,
        )

        active_ranked = [s for s in strategy_rankings if s["total_trades"] > 0]
        best_strategy = active_ranked[0] if active_ranked else (strategy_rankings[0] if strategy_rankings else None)
        weakest_strategy = (
            active_ranked[-1] if active_ranked else (strategy_rankings[-1] if strategy_rankings else None)
        )

        regime_breakdown: list[dict[str, Any]] = []
        for reg, b in regime_buckets.items():
            wr = round((b["winning_trades"] / b["total_trades"]) * 100.0, 2) if b["total_trades"] > 0 else 0.0
            regime_breakdown.append(
                {
                    "regime": reg,
                    "total_trades": b["total_trades"],
                    "winning_trades": b["winning_trades"],
                    "win_rate_pct": wr,
                    "net_pnl_usd": round(b["net_pnl_usd"], 2),
                }
            )
        regime_breakdown.sort(key=lambda r: r["net_pnl_usd"], reverse=True)

        # Risk Shield report
        state = db.query(BotState).first()
        daily_pnl_pct = (
            ((selected_equity - starting_equity) / starting_equity) * 100.0
            if starting_equity > 0
            else 0.0
        )
        daily_loss_used_pct = round(max(0.0, -daily_pnl_pct), 2)

        logs = db.query(SystemLog).all()
        confluence_blocked = sum(
            1
            for log in logs
            if log.source in {"ConfluenceFilter", "ConfluenceEvaluator"}
            and any(k in (log.message or "").upper() for k in ("BLOCKED", "VETOED"))
        )
        session_blocked = sum(
            1
            for log in logs
            if log.source == "SessionGuard" and "PAUSED" in (log.message or "").upper()
        )

        risk_shield_report = {
            "daily_loss_used_pct": daily_loss_used_pct,
            "max_daily_loss_pct": settings.MAX_DAILY_LOSS_PCT,
            "kill_switch_active": bool(state.kill_switch_active) if state else False,
            "kill_switch_reason": state.kill_switch_reason if state else None,
            "confluence_blocked_count": confluence_blocked,
            "session_blocked_count": session_blocked,
            "total_protected_skips": confluence_blocked + session_blocked,
        }

        plain_english = self._compose_plain_english(
            market=market_upper,
            core=core,
            best_strategy=best_strategy,
            weakest_strategy=weakest_strategy,
            risk_shield=risk_shield_report,
            forex_session=forex_session,
        )

        return {
            "market": market_upper,
            **core,
            "best_strategy": best_strategy,
            "weakest_strategy": weakest_strategy,
            "strategy_rankings": strategy_rankings,
            "brain_strategy_performance": brain_strategy_performance,
            "regime_breakdown": regime_breakdown,
            "risk_shield_report": risk_shield_report,
            "forex_session": forex_session,
            "plain_english": plain_english,
        }

    @staticmethod
    def _compose_plain_english(
        market: str,
        core: dict[str, Any],
        best_strategy: Optional[dict[str, Any]],
        weakest_strategy: Optional[dict[str, Any]],
        risk_shield: dict[str, Any],
        forex_session: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        grade = core["health_grade"]
        total = core["total_closed_trades"]
        wr = core["win_rate_pct"]
        be = core["break_even_win_rate_pct"]
        exp = core["expectancy_usd"]
        pf = core["profit_factor"]

        if grade == "CALIBRATING":
            headline = (
                f"Calibrating in Paper Mode ({total}/5 closed trades): "
                "gathering statistically meaningful executions to measure true mathematical edge."
            )
        elif exp > 0 and wr >= be:
            exp_sign = "+" if exp >= 0 else "-"
            headline = (
                f"Healthy Edge ({grade}): {wr:.1f}% accuracy beats your {be:.1f}% break-even target "
                f"with {exp_sign}${abs(exp):.2f} average expectancy per trade (PF {pf:.2f})."
            )
        else:
            headline = (
                f"Caution ({grade}): {wr:.1f}% accuracy is below the {be:.1f}% break-even target "
                f"(PF {pf:.2f}); Risk Shield is actively protecting capital."
            )

        bullets: list[str] = []

        if best_strategy and best_strategy["total_trades"] > 0:
            sign = "+" if best_strategy["net_pnl_usd"] >= 0 else "-"
            bullets.append(
                f"Top Performer: {best_strategy['display_name']} leads with "
                f"{best_strategy['win_rate_pct']:.1f}% accuracy and "
                f"{sign}${abs(best_strategy['net_pnl_usd']):.2f} net PnL across {best_strategy['total_trades']} trade(s)."
            )
        else:
            bullets.append(
                f"Sample Building: No closed {market} trades have settled yet; strategies are scanning live candles."
            )

        if weakest_strategy and weakest_strategy["total_trades"] > 0 and weakest_strategy != best_strategy:
            w_sign = "+" if weakest_strategy["net_pnl_usd"] >= 0 else "-"
            bullets.append(
                f"Watch List: {weakest_strategy['display_name']} is currently the softest performer "
                f"({weakest_strategy['win_rate_pct']:.1f}% accuracy, {w_sign}${abs(weakest_strategy['net_pnl_usd']):.2f})."
            )

        if forex_session:
            s_name = forex_session.get("session_name", "UNKNOWN")
            if forex_session.get("is_news_blackout"):
                bullets.append(f"Forex Guard: News Blackout active — {forex_session.get('reason', 'Macro event')}.")
            elif forex_session.get("can_trade"):
                bullets.append(f"Forex Guard: {s_name} is open for high-liquidity entries.")
            else:
                bullets.append(
                    f"Forex Guard: New standard FX entries paused ({s_name}) — {forex_session.get('reason', 'Protection active')}."
                )

        skips = risk_shield["total_protected_skips"]
        if risk_shield["kill_switch_active"]:
            bullets.append(
                f"Risk Shield: EMERGENCY KILL SWITCH LOCKED ({risk_shield.get('kill_switch_reason') or 'Manual lock'})."
            )
        else:
            bullets.append(
                f"Risk Shield: Daily drawdown used is {risk_shield['daily_loss_used_pct']:.2f}% / "
                f"{risk_shield['max_daily_loss_pct']:.2f}% cap; blocked {skips} weak or off-session setup(s)."
            )

        if grade == "CALIBRATING":
            bullets.append("Next Action: Let the paper loop collect at least 5–10 closed trades before tuning weights.")
        elif grade in {"A+", "A", "B"}:
            bullets.append("Next Action: Keep current settings running; mathematical expectancy is positive.")
        else:
            bullets.append("Next Action: Keep Paper Mode active and consider toggling off the weakest strategy.")

        return {
            "headline": headline,
            "bullets": bullets[:5],
        }
