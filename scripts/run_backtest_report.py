"""Diagnostic CLI script running a real 60-day BTCUSDT 1h backtest via HistoricalDataFetcher."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine.backtester import Backtester
from app.engine.historical_data import HistoricalDataFetcher


def main() -> int:
    symbol = "BTCUSDT"
    interval = "1h"
    days = 60
    starting_balance = 10000.0

    fetcher = HistoricalDataFetcher()
    try:
        df, data_source = fetcher.fetch(symbol=symbol, interval=interval, days=days)
    finally:
        fetcher.close()

    backtester = Backtester()
    res = backtester.run(
        df=df,
        symbol=symbol,
        interval=interval,
        starting_balance=starting_balance,
        enable_partial_tp=True,
        enable_confluence=True,
        data_source=data_source,
    )

    print("=" * 68)
    print(f"APEX-PULSE PHASE 5 BACKTEST REPORT — {res.symbol} ({res.interval}, {days}d)")
    print("=" * 68)
    print(f"Data Source           : {res.data_source}")
    print(f"Candles Tested        : {res.candles_tested}")
    print(f"Starting Balance      : ${res.starting_balance:,.2f}")
    print(f"Ending Balance        : ${res.ending_balance:,.2f}")
    print(f"Net PnL ($)           : ${res.net_pnl_usd:+,.2f}")
    print(f"Net Return (%)        : {res.net_pnl_pct:+.2f}%  (Buy & Hold: {res.buy_and_hold_pct:+.2f}%)")
    print(f"Total Trades          : {res.total_trades} ({res.winning_trades}W / {res.losing_trades}L)")
    print(f"Win Rate (%)          : {res.win_rate_pct:.2f}%  (Break-Even Needed: {res.break_even_win_rate_pct:.2f}%)")
    print(f"Profit Factor         : {res.profit_factor:.2f}")
    print(f"Expectancy ($/trade)  : ${res.expectancy_usd:+.2f}")
    print(f"Best / Worst Trade    : ${res.best_trade_usd:+.2f} / ${res.worst_trade_usd:+.2f}")
    print(f"Max Drawdown (%)      : {res.max_drawdown_pct:.2f}%")
    print(f"Partial TP Hits (+1R) : {res.partial_tp_hits}")
    print(f"Circuit Breaker Trips : {res.circuit_breaker_trips}")
    print("-" * 68)
    print("Per-Strategy Breakdown:")
    if not res.per_strategy_breakdown:
        print("  (No trades triggered in window)")
    else:
        for strat_name, stats in res.per_strategy_breakdown.items():
            print(
                f"  - {strat_name:<22} | Trades: {stats['trades']:>3} | "
                f"WR: {stats['win_rate']:>5.1f}% | Net PnL: ${stats['net_pnl']:+8.2f}"
            )
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
