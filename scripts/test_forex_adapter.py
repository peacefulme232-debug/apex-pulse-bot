"""Phase 4 Diagnostic Script: SessionGuard, ForexRouter fallback, and optional public Deriv candle check."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.brokers.forex_deriv import DerivBrokerError, DerivForexBroker
from app.brokers.forex_router import ForexRouter
from app.engine.session_guard import SessionGuard


def main() -> None:
    print("=== APEX-PULSE Phase 4 Forex Adapter & SessionGuard Diagnostics ===")

    guard = SessionGuard(killzone_only=True, news_blackout_minutes=30)

    checks = [
        ("Weekend Close (Sat 12:00 UTC)", "EUR_USD", datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)),
        ("Daily Rollover (Tue 22:00 UTC)", "EUR_USD", datetime(2026, 6, 9, 22, 0, tzinfo=timezone.utc)),
        ("London/NY Overlap (Tue 14:00 UTC)", "EUR_USD", datetime(2026, 6, 9, 14, 0, tzinfo=timezone.utc)),
        ("Asian Low-Liquidity (Tue 03:00 UTC)", "EUR_USD", datetime(2026, 6, 9, 3, 0, tzinfo=timezone.utc)),
        ("NFP News Blackout (1st Fri 13:00 UTC)", "EUR_USD", datetime(2026, 6, 5, 13, 0, tzinfo=timezone.utc)),
        ("Deriv R_100 24/7 Bypass (Sat 12:00 UTC)", "R_100", datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)),
    ]

    print("\n[1] SessionGuard Checks:")
    for label, sym, ts in checks:
        res = guard.evaluate(symbol=sym, now_utc=ts)
        print(
            f"  - {label}: can_trade={res.can_trade} | session={res.session_name} | "
            f"weekend={res.is_weekend_closed} | rollover={res.is_rollover_pause} | "
            f"news={res.is_news_blackout} | reason='{res.reason}'"
        )

    print("\n[2] ForexRouter Safety Fallback Check:")
    router_oanda = ForexRouter(provider="oanda", oanda_api_key="", oanda_account_id="")
    router_deriv = ForexRouter(provider="deriv", deriv_api_token="")
    print(f"  - Blank OANDA -> {router_oanda.get_status()['status']} ({router_oanda.get_status()['last_sync_message']})")
    print(f"  - Blank Deriv -> {router_deriv.get_status()['status']} ({router_deriv.get_status()['last_sync_message']})")

    print("\n[3] Optional Public Deriv WS Candle Fetch (R_100, 15m, count=5):")
    try:
        deriv = DerivForexBroker(api_token="", app_id="1089", max_retries=1, timeout_seconds=5.0)
        df = deriv.get_ohlcv("R_100", interval="15m", limit=5)
        print(f"  - SUCCESS: Retrieved {len(df)} candles. Latest close={df['close'].iloc[-1] if not df.empty else 'N/A'}")
    except (DerivBrokerError, Exception) as exc:
        print(f"  - SKIPPED / NETWORK BLOCKED (non-fatal): {exc}")

    print("\n=== Diagnostics Complete ===")


if __name__ == "__main__":
    main()
