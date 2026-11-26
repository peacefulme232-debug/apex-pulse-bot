"""Diagnostic script: Test public Bybit V5 BTCUSDT 15m + 4h klines -> RegimeDetector + ConfluenceFilter."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.brokers.bybit_broker import BybitBroker, BybitBrokerError
from app.engine.confluence import ConfluenceFilter
from app.engine.indicators import add_all_indicators
from app.engine.regime import RegimeDetector
from app.engine.strategies import StrategyDispatcher, TradeSignal


def main() -> int:
    print("=== APEX-PULSE Phase 3 Bybit V5 Diagnostic ===")
    broker = BybitBroker(api_key="", api_secret="", mode="demo", category="linear")

    try:
        ltf_df = broker.get_ohlcv("BTCUSDT", interval="15m", limit=250)
        htf_df = broker.get_ohlcv("BTCUSDT", interval="4h", limit=100)
    except BybitBrokerError as exc:
        if exc.ret_code == 10009 or exc.http_status == 403:
            print(
                "[GEO_BLOCKED] Bybit returned HTTP 403 / retCode 10009 from this VM's IP region.\n"
                "This is expected on US-hosted cloud VMs. PaperSimBroker fallback remains active,\n"
                "and unit tests mock pybit deterministically."
            )
            return 0
        print(f"[BYBIT_ERROR] retCode={exc.ret_code} http={exc.http_status}: {exc}")
        return 1
    except Exception as exc:
        print(f"[CONNECTION_DIAGNOSTIC] Unable to reach Bybit public API from this VM: {exc}")
        return 0

    print(f"Fetched BTCUSDT 15m: {len(ltf_df)} bars | 4h: {len(htf_df)} bars")
    enriched = add_all_indicators(ltf_df)
    regime_result = RegimeDetector().detect(enriched)
    dispatcher = StrategyDispatcher()
    _, signal = dispatcher.evaluate(ltf_df, {"trend_rider", "mean_reversion", "volatility_breakout"})

    # If strategy is HOLD on current live bar, also show hypothetical BUY confluence telemetry
    eval_signal = signal
    if signal.action == "HOLD":
        close = float(ltf_df["close"].iloc[-2])
        eval_signal = TradeSignal(
            action="BUY",
            strategy_name="diagnostic_probe",
            regime=regime_result.regime.value,
            confidence=regime_result.confidence,
            entry_price=close,
            stop_loss=close * 0.99,
            take_profit=close * 1.02,
            reason="Diagnostic probe",
        )

    conf = ConfluenceFilter().evaluate(eval_signal, ltf_df, htf_df=htf_df)
    print(f"Detected Regime : {regime_result.regime.value} (conf={regime_result.confidence:.2f})")
    print(f"Strategy Signal : {signal.action} ({signal.strategy_name})")
    print(
        f"Confluence Radar: score={conf.score}/100 | 4H={conf.htf_bias} | "
        f"VWAP={conf.vwap_aligned} | Vol={conf.volume_ratio:.2f}x | passed={conf.passed}"
    )
    print(f"Reasons         : {' | '.join(conf.reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
