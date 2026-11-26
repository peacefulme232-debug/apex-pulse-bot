"""Bybit V5 Unified Trading Demo/Public Broker implementing BaseBroker."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import logging
import random
import re
import time
from typing import Any, Callable, Optional

import pandas as pd
from pybit.unified_trading import HTTP

from app.brokers.base import BaseBroker
from app.config import MAX_LEVERAGE, settings

logger = logging.getLogger(__name__)

INTERVAL_MAP: dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
    "1w": "W",
    "1M": "M",
}

NATIVE_INTERVALS = set(INTERVAL_MAP.values())
RATE_LIMIT_RETCODES = {10006, 10018}
GEO_BLOCK_RETCODES = {10009}


class BybitBrokerError(RuntimeError):
    """Structured Bybit V5 API error with optional retCode and HTTP status."""

    def __init__(self, message: str, ret_code: Optional[int] = None, http_status: Optional[int] = None) -> None:
        super().__init__(message)
        self.ret_code = ret_code
        self.http_status = http_status


@dataclass(frozen=True)
class InstrumentRules:
    """Cached Bybit V5 instrument precision and order limits."""

    symbol: str
    qty_step: Decimal
    min_order_qty: Decimal
    max_order_qty: Decimal
    tick_size: Decimal
    min_notional_value: Decimal


def _extract_error_codes(exc: Exception) -> tuple[Optional[int], Optional[int]]:
    """Extract Bybit retCode and HTTP status code from pybit exceptions or messages."""
    ret_code = getattr(exc, "errcode", None) or getattr(exc, "ret_code", None)
    http_status = getattr(exc, "status_code", None)

    msg = str(exc)
    if ret_code is None:
        m = re.search(r"(?:ErrCode|retCode)[\s:=]+(\d+)", msg, re.IGNORECASE)
        if m:
            ret_code = int(m.group(1))
    if http_status is None:
        m = re.search(r"(?:HTTP|status[_\s]?code)[\s:=]+(\d{3})", msg, re.IGNORECASE)
        if m:
            http_status = int(m.group(1))
    return (int(ret_code) if ret_code is not None else None, int(http_status) if http_status is not None else None)


class BybitBroker(BaseBroker):
    """Bybit V5 Unified Trading Demo connector with Decimal precision and resilient retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        category: Optional[str] = None,
        leverage: Optional[int] = None,
        mode: Optional[str] = None,
        session: Optional[Any] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_mode = (mode or settings.BYBIT_MODE or "paper").strip().lower()
        if resolved_mode not in {"paper", "demo"}:
            raise ValueError(
                f"BYBIT_MODE='{resolved_mode}' is blocked in Phase 3. Only 'paper' or 'demo' is allowed."
            )

        self.mode = resolved_mode
        self.category = (category or settings.BYBIT_CATEGORY or "linear").strip().lower()
        raw_lev = settings.BYBIT_LEVERAGE if leverage is None else int(leverage)
        self.leverage = max(1, min(MAX_LEVERAGE, raw_lev))

        self._api_key = (settings.BYBIT_API_KEY if api_key is None else api_key).strip()
        self._api_secret = (settings.BYBIT_API_SECRET if api_secret is None else api_secret).strip()
        self._sleep = sleep_fn

        if session is not None:
            self.session = session
        elif self._api_key and self._api_secret:
            self.session = HTTP(
                demo=True,
                testnet=False,
                api_key=self._api_key,
                api_secret=self._api_secret,
                recv_window=10000,
            )
        else:
            # Public read-only demo endpoint (e.g. for public kline diagnostics)
            self.session = HTTP(demo=True, testnet=False, recv_window=10000)

        self._instruments_cache: dict[str, InstrumentRules] = {}
        self._leverage_configured: set[str] = set()
        self.status: str = "CONNECTED_DEMO" if (self._api_key and self._api_secret) else "PUBLIC_READONLY"
        self.last_sync_message: str = "Initialized Bybit V5 Demo session"

    @staticmethod
    def map_interval(interval: str) -> str:
        """Translate human interval (e.g. '15m', '4h') to Bybit V5 interval string."""
        cleaned = str(interval).strip()
        if cleaned in INTERVAL_MAP:
            return INTERVAL_MAP[cleaned]
        if cleaned in NATIVE_INTERVALS:
            return cleaned
        raise ValueError(f"Unsupported Bybit interval: '{interval}'")

    def _call_with_retry(
        self,
        fn: Callable[..., dict[str, Any]],
        *args: Any,
        allow_ret_codes: Optional[set[int]] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute a pybit REST call with exponential backoff + jitter and structured error handling."""
        allowed = allow_ret_codes or set()
        attempt = 0
        while True:
            try:
                resp = fn(*args, **kwargs)
                if isinstance(resp, dict):
                    ret_code = resp.get("retCode", 0)
                    if ret_code not in (0, None) and int(ret_code) not in allowed:
                        ret_code_int = int(ret_code)
                        ret_msg = str(resp.get("retMsg", "Unknown Bybit error"))
                        if ret_code_int in RATE_LIMIT_RETCODES and attempt < max_retries:
                            delay = (0.5 * (2**attempt)) + random.uniform(0.05, 0.25)
                            logger.warning("Bybit rate limit (%s): retrying in %.2fs", ret_code_int, delay)
                            self._sleep(delay)
                            attempt += 1
                            continue
                        self._handle_error_code(ret_code_int, None, ret_msg)
                self.last_sync_message = "OK"
                if self._api_key and self._api_secret and self.status in {"ERROR", "PUBLIC_READONLY"}:
                    self.status = "CONNECTED_DEMO"
                return resp
            except BybitBrokerError:
                raise
            except Exception as exc:
                ret_code, http_status = _extract_error_codes(exc)
                if ret_code is not None and ret_code in allowed:
                    return {"retCode": ret_code, "retMsg": str(exc), "result": {}}

                is_rate_limited = (ret_code in RATE_LIMIT_RETCODES) or (http_status == 429)
                if is_rate_limited and attempt < max_retries:
                    delay = (0.5 * (2**attempt)) + random.uniform(0.05, 0.25)
                    logger.warning("Bybit rate limit/429 (%s/%s): retrying in %.2fs", ret_code, http_status, delay)
                    self._sleep(delay)
                    attempt += 1
                    continue

                self._handle_error_code(ret_code, http_status, str(exc))

    def _handle_error_code(
        self,
        ret_code: Optional[int],
        http_status: Optional[int],
        raw_msg: str,
    ) -> None:
        """Map Bybit V5 error codes and HTTP status codes to clear diagnostics and broker status."""
        if ret_code in GEO_BLOCK_RETCODES or http_status == 403 or "10009" in raw_msg or "403" in raw_msg:
            msg = "Bybit geo-restriction (10009/403) from server IP"
            self.status = "GEO_BLOCKED"
            self.last_sync_message = msg
            logger.error("%s | detail=%s", msg, raw_msg)
            raise BybitBrokerError(msg, ret_code=ret_code or 10009, http_status=http_status or 403)

        if ret_code == 10003:
            msg = "Bybit Auth/Env Mismatch (10003): verify key was created inside Bybit Mainnet Demo Trading"
            self.status = "ERROR"
            self.last_sync_message = msg
            logger.error("%s | detail=%s", msg, raw_msg)
            raise BybitBrokerError(msg, ret_code=10003, http_status=http_status)

        if ret_code == 10002:
            msg = "Clock drift detected (10002)"
            self.status = "ERROR"
            self.last_sync_message = msg
            logger.error("%s | detail=%s", msg, raw_msg)
            raise BybitBrokerError(msg, ret_code=10002, http_status=http_status)

        self.status = "ERROR"
        self.last_sync_message = raw_msg
        logger.error("Bybit API error (retCode=%s, http=%s): %s", ret_code, http_status, raw_msg)
        raise BybitBrokerError(raw_msg, ret_code=ret_code, http_status=http_status)

    def get_instrument_rules(self, symbol: str) -> InstrumentRules:
        """Fetch and cache qtyStep, minOrderQty, maxOrderQty, tickSize, and minNotionalValue."""
        if symbol in self._instruments_cache:
            return self._instruments_cache[symbol]

        resp = self._call_with_retry(
            self.session.get_instruments_info,
            category=self.category,
            symbol=symbol,
        )
        items = resp.get("result", {}).get("list", [])
        if not items:
            raise BybitBrokerError(f"Instrument info not found for {symbol} ({self.category})")

        info = items[0]
        lot = info.get("lotSizeFilter", {})
        price_filter = info.get("priceFilter", {})

        qty_step = Decimal(str(lot.get("qtyStep") or lot.get("basePrecision") or "0.001"))
        min_order_qty = Decimal(str(lot.get("minOrderQty") or "0.001"))
        max_order_qty = Decimal(str(lot.get("maxOrderQty") or "1000000"))
        tick_size = Decimal(str(price_filter.get("tickSize") or "0.01"))
        default_min_notional = "5" if self.category == "spot" else str(lot.get("minNotionalValue") or "5")
        min_notional = Decimal(str(lot.get("minNotionalValue") or default_min_notional))

        rules = InstrumentRules(
            symbol=symbol,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            max_order_qty=max_order_qty,
            tick_size=tick_size,
            min_notional_value=min_notional,
        )
        self._instruments_cache[symbol] = rules
        return rules

    @staticmethod
    def _quantize_step(value: Decimal, step: Decimal) -> Decimal:
        """Floor value to an exact multiple of step using Decimal ROUND_DOWN."""
        steps = (value / step).to_integral_value(rounding=ROUND_DOWN)
        return (steps * step).quantize(step, rounding=ROUND_DOWN)

    def quantize_qty(self, symbol: str, raw_qty: float | Decimal, price: Optional[float | Decimal] = None) -> str:
        """Quantize order quantity using Decimal ROUND_DOWN against cached qtyStep."""
        rules = self.get_instrument_rules(symbol)
        d_qty = Decimal(str(raw_qty))
        quantized = self._quantize_step(d_qty, rules.qty_step)

        if quantized < rules.min_order_qty:
            raise ValueError(
                f"Quantized qty {quantized} for {symbol} is below minOrderQty {rules.min_order_qty}."
            )
        if quantized > rules.max_order_qty:
            quantized = self._quantize_step(rules.max_order_qty, rules.qty_step)

        if price is not None:
            notional = quantized * Decimal(str(price))
            min_req = max(Decimal("5"), rules.min_notional_value) if self.category == "spot" else rules.min_notional_value
            if notional < min_req:
                raise ValueError(
                    f"Order notional {notional:.4f} USDT for {symbol} is below minimum {min_req} USDT."
                )

        return format(quantized, "f")

    def quantize_price(self, symbol: str, raw_price: float | Decimal) -> str:
        """Quantize SL/TP/limit price using Decimal ROUND_DOWN against cached tickSize."""
        rules = self.get_instrument_rules(symbol)
        d_price = Decimal(str(raw_price))
        quantized = self._quantize_step(d_price, rules.tick_size)
        if quantized <= Decimal("0"):
            raise ValueError(f"Quantized price {quantized} for {symbol} must be positive.")
        return format(quantized, "f")

    def ensure_leverage(self, symbol: str) -> None:
        """Set linear leverage once per symbol; treat retCode 110043 as non-fatal success."""
        if self.category != "linear" or symbol in self._leverage_configured:
            return
        lev_str = str(self.leverage)
        self._call_with_retry(
            self.session.set_leverage,
            category="linear",
            symbol=symbol,
            buyLeverage=lev_str,
            sellLeverage=lev_str,
            allow_ret_codes={110043},
        )
        self._leverage_configured.add(symbol)

    def get_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 250) -> pd.DataFrame:
        """Fetch Bybit V5 klines, strictly sort ascending by timestamp, and cast OHLCV to float."""
        mapped_interval = self.map_interval(interval)
        resp = self._call_with_retry(
            self.session.get_kline,
            category=self.category,
            symbol=symbol,
            interval=mapped_interval,
            limit=min(int(limit), 1000),
        )
        raw_list = resp.get("result", {}).get("list", [])
        if not raw_list:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Bybit returns klines newest-first; sort strictly ascending by timestamp ms
        sorted_rows = sorted(raw_list, key=lambda r: int(r[0]))
        records = [
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in sorted_rows
        ]
        return pd.DataFrame(records)

    def get_balance(self) -> dict[str, float]:
        """Return UNIFIED wallet USDT equity and walletBalance."""
        resp = self._call_with_retry(
            self.session.get_wallet_balance,
            accountType="UNIFIED",
        )
        accounts = resp.get("result", {}).get("list", [])
        usdt_equity = 0.0
        usdt_wallet = 0.0
        for acct in accounts:
            for coin in acct.get("coin", []):
                if coin.get("coin") == "USDT":
                    usdt_equity = float(coin.get("equity") or coin.get("walletBalance") or 0.0)
                    usdt_wallet = float(coin.get("walletBalance") or usdt_equity)
                    break
            if usdt_equity == 0.0 and acct.get("totalEquity"):
                usdt_equity = float(acct["totalEquity"])
                usdt_wallet = float(acct.get("totalWalletBalance") or usdt_equity)

        return {
            "bybit_equity": round(usdt_equity, 2),
            "bybit_wallet_balance": round(usdt_wallet, 2),
            "forex_equity": 0.0,
            "total_equity": round(usdt_equity, 2),
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Return active Bybit linear positions with non-zero size."""
        resp = self._call_with_retry(
            self.session.get_positions,
            category=self.category,
            settleCoin="USDT",
        )
        positions: list[dict[str, Any]] = []
        for pos in resp.get("result", {}).get("list", []):
            size = float(pos.get("size") or 0.0)
            if size <= 0.0:
                continue
            positions.append(
                {
                    "market": "BYBIT",
                    "symbol": pos.get("symbol"),
                    "side": "LONG" if pos.get("side") == "Buy" else "SHORT",
                    "qty": size,
                    "entry_price": float(pos.get("avgPrice") or 0.0),
                    "current_price": float(pos.get("markPrice") or pos.get("avgPrice") or 0.0),
                    "stop_loss": float(pos.get("stopLoss") or 0.0),
                    "take_profit": float(pos.get("takeProfit") or 0.0),
                    "pnl_usd": round(float(pos.get("unrealisedPnl") or 0.0), 2),
                    "status": "OPEN",
                }
            )
        return positions

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str,
        reason: str,
        entry_price_ref: Optional[float] = None,
    ) -> dict[str, Any]:
        """Place a Market order via Bybit V5 REST HTTP with quantized qty and SL/TP."""
        self.ensure_leverage(symbol)
        bybit_side = "Buy" if side.upper() in {"BUY", "LONG"} else "Sell"
        q_qty = self.quantize_qty(symbol, qty, price=entry_price_ref)
        q_sl = self.quantize_price(symbol, stop_loss)
        q_tp = self.quantize_price(symbol, take_profit)

        resp = self._call_with_retry(
            self.session.place_order,
            category=self.category,
            symbol=symbol,
            side=bybit_side,
            orderType="Market",
            qty=q_qty,
            stopLoss=q_sl,
            takeProfit=q_tp,
            tpslMode="Full",
        )
        result = resp.get("result", {})
        return {
            "order_id": result.get("orderId"),
            "order_link_id": result.get("orderLinkId"),
            "market": "BYBIT",
            "symbol": symbol,
            "side": "LONG" if bybit_side == "Buy" else "SHORT",
            "qty": float(q_qty),
            "stop_loss": float(q_sl),
            "take_profit": float(q_tp),
            "strategy_name": strategy_name,
            "entry_reason": reason,
            "status": "OPEN",
        }

    def update_stop_loss(self, symbol: str, new_sl: float, position_idx: int = 0) -> dict[str, Any]:
        """Update live trailing/breakeven stop-loss on Bybit V5."""
        q_sl = self.quantize_price(symbol, new_sl)
        resp = self._call_with_retry(
            self.session.set_trading_stop,
            category=self.category,
            symbol=symbol,
            stopLoss=q_sl,
            positionIdx=position_idx,
        )
        return {
            "symbol": symbol,
            "stop_loss": float(q_sl),
            "ret_code": resp.get("retCode", 0),
        }

    def close_position(self, trade_id: int, reason: str, symbol: Optional[str] = None, side: Optional[str] = None, qty: Optional[float] = None) -> dict[str, Any]:
        """Close an open Bybit position via reduceOnly Market order when symbol/side/qty are known."""
        if not symbol or not side or not qty:
            return {"id": trade_id, "status": "CLOSED", "reason": reason}
        close_side = "Sell" if side.upper() in {"BUY", "LONG"} else "Buy"
        q_qty = self.quantize_qty(symbol, qty)
        resp = self._call_with_retry(
            self.session.place_order,
            category=self.category,
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=q_qty,
            reduceOnly=True,
        )
        return {
            "id": trade_id,
            "order_id": resp.get("result", {}).get("orderId"),
            "status": "CLOSED",
            "reason": reason,
        }
