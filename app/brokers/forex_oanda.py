"""OANDA v20 Practice REST broker adapter implementing BaseBroker (100% cloud-safe via httpx)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import logging
import time
from typing import Any, Callable, Optional

import httpx
import pandas as pd

from app.brokers.base import BaseBroker
from app.config import settings

logger = logging.getLogger(__name__)

OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"

GRANULARITY_MAP: dict[str, str] = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
}


class OandaBrokerError(RuntimeError):
    """Structured exception raised by OandaForexBroker on API or auth failures."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OandaForexBroker(BaseBroker):
    """OANDA v20 Practice REST API adapter for Forex & Metals (EUR_USD, GBP_USD, XAU_USD)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.api_key = (settings.OANDA_API_KEY if api_key is None else api_key).strip()
        self.account_id = (settings.OANDA_ACCOUNT_ID if account_id is None else account_id).strip()
        resolved_url = (settings.OANDA_BASE_URL if base_url is None else base_url).strip().rstrip("/")

        if resolved_url != OANDA_PRACTICE_URL:
            raise ValueError(
                f"OandaForexBroker strictly requires Practice URL '{OANDA_PRACTICE_URL}', got '{resolved_url}'."
            )

        self.base_url = resolved_url
        self.max_retries = max(1, int(max_retries))
        self._sleep = sleep_fn
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers=self._build_headers(),
        )
        self.status: str = "CONNECTED_PRACTICE"
        self.last_sync_message: str = "OANDA Practice REST adapter initialized"

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    @staticmethod
    def map_granularity(interval: str) -> str:
        """Map APEX-PULSE timeframe string to OANDA v20 granularity code."""
        cleaned = (interval or "15m").strip()
        lower = cleaned.lower()
        if lower in GRANULARITY_MAP:
            return GRANULARITY_MAP[lower]
        upper = cleaned.upper()
        if upper in GRANULARITY_MAP.values():
            return upper
        raise ValueError(f"Unsupported OANDA granularity/interval: '{interval}'")

    @staticmethod
    def get_price_decimals(symbol: str) -> int:
        """Return standard pip/sub-pip decimal precision for OANDA instruments."""
        upper = (symbol or "").strip().upper()
        if "XAU" in upper or "XAG" in upper:
            return 2
        if upper.endswith("_JPY") or upper.endswith("JPY"):
            return 3
        return 5

    @classmethod
    def format_price(cls, symbol: str, price: float) -> str:
        """Quantize price with Decimal to exact instrument precision without float artifacts."""
        decimals = cls.get_price_decimals(symbol)
        quant = Decimal("1").scaleb(-decimals)
        dec_val = Decimal(str(price)).quantize(quant, rounding=ROUND_HALF_UP)
        return f"{dec_val:.{decimals}f}"

    @staticmethod
    def _parse_oanda_time_ms(raw_time: Any) -> int:
        if isinstance(raw_time, (int, float)):
            val = float(raw_time)
            return int(val * 1000) if val < 1e12 else int(val)
        text = str(raw_time or "").strip()
        if not text:
            return 0
        # Handle RFC3339 with nanosecond precision (e.g. 2026-01-01T12:00:00.000000000Z)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "." in text:
            date_part, rest = text.split(".", 1)
            tz_idx = max(rest.find("+"), rest.find("-"))
            if tz_idx != -1:
                frac = rest[:tz_idx][:6]
                tz_part = rest[tz_idx:]
                text = f"{date_part}.{frac}{tz_part}"
            else:
                text = f"{date_part}.{rest[:6]}"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        headers = self._build_headers()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                self.status = "ERROR"
                self.last_sync_message = f"OANDA network error: {exc}"
                raise OandaBrokerError(f"OANDA network error: {exc}") from exc

            code = response.status_code
            if code in {401, 403}:
                msg = (
                    f"OANDA authentication failed (HTTP {code}). "
                    "Verify OANDA_API_KEY and OANDA_ACCOUNT_ID belong to an OANDA Practice account."
                )
                logger.error(msg)
                self.status = "ERROR"
                self.last_sync_message = msg
                raise OandaBrokerError(msg, status_code=code)

            if code == 429 or 500 <= code < 600:
                if attempt + 1 < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                msg = f"OANDA transient HTTP {code} after {self.max_retries} attempts: {response.text[:200]}"
                self.status = "ERROR"
                self.last_sync_message = msg
                raise OandaBrokerError(msg, status_code=code)

            if code >= 400:
                msg = f"OANDA request rejected (HTTP {code}): {response.text[:250]}"
                self.status = "ERROR"
                self.last_sync_message = msg
                raise OandaBrokerError(msg, status_code=code)

            self.status = "CONNECTED_PRACTICE"
            self.last_sync_message = f"OANDA Practice synced ({method} {path})"
            return response.json()

        raise OandaBrokerError(f"OANDA request failed: {last_exc}")

    def get_balance(self) -> dict[str, float]:
        """Fetch OANDA Practice account NAV and balance via /accounts/{account_id}/summary."""
        data = self._request("GET", f"/accounts/{self.account_id}/summary")
        acct = data.get("account", {})
        nav = float(acct.get("NAV", acct.get("balance", 0.0)))
        bal = float(acct.get("balance", nav))
        return {
            "forex_equity": round(nav, 2),
            "total_equity": round(nav, 2),
            "balance": round(bal, 2),
        }

    def get_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 250) -> pd.DataFrame:
        """Fetch mid-price OHLCV candles in ascending chronological order with complete awareness."""
        granularity = self.map_granularity(interval)
        data = self._request(
            "GET",
            f"/instruments/{symbol}/candles",
            params={
                "granularity": granularity,
                "count": int(limit),
                "price": "M",
            },
        )
        candles = data.get("candles", [])
        rows: list[dict[str, Any]] = []
        for c in candles:
            mid = c.get("mid") or {}
            if not mid:
                continue
            rows.append(
                {
                    "timestamp": self._parse_oanda_time_ms(c.get("time")),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": float(c.get("volume", 1.0)),
                    "complete": bool(c.get("complete", True)),
                }
            )

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "complete"])

        df = pd.DataFrame(rows).sort_values("timestamp", ascending=True).reset_index(drop=True)

        # Guarantee iloc[-2] is always a closed candle (complete == True) when >= 2 candles exist.
        if len(df) >= 2 and not bool(df["complete"].iloc[-2]):
            closed_df = df[df["complete"]].reset_index(drop=True)
            forming_df = df[~df["complete"]].tail(1).reset_index(drop=True)
            df = pd.concat([closed_df, forming_df], ignore_index=True)

        return df

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str,
        reason: str,
    ) -> dict[str, Any]:
        """Submit a Market FOK order with signed integer units and GTC SL/TP on fill."""
        side_upper = side.strip().upper()
        is_long = side_upper in {"BUY", "LONG"}
        abs_units = max(1, int(round(abs(float(qty)))))
        signed_units = abs_units if is_long else -abs_units

        formatted_sl = self.format_price(symbol, stop_loss)
        formatted_tp = self.format_price(symbol, take_profit)

        payload = {
            "order": {
                "type": "MARKET",
                "instrument": symbol,
                "units": str(signed_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": formatted_sl,
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": formatted_tp,
                    "timeInForce": "GTC",
                },
            }
        }

        res = self._request("POST", f"/accounts/{self.account_id}/orders", json_body=payload)
        fill = res.get("orderFillTransaction") or {}
        trade_opened = fill.get("tradeOpened") or {}
        trade_id = trade_opened.get("tradeID") or fill.get("id") or res.get("lastTransactionID", "")
        fill_price = float(fill.get("price", 0.0))

        return {
            "order_id": str(trade_id),
            "trade_id": str(trade_id),
            "market": "FOREX",
            "provider": "oanda",
            "symbol": symbol,
            "side": "LONG" if is_long else "SHORT",
            "units": signed_units,
            "qty": float(abs_units),
            "entry_price": fill_price,
            "stop_loss": float(formatted_sl),
            "take_profit": float(formatted_tp),
            "strategy_name": strategy_name,
            "entry_reason": reason,
            "status": "OPEN",
        }

    def update_stop_loss(
        self,
        trade_id: int | str,
        new_sl: float,
        symbol: str = "EUR_USD",
    ) -> dict[str, Any]:
        """Update live trailing Stop Loss on an open OANDA trade."""
        formatted_sl = self.format_price(symbol, new_sl)
        payload = {
            "stopLoss": {
                "price": formatted_sl,
                "timeInForce": "GTC",
            }
        }
        res = self._request(
            "PUT",
            f"/accounts/{self.account_id}/trades/{trade_id}/orders",
            json_body=payload,
        )
        return {
            "trade_id": str(trade_id),
            "stop_loss": float(formatted_sl),
            "transaction_id": str(res.get("lastTransactionID", "")),
        }

    def close_position(self, trade_id: int | str, reason: str) -> dict[str, Any]:
        """Close an open OANDA trade by trade_id."""
        res = self._request(
            "PUT",
            f"/accounts/{self.account_id}/trades/{trade_id}/close",
            json_body={"units": "ALL"},
        )
        fill = res.get("orderFillTransaction") or {}
        return {
            "trade_id": str(trade_id),
            "status": "CLOSED",
            "exit_price": float(fill.get("price", 0.0)),
            "pnl_usd": float(fill.get("pl", 0.0)),
            "reason": reason,
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch currently open trades via /accounts/{account_id}/openTrades."""
        data = self._request("GET", f"/accounts/{self.account_id}/openTrades")
        trades = data.get("trades", [])
        positions: list[dict[str, Any]] = []
        for t in trades:
            units = float(t.get("currentUnits", 0.0))
            side = "LONG" if units >= 0 else "SHORT"
            sl_order = t.get("stopLossOrder") or {}
            tp_order = t.get("takeProfitOrder") or {}
            positions.append(
                {
                    "id": str(t.get("id", "")),
                    "market": "FOREX",
                    "symbol": str(t.get("instrument", "")),
                    "side": side,
                    "qty": abs(units),
                    "entry_price": float(t.get("price", 0.0)),
                    "current_price": float(t.get("price", 0.0)),
                    "stop_loss": float(sl_order.get("price", 0.0)) if sl_order.get("price") else 0.0,
                    "take_profit": float(tp_order.get("price", 0.0)) if tp_order.get("price") else 0.0,
                    "pnl_usd": float(t.get("unrealizedPL", 0.0)),
                    "status": "OPEN",
                }
            )
        return positions

    def close(self) -> None:
        """Close owned httpx client cleanly."""
        if self._owns_client:
            self._client.close()
