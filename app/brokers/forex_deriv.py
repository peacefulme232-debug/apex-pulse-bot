"""Deriv WebSocket & 2026 Options OTP broker adapter implementing BaseBroker (100% cloud-safe)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

import httpx
import pandas as pd
from websockets.sync.client import connect as ws_sync_connect

from app.brokers.base import BaseBroker
from app.config import settings

logger = logging.getLogger(__name__)

DERIV_PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
DERIV_OTP_URL_TEMPLATE = "https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp"
DERIV_LEGACY_WS_TEMPLATE = "wss://ws.derivws.com/websockets/v3?app_id={app_id}"
PING_INTERVAL_SECONDS = 25

GRANULARITY_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class DerivBrokerError(RuntimeError):
    """Structured exception raised by DerivForexBroker on WS/HTTP/API errors."""

    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class DerivForexBroker(BaseBroker):
    """Deriv Demo adapter supporting modern 2026 Options OTP WS and legacy v3 WS."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        app_id: Optional[str] = None,
        account_id: Optional[str] = None,
        auth_mode: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        ws_connect: Optional[Callable[..., Any]] = None,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.api_token = (settings.DERIV_API_TOKEN if api_token is None else api_token).strip()
        self.app_id = (settings.DERIV_APP_ID if app_id is None else str(app_id)).strip() or "1089"
        self.account_id = (settings.DERIV_ACCOUNT_ID if account_id is None else account_id).strip()
        self.auth_mode = (settings.DERIV_AUTH_MODE if auth_mode is None else auth_mode).strip().lower()

        if self.auth_mode not in {"auto", "otp", "legacy"}:
            raise ValueError(f"Unsupported DERIV_AUTH_MODE: '{self.auth_mode}'")

        # Phase 4 Demo-only safety lock: block real-money CR* Deriv accounts
        if self.account_id.upper().startswith("CR"):
            raise ValueError(
                f"Real Deriv account '{self.account_id}' is blocked in Phase 4. Use a Demo (VRTC) account."
            )

        self.max_retries = max(1, int(max_retries))
        self._sleep = sleep_fn
        self._timeout = timeout_seconds
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self._ws_connect = ws_connect or ws_sync_connect

        self.status: str = "CONNECTED_DEMO"
        self.last_sync_message: str = f"Deriv Demo adapter initialized ({self.resolve_auth_mode()} mode)"

    def resolve_auth_mode(self) -> str:
        """Resolve effective auth mode: 'otp' when account_id present in auto mode, else 'legacy'."""
        if self.auth_mode == "auto":
            return "otp" if bool(self.account_id) else "legacy"
        return self.auth_mode

    @staticmethod
    def map_granularity(interval: str | int) -> int:
        """Map APEX-PULSE timeframe string to Deriv integer seconds granularity."""
        if isinstance(interval, int):
            if interval in GRANULARITY_SECONDS.values():
                return interval
            raise ValueError(f"Unsupported Deriv integer granularity: {interval}")
        cleaned = (interval or "15m").strip().lower()
        if cleaned in GRANULARITY_SECONDS:
            return GRANULARITY_SECONDS[cleaned]
        if cleaned.isdigit() and int(cleaned) in GRANULARITY_SECONDS.values():
            return int(cleaned)
        raise ValueError(f"Unsupported Deriv granularity/interval: '{interval}'")

    @staticmethod
    def build_ping_payload() -> dict[str, int]:
        """Return Deriv 25-second keepalive ping payload."""
        return {"ping": 1}

    def send_keepalive_ping(self, ws: Any) -> dict[str, Any]:
        """Send a single {"ping": 1} keepalive frame on an open WebSocket."""
        ws.send(json.dumps(self.build_ping_payload()))
        raw = ws.recv()
        return json.loads(raw) if isinstance(raw, (str, bytes)) else raw

    def obtain_otp_ws_url(self) -> str:
        """Call Deriv 2026 Options OTP endpoint to obtain a single-use 120s authenticated WS URL."""
        if not self.account_id:
            raise DerivBrokerError("DERIV_ACCOUNT_ID is required for OTP authentication mode.")
        if not self.api_token:
            raise DerivBrokerError("DERIV_API_TOKEN is required for OTP authentication mode.")

        url = DERIV_OTP_URL_TEMPLATE.format(account_id=self.account_id)
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Deriv-App-ID": self.app_id,
            "Content-Type": "application/json",
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._http_client.post(url, headers=headers)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                self.status = "ERROR"
                self.last_sync_message = f"Deriv OTP network error: {exc}"
                raise DerivBrokerError(f"Deriv OTP network error: {exc}") from exc

            if resp.status_code in {401, 403}:
                msg = f"Deriv OTP auth rejected (HTTP {resp.status_code}). Check DERIV_API_TOKEN and DERIV_APP_ID."
                logger.error(msg)
                self.status = "ERROR"
                self.last_sync_message = msg
                raise DerivBrokerError(msg, error_code=str(resp.status_code))

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if attempt + 1 < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                msg = f"Deriv OTP transient HTTP {resp.status_code}: {resp.text[:200]}"
                self.status = "ERROR"
                self.last_sync_message = msg
                raise DerivBrokerError(msg, error_code=str(resp.status_code))

            if resp.status_code >= 400:
                msg = f"Deriv OTP failed (HTTP {resp.status_code}): {resp.text[:200]}"
                self.status = "ERROR"
                self.last_sync_message = msg
                raise DerivBrokerError(msg, error_code=str(resp.status_code))

            body = resp.json()
            ws_url = (body.get("data") or {}).get("url") or body.get("url")
            if not ws_url:
                raise DerivBrokerError("Deriv OTP response missing data.url")
            return str(ws_url)

        raise DerivBrokerError(f"Deriv OTP request failed: {last_exc}")

    def _ws_exchange(
        self,
        ws_url: str,
        messages: list[dict[str, Any]],
        auth_message: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a sequence of JSON request/response exchanges with reconnect exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with self._ws_connect(ws_url) as ws:
                    if auth_message is not None:
                        ws.send(json.dumps(auth_message))
                        auth_raw = ws.recv()
                        auth_resp = json.loads(auth_raw) if isinstance(auth_raw, (str, bytes)) else auth_raw
                        if auth_resp.get("error"):
                            err = auth_resp["error"]
                            msg = f"Deriv authorize failed: {err.get('message', err)}"
                            self.status = "ERROR"
                            self.last_sync_message = msg
                            raise DerivBrokerError(msg, error_code=str(err.get("code", "")))
                        # Demo-only guard on legacy authorize response
                        auth_info = auth_resp.get("authorize") or {}
                        if auth_info and auth_info.get("is_virtual") == 0:
                            raise DerivBrokerError(
                                "Real-money Deriv account detected in authorize response; Demo (virtual) only in Phase 4.",
                                error_code="LIVE_BLOCKED",
                            )

                    responses: list[dict[str, Any]] = []
                    for payload in messages:
                        ws.send(json.dumps(payload))
                        raw = ws.recv()
                        resp = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                        if resp.get("error"):
                            err = resp["error"]
                            msg = f"Deriv API error ({err.get('code', 'UNKNOWN')}): {err.get('message', err)}"
                            self.status = "ERROR"
                            self.last_sync_message = msg
                            raise DerivBrokerError(msg, error_code=str(err.get("code", "")))
                        responses.append(resp)

                    self.status = "CONNECTED_DEMO"
                    self.last_sync_message = "Deriv Demo WebSocket synced"
                    return responses
            except DerivBrokerError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                self.status = "ERROR"
                self.last_sync_message = f"Deriv WebSocket connection failed: {exc}"
                raise DerivBrokerError(f"Deriv WebSocket connection failed: {exc}") from exc

        raise DerivBrokerError(f"Deriv WebSocket exchange failed: {last_exc}")

    def _authenticated_exchange(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        mode = self.resolve_auth_mode()
        if mode == "otp":
            ws_url = self.obtain_otp_ws_url()
            return self._ws_exchange(ws_url, messages, auth_message=None)

        if not self.api_token:
            raise DerivBrokerError("DERIV_API_TOKEN is required for legacy authentication mode.")
        ws_url = DERIV_LEGACY_WS_TEMPLATE.format(app_id=self.app_id)
        return self._ws_exchange(ws_url, messages, auth_message={"authorize": self.api_token})

    def get_ohlcv(self, symbol: str, interval: str = "15m", limit: int = 250) -> pd.DataFrame:
        """Fetch historical OHLC candles via public Deriv WS (zero authentication required)."""
        gran = self.map_granularity(interval)
        req = {
            "ticks_history": symbol,
            "end": "latest",
            "count": int(limit),
            "style": "candles",
            "granularity": gran,
            "adjust_start_time": 1,
        }
        responses = self._ws_exchange(DERIV_PUBLIC_WS_URL, [req])
        candles = responses[0].get("candles", [])
        rows: list[dict[str, Any]] = []
        for c in candles:
            vol = float(c.get("volume", 1.0) or 1.0)
            if vol <= 0.0:
                vol = 1.0
            rows.append(
                {
                    "timestamp": int(c["epoch"]) * 1000,
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": vol,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        return pd.DataFrame(rows).sort_values("timestamp", ascending=True).reset_index(drop=True)

    @staticmethod
    def compute_multiplier_limits(
        stake_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        multiplier: int,
    ) -> tuple[float, float]:
        """Convert price-distance SL/TP into positive USD amounts with SL strictly capped <= stake_usd."""
        safe_stake = max(1.0, round(float(stake_usd), 2))
        safe_mult = max(1, int(multiplier))

        if entry_price > 0:
            sl_pct = abs(entry_price - stop_loss) / entry_price
            tp_pct = abs(take_profit - entry_price) / entry_price
            raw_sl_usd = safe_stake * safe_mult * sl_pct
            raw_tp_usd = safe_stake * safe_mult * tp_pct
        else:
            raw_sl_usd = safe_stake * 0.5
            raw_tp_usd = safe_stake * 1.0

        sl_usd = round(max(0.50, min(raw_sl_usd, safe_stake)), 2)
        tp_usd = round(max(1.0, raw_tp_usd), 2)
        return sl_usd, tp_usd

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
        stake_usd: Optional[float] = None,
        multiplier: int = 10,
    ) -> dict[str, Any]:
        """Execute a risk-capped Deriv Multiplier contract (MULTUP / MULTDOWN) via proposal -> buy."""
        side_upper = side.strip().upper()
        is_long = side_upper in {"BUY", "LONG"}
        contract_type = "MULTUP" if is_long else "MULTDOWN"
        safe_mult = max(1, min(int(multiplier), 100))

        resolved_stake = (
            max(1.0, round(float(stake_usd), 2))
            if stake_usd is not None
            else max(10.0, round(min(abs(float(qty)), 500.0), 2))
        )
        ref_price = float(entry_price_ref) if entry_price_ref is not None else 0.0
        sl_usd, tp_usd = self.compute_multiplier_limits(
            stake_usd=resolved_stake,
            entry_price=ref_price,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            multiplier=safe_mult,
        )

        proposal_req = {
            "proposal": 1,
            "amount": resolved_stake,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "multiplier": safe_mult,
            "underlying_symbol": symbol,
            "symbol": symbol,
            "limit_order": {
                "stop_loss": sl_usd,
                "take_profit": tp_usd,
            },
        }

        # Step A: Request proposal, then Step B: Execute buy on the same authenticated session
        mode = self.resolve_auth_mode()
        if mode == "otp":
            ws_url = self.obtain_otp_ws_url()
            auth_msg = None
        else:
            if not self.api_token:
                raise DerivBrokerError("DERIV_API_TOKEN is required for legacy authentication mode.")
            ws_url = DERIV_LEGACY_WS_TEMPLATE.format(app_id=self.app_id)
            auth_msg = {"authorize": self.api_token}

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with self._ws_connect(ws_url) as ws:
                    if auth_msg is not None:
                        ws.send(json.dumps(auth_msg))
                        auth_raw = ws.recv()
                        auth_resp = json.loads(auth_raw) if isinstance(auth_raw, (str, bytes)) else auth_raw
                        if auth_resp.get("error"):
                            err = auth_resp["error"]
                            raise DerivBrokerError(
                                f"Deriv authorize failed: {err.get('message', err)}",
                                error_code=str(err.get("code", "")),
                            )

                    ws.send(json.dumps(proposal_req))
                    prop_raw = ws.recv()
                    prop_resp = json.loads(prop_raw) if isinstance(prop_raw, (str, bytes)) else prop_raw
                    if prop_resp.get("error"):
                        err = prop_resp["error"]
                        raise DerivBrokerError(
                            f"Deriv proposal failed: {err.get('message', err)}",
                            error_code=str(err.get("code", "")),
                        )

                    proposal = prop_resp.get("proposal") or {}
                    proposal_id = proposal.get("id")
                    ask_price = float(proposal.get("ask_price", resolved_stake))
                    if not proposal_id:
                        raise DerivBrokerError("Deriv proposal response missing proposal.id")

                    buy_req = {"buy": proposal_id, "price": ask_price}
                    ws.send(json.dumps(buy_req))
                    buy_raw = ws.recv()
                    buy_resp = json.loads(buy_raw) if isinstance(buy_raw, (str, bytes)) else buy_raw
                    if buy_resp.get("error"):
                        err = buy_resp["error"]
                        raise DerivBrokerError(
                            f"Deriv buy failed: {err.get('message', err)}",
                            error_code=str(err.get("code", "")),
                        )

                    buy_info = buy_resp.get("buy") or {}
                    contract_id = buy_info.get("contract_id", "")
                    self.status = "CONNECTED_DEMO"
                    self.last_sync_message = f"Deriv {contract_type} contract #{contract_id} opened"
                    return {
                        "order_id": str(contract_id),
                        "contract_id": str(contract_id),
                        "proposal_id": str(proposal_id),
                        "market": "FOREX",
                        "provider": "deriv",
                        "symbol": symbol,
                        "side": "LONG" if is_long else "SHORT",
                        "contract_type": contract_type,
                        "stake_usd": resolved_stake,
                        "buy_price": float(buy_info.get("buy_price", ask_price)),
                        "stop_loss_usd": sl_usd,
                        "take_profit_usd": tp_usd,
                        "multiplier": safe_mult,
                        "strategy_name": strategy_name,
                        "entry_reason": reason,
                        "status": "OPEN",
                    }
            except DerivBrokerError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    self._sleep(0.5 * (2**attempt))
                    continue
                self.status = "ERROR"
                self.last_sync_message = f"Deriv place_order failed: {exc}"
                raise DerivBrokerError(f"Deriv place_order failed: {exc}") from exc

        raise DerivBrokerError(f"Deriv place_order failed: {last_exc}")

    def get_balance(self) -> dict[str, float]:
        """Fetch Deriv Demo account balance."""
        responses = self._authenticated_exchange([{"balance": 1}])
        bal_info = responses[0].get("balance") or {}
        bal = float(bal_info.get("balance", 0.0))
        return {
            "forex_equity": round(bal, 2),
            "total_equity": round(bal, 2),
        }

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Fetch open Deriv Multiplier contracts via portfolio request."""
        responses = self._authenticated_exchange([{"portfolio": 1}])
        contracts = (responses[0].get("portfolio") or {}).get("contracts", [])
        positions: list[dict[str, Any]] = []
        for c in contracts:
            ctype = str(c.get("contract_type", "")).upper()
            side = "LONG" if "UP" in ctype else "SHORT"
            positions.append(
                {
                    "id": str(c.get("contract_id", "")),
                    "market": "FOREX",
                    "symbol": str(c.get("symbol", "")),
                    "side": side,
                    "qty": float(c.get("buy_price", 0.0)),
                    "entry_price": float(c.get("buy_price", 0.0)),
                    "current_price": float(c.get("payout", c.get("buy_price", 0.0))),
                    "pnl_usd": 0.0,
                    "status": "OPEN",
                }
            )
        return positions

    def close_position(self, trade_id: int | str, reason: str) -> dict[str, Any]:
        """Close an open Deriv contract by contract_id via sell request."""
        responses = self._authenticated_exchange([{"sell": int(trade_id), "price": 0}])
        sell_info = responses[0].get("sell") or {}
        return {
            "contract_id": str(trade_id),
            "status": "CLOSED",
            "sold_for": float(sell_info.get("sold_for", 0.0)),
            "reason": reason,
        }

    def close(self) -> None:
        """Close owned httpx client cleanly."""
        if self._owns_http_client:
            self._http_client.close()
