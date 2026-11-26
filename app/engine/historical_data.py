"""Geo-resilient multi-source historical OHLCV candle fetcher (Bybit -> Coinbase -> Deriv -> Synthetic)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
from typing import Any, Callable, Optional

import httpx
import numpy as np
import pandas as pd
from websockets.sync.client import connect as ws_sync_connect

logger = logging.getLogger(__name__)

BYBIT_PUBLIC_KLINE_URL = "https://api.bybit.com/v5/market/kline"
COINBASE_CANDLES_URL_TEMPLATE = (
    "https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
)
DERIV_PUBLIC_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

BYBIT_INTERVAL_MAP: dict[str, str] = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}

COINBASE_GRANULARITY_MAP: dict[str, str] = {
    "1m": "ONE_MINUTE",
    "5m": "FIVE_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h": "ONE_HOUR",
    "4h": "FOUR_HOUR",
    "1d": "ONE_DAY",
}

SYNTHETIC_BASE_PRICES: dict[str, float] = {
    "BTCUSDT": 88000.0,
    "ETHUSDT": 3100.0,
    "SOLUSDT": 180.0,
    "R_100": 1450.0,
    "R_75": 350000.0,
    "FRXEURUSD": 1.0850,
    "EUR_USD": 1.0850,
    "FRXGBPUSD": 1.2700,
    "GBP_USD": 1.2700,
    "XAU_USD": 2650.0,
    "FRXXAUUSD": 2650.0,
}


def _normalize_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a clean, deduplicated, chronologically ascending OHLCV DataFrame."""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(1.0).clip(lower=0.0)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    return df


class HistoricalDataFetcher:
    """Fetches multi-day historical OHLCV candles with automatic geo-block and offline fallback."""

    def __init__(
        self,
        http_client: Optional[httpx.Client] = None,
        ws_connect: Optional[Callable[..., Any]] = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(timeout=timeout_seconds)
        self._ws_connect = ws_connect or ws_sync_connect
        self._timeout = timeout_seconds

    @staticmethod
    def is_deriv_or_forex_symbol(symbol: str) -> bool:
        """Return True if symbol routes to Deriv/Forex rather than Crypto exchanges."""
        cleaned = (symbol or "").strip()
        upper = cleaned.upper()
        if upper.startswith(("R_", "FRX", "1HZ", "CRASH", "BOOM")):
            return True
        if "_" in cleaned and not upper.endswith(("USDT", "USDC")):
            return True
        return False

    @staticmethod
    def map_to_deriv_symbol(symbol: str) -> str:
        """Map EUR_USD -> frxEURUSD while preserving R_100, R_75, frxEURUSD, etc."""
        cleaned = (symbol or "").strip()
        if cleaned.startswith(("R_", "1HZ", "frx", "FRX")):
            return f"frx{cleaned[3:].upper()}" if cleaned.upper().startswith("FRX") else cleaned.upper()
        if "_" in cleaned:
            return f"frx{cleaned.replace('_', '').upper()}"
        return cleaned

    @staticmethod
    def map_to_coinbase_product(symbol: str) -> str:
        """Map BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD, SOLUSDT -> SOL-USD."""
        upper = (symbol or "BTCUSDT").strip().upper().replace("_", "-")
        if "-" in upper:
            return upper
        if upper.endswith("USDT"):
            return f"{upper[:-4]}-USD"
        if upper.endswith("USDC"):
            return f"{upper[:-4]}-USD"
        if upper.endswith("USD") and len(upper) > 3:
            return f"{upper[:-3]}-USD"
        return f"{upper}-USD"

    def _fetch_bybit_public(
        self,
        symbol: str,
        interval: str,
        days: int,
        now: datetime,
    ) -> pd.DataFrame:
        """Fetch paginated historical klines from Bybit Public V5 REST API."""
        bybit_iv = BYBIT_INTERVAL_MAP.get(interval, "60")
        start_ms = int((now - timedelta(days=days)).timestamp() * 1000)
        end_ms = int(now.timestamp() * 1000)
        max_pages = 25
        rows: list[dict[str, Any]] = []

        for _ in range(max_pages):
            params = {
                "category": "linear",
                "symbol": symbol.upper(),
                "interval": bybit_iv,
                "limit": 1000,
                "end": end_ms,
            }
            resp = self._http_client.get(BYBIT_PUBLIC_KLINE_URL, params=params)
            if resp.status_code == 403:
                raise RuntimeError("Bybit Public V5 HTTP 403 (geo-blocked)")
            if resp.status_code >= 400:
                raise RuntimeError(f"Bybit Public V5 HTTP {resp.status_code}")

            body = resp.json()
            ret_code = int(body.get("retCode", -1))
            if ret_code == 10009:
                raise RuntimeError("Bybit Public V5 retCode 10009 (US geo-blocked)")
            if ret_code != 0:
                raise RuntimeError(f"Bybit Public V5 error retCode={ret_code}: {body.get('retMsg')}")

            raw_list = body.get("result", {}).get("list") or []
            if not raw_list:
                break

            earliest_ms = end_ms
            for entry in raw_list:
                ts_ms = int(entry[0])
                if ts_ms < earliest_ms:
                    earliest_ms = ts_ms
                if ts_ms >= start_ms:
                    rows.append(
                        {
                            "timestamp": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                            "open": float(entry[1]),
                            "high": float(entry[2]),
                            "low": float(entry[3]),
                            "close": float(entry[4]),
                            "volume": float(entry[5]),
                        }
                    )

            if earliest_ms <= start_ms or len(raw_list) < 2:
                break
            end_ms = earliest_ms - 1

        return _normalize_df(rows)

    def _fetch_coinbase_public(
        self,
        symbol: str,
        interval: str,
        days: int,
        now: datetime,
    ) -> pd.DataFrame:
        """Fetch paginated historical candles from Coinbase Public Advanced Trade API (<=300 bars/chunk)."""
        product_id = self.map_to_coinbase_product(symbol)
        granularity = COINBASE_GRANULARITY_MAP.get(interval, "ONE_HOUR")
        step_sec = INTERVAL_SECONDS.get(interval, 3600)
        chunk_bars = 290
        chunk_span_sec = chunk_bars * step_sec

        start_sec = int((now - timedelta(days=days)).timestamp())
        chunk_end = int(now.timestamp())
        url = COINBASE_CANDLES_URL_TEMPLATE.format(product_id=product_id)

        rows: list[dict[str, Any]] = []
        max_chunks = 80
        for _ in range(max_chunks):
            if chunk_end <= start_sec:
                break
            chunk_start = max(start_sec, chunk_end - chunk_span_sec)
            params = {
                "start": str(chunk_start),
                "end": str(chunk_end),
                "granularity": granularity,
            }
            resp = self._http_client.get(url, params=params)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Coinbase Public Candles HTTP {resp.status_code} for {product_id}"
                )

            body = resp.json()
            candles = body.get("candles") or []
            for item in candles:
                ts_sec = int(item["start"])
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(ts_sec, tz=timezone.utc),
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": float(item.get("volume", 1.0) or 1.0),
                    }
                )

            if chunk_start <= start_sec:
                break
            chunk_end = chunk_start - 1

        return _normalize_df(rows)

    def _fetch_deriv_public(
        self,
        symbol: str,
        interval: str,
        days: int,
        now: datetime,
    ) -> pd.DataFrame:
        """Fetch paginated historical candles from Deriv Public WebSocket ticks_history."""
        deriv_symbol = self.map_to_deriv_symbol(symbol)
        granularity = INTERVAL_SECONDS.get(interval, 3600)
        start_epoch = int((now - timedelta(days=days)).timestamp())
        end_param: int | str = "latest"
        rows: list[dict[str, Any]] = []
        max_pages = 20

        with self._ws_connect(DERIV_PUBLIC_WS_URL, open_timeout=self._timeout) as ws:
            for _ in range(max_pages):
                payload = {
                    "ticks_history": deriv_symbol,
                    "adjust_start_time": 1,
                    "count": 1000,
                    "end": end_param,
                    "granularity": granularity,
                    "style": "candles",
                }
                ws.send(json.dumps(payload))
                raw = ws.recv()
                body = json.loads(raw) if isinstance(raw, (str, bytes)) else raw

                if isinstance(body, dict) and body.get("error"):
                    err = body["error"]
                    raise RuntimeError(
                        f"Deriv ticks_history error ({err.get('code')}): {err.get('message')}"
                    )

                candles = body.get("candles") or []
                if not candles:
                    break

                earliest_epoch: Optional[int] = None
                for c in candles:
                    ep = int(c["epoch"])
                    if earliest_epoch is None or ep < earliest_epoch:
                        earliest_epoch = ep
                    if ep >= start_epoch:
                        rows.append(
                            {
                                "timestamp": datetime.fromtimestamp(ep, tz=timezone.utc),
                                "open": float(c["open"]),
                                "high": float(c["high"]),
                                "low": float(c["low"]),
                                "close": float(c["close"]),
                                "volume": float(c.get("volume", 1.0) or 1.0),
                            }
                        )

                if earliest_epoch is None or earliest_epoch <= start_epoch or len(candles) < 2:
                    break
                end_param = earliest_epoch - 1

        return _normalize_df(rows)

    @staticmethod
    def generate_synthetic_fallback(
        symbol: str,
        interval: str = "1h",
        days: int = 60,
        now: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate a deterministic multi-regime OHLCV dataset (trending + ranging + squeeze waves)."""
        ref_now = now or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        step_sec = INTERVAL_SECONDS.get(interval, 3600)
        n_bars = max(320, int((days * 86400) // step_sec))

        seed_bytes = hashlib.sha256(f"{symbol.upper()}:{interval}:{days}".encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:4], "big")
        rng = np.random.default_rng(seed)

        base_price = SYNTHETIC_BASE_PRICES.get(symbol.upper(), 100.0)
        is_fx = base_price < 10.0
        vol_scale = 0.0014 if is_fx else 0.0065

        closes = np.empty(n_bars, dtype=float)
        price = float(base_price)
        cycle_len = max(120, n_bars // 4)

        for i in range(n_bars):
            phase = (i % cycle_len) / float(cycle_len)
            if phase < 0.30:
                # Uptrend phase
                drift = 0.22 * vol_scale
                noise = rng.normal(0.0, vol_scale * 0.75)
            elif phase < 0.55:
                # Range-bound oscillation
                drift = math.sin(i * 0.45) * vol_scale * 0.45
                noise = rng.normal(0.0, vol_scale * 0.55)
            elif phase < 0.72:
                # Low-volatility compression squeeze
                drift = 0.02 * vol_scale
                noise = rng.normal(0.0, vol_scale * 0.20)
            elif phase < 0.86:
                # Breakout expansion impulse
                drift = 0.38 * vol_scale
                noise = rng.normal(0.0, vol_scale * 1.05)
            else:
                # Downtrend pullback
                drift = -0.24 * vol_scale
                noise = rng.normal(0.0, vol_scale * 0.80)

            price = max(price * (1.0 + drift + noise), 1e-5)
            closes[i] = price

        start_ts = ref_now - timedelta(seconds=step_sec * n_bars)
        rows: list[dict[str, Any]] = []
        for i in range(n_bars):
            close_p = float(closes[i])
            open_p = float(closes[i - 1]) if i > 0 else close_p
            wick_up = abs(float(rng.normal(0.0, vol_scale * 0.45)))
            wick_dn = abs(float(rng.normal(0.0, vol_scale * 0.45)))
            high_p = max(open_p, close_p) * (1.0 + wick_up)
            low_p = max(min(open_p, close_p) * (1.0 - wick_dn), 1e-6)
            base_vol = 200.0 + 120.0 * abs(close_p - open_p) / max(open_p * vol_scale, 1e-9)
            volume = float(max(10.0, base_vol * float(rng.uniform(0.65, 1.85))))
            rows.append(
                {
                    "timestamp": start_ts + timedelta(seconds=step_sec * i),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": round(volume, 2),
                }
            )

        return _normalize_df(rows)

    def fetch(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 60,
    ) -> tuple[pd.DataFrame, str]:
        """Fetch chronological OHLCV history with automatic geo-resilient & offline fallback."""
        cleaned_symbol = (symbol or "BTCUSDT").strip()
        cleaned_interval = (interval or "1h").strip().lower()
        if cleaned_interval not in INTERVAL_SECONDS:
            raise ValueError(
                f"Unsupported interval '{interval}'. Supported: {', '.join(INTERVAL_SECONDS)}"
            )
        clamped_days = max(1, int(days))
        now = datetime.now(timezone.utc)

        if self.is_deriv_or_forex_symbol(cleaned_symbol):
            try:
                df = self._fetch_deriv_public(cleaned_symbol, cleaned_interval, clamped_days, now)
                if len(df) >= 20:
                    return df, "DERIV_PUBLIC"
            except Exception as exc:
                logger.warning("Deriv Public fetch failed for %s: %s", cleaned_symbol, exc)
        else:
            # 1. Try Bybit Public V5 first
            try:
                df = self._fetch_bybit_public(cleaned_symbol, cleaned_interval, clamped_days, now)
                if len(df) >= 20:
                    return df, "BYBIT_PUBLIC"
            except Exception as exc:
                logger.info("Bybit Public fetch failed for %s (%s); falling back to Coinbase Public.", cleaned_symbol, exc)

            # 2. Automatic geo-resilient fallback to Coinbase Public Advanced Trade Candles
            try:
                df = self._fetch_coinbase_public(cleaned_symbol, cleaned_interval, clamped_days, now)
                if len(df) >= 20:
                    return df, "COINBASE_PUBLIC"
            except Exception as exc:
                logger.warning("Coinbase Public fetch failed for %s: %s", cleaned_symbol, exc)

        # 3. Offline / sandboxed deterministic fallback
        synthetic_df = self.generate_synthetic_fallback(
            cleaned_symbol,
            interval=cleaned_interval,
            days=clamped_days,
            now=now,
        )
        return synthetic_df, "SYNTHETIC_FALLBACK"

    def close(self) -> None:
        """Close owned HTTP client if applicable."""
        if self._owns_http_client:
            self._http_client.close()
