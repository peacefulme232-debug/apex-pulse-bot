"""Forex Kill-Zone Session, Weekend Close, Rollover, and Economic News Blackout Guard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Optional

from app.config import settings

HIGH_IMPACT_CURRENCIES = {"USD", "EUR", "GBP", "ALL", ""}


@dataclass(frozen=True)
class SessionStatus:
    """Structured gate result for Forex/Synthetic trading availability."""

    can_trade: bool
    session_name: str
    is_weekend_closed: bool
    is_rollover_pause: bool
    is_news_blackout: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize SessionStatus to a JSON-friendly dictionary."""
        return asdict(self)


class SessionGuard:
    """Guards Forex entries against weekend close, rollover spikes, low liquidity, and macro news."""

    def __init__(
        self,
        killzone_only: Optional[bool] = None,
        news_blackout_minutes: Optional[int] = None,
        news_events: Optional[list[dict[str, Any]]] = None,
        news_file: Optional[str | Path] = None,
        enable_default_nfp_guard: bool = True,
    ) -> None:
        self.killzone_only = (
            settings.FOREX_KILLZONE_ONLY if killzone_only is None else bool(killzone_only)
        )
        self.news_blackout_minutes = (
            settings.NEWS_BLACKOUT_MINUTES
            if news_blackout_minutes is None
            else int(news_blackout_minutes)
        )
        self.enable_default_nfp_guard = enable_default_nfp_guard
        self._events: list[dict[str, Any]] = []
        self._load_events(news_events=news_events, news_file=news_file)

    @staticmethod
    def _normalize_utc(dt: Optional[datetime]) -> datetime:
        if dt is None:
            return datetime.now(timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _parse_event_timestamp(raw_ts: Any) -> Optional[datetime]:
        if isinstance(raw_ts, datetime):
            return SessionGuard._normalize_utc(raw_ts)
        if isinstance(raw_ts, (int, float)):
            return datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
        if isinstance(raw_ts, str):
            text = raw_ts.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
                return SessionGuard._normalize_utc(parsed)
            except ValueError:
                return None
        return None

    def _load_events(
        self,
        news_events: Optional[list[dict[str, Any]]] = None,
        news_file: Optional[str | Path] = None,
    ) -> None:
        raw_items: list[dict[str, Any]] = []

        # 1. Optional JSON file (news_events.json)
        candidate_path = Path(news_file) if news_file is not None else Path("news_events.json")
        if candidate_path.exists() and candidate_path.is_file():
            try:
                loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    raw_items.extend(item for item in loaded if isinstance(item, dict))
            except (OSError, ValueError):
                pass

        # 2. Optional env override (NEWS_EVENTS_JSON)
        env_json = os.getenv("NEWS_EVENTS_JSON", "").strip()
        if env_json:
            try:
                loaded_env = json.loads(env_json)
                if isinstance(loaded_env, list):
                    raw_items.extend(item for item in loaded_env if isinstance(item, dict))
            except ValueError:
                pass

        # 3. Explicit in-memory override/additions
        if news_events:
            raw_items.extend(item for item in news_events if isinstance(item, dict))

        parsed_events: list[dict[str, Any]] = []
        for item in raw_items:
            ts = self._parse_event_timestamp(item.get("timestamp") or item.get("time_utc"))
            if ts is None:
                continue
            parsed_events.append(
                {
                    "name": str(item.get("name") or item.get("event") or "High-Impact Macro Event"),
                    "timestamp": ts,
                    "currency": str(item.get("currency") or "USD").strip().upper(),
                }
            )
        self._events = parsed_events

    @staticmethod
    def is_synthetic_index(symbol: Optional[str]) -> bool:
        """Return True for Deriv 24/7 Derived/Synthetic indices (R_* or 1HZ*)."""
        if not symbol:
            return False
        upper = symbol.strip().upper()
        return upper.startswith("R_") or upper.startswith("1HZ")

    @staticmethod
    def _classify_session(now_utc: datetime) -> str:
        hour = now_utc.hour
        if 7 <= hour < 12:
            return "LONDON_SESSION"
        if 12 <= hour < 17:
            return "LONDON_NY_OVERLAP"
        if 17 <= hour < 20:
            return "NEW_YORK_PM"
        return "ASIAN_LOW_LIQUIDITY"

    @staticmethod
    def _is_weekend_closed(now_utc: datetime) -> bool:
        weekday = now_utc.weekday()  # Mon=0 .. Fri=4, Sat=5, Sun=6
        hm = (now_utc.hour, now_utc.minute)
        if weekday == 4 and hm >= (21, 45):
            return True
        if weekday == 5:
            return True
        if weekday == 6 and hm < (22, 15):
            return True
        return False

    @staticmethod
    def _is_rollover_pause(now_utc: datetime) -> bool:
        mins = now_utc.hour * 60 + now_utc.minute
        # Daily 21:55 - 22:10 UTC rollover spread spike window
        return (21 * 60 + 55) <= mins <= (22 * 60 + 10)

    def _check_news_blackout(self, now_utc: datetime) -> Optional[str]:
        if self.news_blackout_minutes <= 0:
            return None

        window_seconds = self.news_blackout_minutes * 60
        for ev in self._events:
            if ev["currency"] not in HIGH_IMPACT_CURRENCIES:
                continue
            delta_sec = abs((now_utc - ev["timestamp"]).total_seconds())
            if delta_sec <= window_seconds:
                return f"{ev['name']} ({ev['currency']})"

        # Safe default recurring monthly high-impact guard:
        # US Non-Farm Payrolls (NFP) — first Friday of each month, 12:30–13:30 UTC
        if self.enable_default_nfp_guard:
            if now_utc.weekday() == 4 and 1 <= now_utc.day <= 7:
                mins = now_utc.hour * 60 + now_utc.minute
                if (12 * 60 + 30) <= mins <= (13 * 60 + 30):
                    return "US Non-Farm Payrolls (NFP)"

        return None

    def evaluate(
        self,
        symbol: str = "EUR_USD",
        now_utc: Optional[datetime] = None,
    ) -> SessionStatus:
        """Evaluate whether a new Forex/Synthetic entry is permitted at the given UTC time."""
        current_utc = self._normalize_utc(now_utc)

        # 1. Deriv 24/7 Synthetic Indices bypass weekend, rollover, kill-zone, and macro news gates
        if self.is_synthetic_index(symbol):
            return SessionStatus(
                can_trade=True,
                session_name="SYNTHETIC_24_7",
                is_weekend_closed=False,
                is_rollover_pause=False,
                is_news_blackout=False,
                reason=f"24/7 Synthetic Index ({symbol}) bypasses Forex market hours",
            )

        # 2. Forex Weekend Close (Fri 21:45 UTC -> Sun 22:15 UTC)
        if self._is_weekend_closed(current_utc):
            return SessionStatus(
                can_trade=False,
                session_name="WEEKEND_CLOSED",
                is_weekend_closed=True,
                is_rollover_pause=False,
                is_news_blackout=False,
                reason="Forex weekend market close",
            )

        # 3. Daily Rollover Spread Spike Pause (21:55 - 22:10 UTC)
        if self._is_rollover_pause(current_utc):
            return SessionStatus(
                can_trade=False,
                session_name="ROLLOVER_PAUSE",
                is_weekend_closed=False,
                is_rollover_pause=True,
                is_news_blackout=False,
                reason="Daily rollover spread spike protection",
            )

        session_name = self._classify_session(current_utc)

        # 4. High-Impact Economic News Blackout Guard
        blackout_event = self._check_news_blackout(current_utc)
        if blackout_event is not None:
            return SessionStatus(
                can_trade=False,
                session_name=session_name,
                is_weekend_closed=False,
                is_rollover_pause=False,
                is_news_blackout=True,
                reason=f"High-impact news blackout: {blackout_event}",
            )

        # 5. Kill-Zone Liquidity Filter
        if self.killzone_only and session_name == "ASIAN_LOW_LIQUIDITY":
            return SessionStatus(
                can_trade=False,
                session_name="ASIAN_LOW_LIQUIDITY",
                is_weekend_closed=False,
                is_rollover_pause=False,
                is_news_blackout=False,
                reason="Low-liquidity session pause",
            )

        return SessionStatus(
            can_trade=True,
            session_name=session_name,
            is_weekend_closed=False,
            is_rollover_pause=False,
            is_news_blackout=False,
            reason=f"Active Forex session ({session_name})",
        )
