"""Phone PIN Gate, HMAC-SHA256 session cookies, anti-spoofing rate limiter, and Cron auth."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import secrets
import threading
import time
from collections import deque
from typing import Optional

from fastapi import Header, HTTPException, Request, Response

from app.config import settings

logger = logging.getLogger("apex_pulse.security")

SESSION_COOKIE_NAME = "apex_session"
_EPHEMERAL_SESSION_SECRET: str = secrets.token_hex(32)


def is_auth_enabled() -> bool:
    """Return True when DASHBOARD_PIN is configured (non-empty)."""
    return bool((settings.DASHBOARD_PIN or "").strip())


def get_effective_session_secret() -> str:
    """Return configured SESSION_SECRET or ephemeral per-process CSPRNG fallback."""
    configured = (settings.SESSION_SECRET or "").strip()
    if configured:
        return configured
    return _EPHEMERAL_SESSION_SECRET


def log_security_startup_warnings() -> None:
    """Emit clear startup security warnings for unlocked or ephemeral-secret configs."""
    pin_set = is_auth_enabled()
    secret_set = bool((settings.SESSION_SECRET or "").strip())
    if not pin_set:
        logger.warning(
            "DASHBOARD_PIN is blank: Dashboard PIN Gate is UNLOCKED. "
            "Set DASHBOARD_PIN and SESSION_SECRET before exposing APEX-PULSE to the internet."
        )
    elif not secret_set:
        logger.critical(
            "DASHBOARD_PIN is set but SESSION_SECRET is blank! Using ephemeral per-process "
            "CSPRNG secret — Cloud Run cold starts will invalidate sessions and force re-login. "
            "Set a persistent SESSION_SECRET in environment variables."
        )


def verify_pin(candidate_pin: str) -> bool:
    """Constant-time PIN verification against active DASHBOARD_PIN."""
    expected = (settings.DASHBOARD_PIN or "").strip()
    if not expected:
        return True
    provided = (candidate_pin or "").strip()
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + padding).encode("ascii"))


def create_session_token(ttl_hours: Optional[int] = None, now: Optional[float] = None) -> str:
    """Create an HMAC-SHA256 signed session token: base64url(payload).hex_signature."""
    current_ts = int(time.time() if now is None else now)
    hours = settings.SESSION_TTL_HOURS if ttl_hours is None else ttl_hours
    payload_obj = {
        "iat": current_ts,
        "exp": current_ts + int(hours) * 3600,
    }
    payload_bytes = json.dumps(payload_obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64url_encode(payload_bytes)
    secret = get_effective_session_secret().encode("utf-8")
    sig = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_session_token(token: Optional[str], now: Optional[float] = None) -> bool:
    """Verify HMAC-SHA256 signature (constant-time) and expiration timestamp."""
    if not token or "." not in token:
        return False
    parts = token.split(".")
    if len(parts) != 2:
        return False

    payload_b64, provided_sig = parts
    secret = get_effective_session_secret().encode("utf-8")
    expected_sig = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(provided_sig.encode("ascii"), expected_sig.encode("ascii")):
        return False

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        exp = int(payload["exp"])
    except (ValueError, KeyError, TypeError):
        return False

    current_ts = int(time.time() if now is None else now)
    return current_ts < exp


def is_secure_request(request: Request) -> bool:
    """Detect HTTPS either directly or behind a TLS-terminating proxy (Cloud Run)."""
    if request.url.scheme == "https":
        return True
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    first_proto = forwarded_proto.split(",")[0].strip().lower() if forwarded_proto else ""
    return first_proto == "https"


def set_session_cookie(response: Response, request: Request, token: Optional[str] = None) -> None:
    """Attach the signed HttpOnly apex_session cookie to the response."""
    cookie_val = token or create_session_token()
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=cookie_val,
        max_age=int(settings.SESSION_TTL_HOURS) * 3600,
        httponly=True,
        samesite="lax",
        secure=is_secure_request(request),
        path="/",
    )


def clear_session_cookie(response: Response, request: Request) -> None:
    """Expire and remove the apex_session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=is_secure_request(request),
        path="/",
    )


def extract_client_ip(request: Request) -> str:
    """Safely extract client IP without trusting forgeable leftmost X-Forwarded-For hops.

    Prefers request.client.host (populated by Uvicorn --proxy-headers in production).
    Falls back to the rightmost non-empty IP in X-Forwarded-For when client.host is absent
    (or in ASGI TestClient where client.host is the sentinel 'testclient').
    """
    host = request.client.host if (request.client and request.client.host) else ""
    if host and host != "testclient":
        return host.strip()

    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        hops = [part.strip() for part in xff.split(",") if part.strip()]
        if hops:
            return hops[-1]

    return host.strip() if host else "unknown"


class BruteForceRateLimiter:
    """Sliding-window dual rate limiter: per-IP cap AND global anti-spoofing cap."""

    def __init__(
        self,
        window_seconds: int = 900,
        max_per_ip: int = 5,
        max_global: int = 10,
    ) -> None:
        self.window_seconds = window_seconds
        self.max_per_ip = max_per_ip
        self.max_global = max_global
        self._lock = threading.Lock()
        self._ip_attempts: dict[str, deque[float]] = {}
        self._global_attempts: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._global_attempts and self._global_attempts[0] <= cutoff:
            self._global_attempts.popleft()

        empty_ips: list[str] = []
        for ip, dq in self._ip_attempts.items():
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if not dq:
                empty_ips.append(ip)
        for ip in empty_ips:
            del self._ip_attempts[ip]

    def check_allowed(self, ip: str, now: Optional[float] = None) -> tuple[bool, int]:
        """Check whether an IP is allowed to attempt login.

        Returns (allowed, retry_after_seconds).
        """
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._prune(ts)
            ip_dq = self._ip_attempts.get(ip)

            retry_candidates: list[int] = []
            if ip_dq and len(ip_dq) >= self.max_per_ip:
                remaining = self.window_seconds - (ts - ip_dq[0])
                retry_candidates.append(max(1, math.ceil(remaining)))

            if len(self._global_attempts) >= self.max_global:
                remaining = self.window_seconds - (ts - self._global_attempts[0])
                retry_candidates.append(max(1, math.ceil(remaining)))

            if retry_candidates:
                return False, max(retry_candidates)
            return True, 0

    def record_failure(self, ip: str, now: Optional[float] = None) -> None:
        """Record a failed PIN attempt for both per-IP and global sliding windows."""
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._prune(ts)
            self._ip_attempts.setdefault(ip, deque()).append(ts)
            self._global_attempts.append(ts)

    def reset(self) -> None:
        """Clear all tracked attempts (for unit test isolation)."""
        with self._lock:
            self._ip_attempts.clear()
            self._global_attempts.clear()


rate_limiter = BruteForceRateLimiter(window_seconds=900, max_per_ip=5, max_global=10)


def reset_rate_limiter() -> None:
    """Reset the global brute-force rate limiter state (intended for unit tests)."""
    rate_limiter.reset()


def is_request_authenticated(request: Request) -> bool:
    """Return True if auth is disabled or the request carries a valid session cookie."""
    if not is_auth_enabled():
        return True
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return verify_session_token(token)


def require_dashboard_auth(request: Request) -> None:
    """FastAPI dependency enforcing the Phone PIN Gate when DASHBOARD_PIN is set."""
    if not is_auth_enabled():
        return
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not verify_session_token(token):
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Unlock the dashboard with your PIN.",
        )


def verify_cron_secret(
    x_cron_secret: Optional[str] = Header(default=None, alias="X-Cron-Secret"),
) -> None:
    """FastAPI dependency enforcing X-Cron-Secret via constant-time comparison."""
    expected = (settings.CRON_SECRET or "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Cron endpoint disabled: CRON_SECRET is not configured.",
        )
    if not x_cron_secret or not hmac.compare_digest(
        x_cron_secret.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Cron-Secret header.",
        )
