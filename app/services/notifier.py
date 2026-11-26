"""HTML-safe asynchronous Telegram sleep alerts for APEX-PULSE."""

from __future__ import annotations

import html
import logging
from typing import Any, Optional

import httpx

from app.config import settings

logger = logging.getLogger("apex_pulse.notifier")

MAX_TELEGRAM_HTML_CHARS = 4000


def escape_html(value: Any) -> str:
    """Escape dynamic values for safe insertion into Telegram HTML parse_mode messages."""
    return html.escape(str(value), quote=True)


def truncate_html_safe(text: str, max_len: int = MAX_TELEGRAM_HTML_CHARS) -> str:
    """Truncate text to <= max_len characters without leaving dangling HTML entities or tags."""
    if len(text) <= max_len:
        return text
    suffix = "\n... (truncated)"
    cut = max(0, max_len - len(suffix))
    candidate = text[:cut]
    # Avoid splitting inside an HTML tag like <b... or </code>
    last_open = candidate.rfind("<")
    last_close = candidate.rfind(">")
    if last_open > last_close:
        candidate = candidate[:last_open]
    # Close any unclosed <b> or <code> tags cleanly
    for tag in ("code", "b"):
        opens = candidate.count(f"<{tag}>")
        closes = candidate.count(f"</{tag}>")
        if opens > closes:
            candidate += f"</{tag}>" * (opens - closes)
    return (candidate[: max_len - len(suffix)] + suffix)[:max_len]


class TelegramNotifier:
    """Non-blocking HTML-safe Telegram Bot API notifier."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._bot_token_override = bot_token
        self._chat_id_override = chat_id
        self.timeout_seconds = timeout_seconds

    @property
    def bot_token(self) -> str:
        raw = settings.TELEGRAM_BOT_TOKEN if self._bot_token_override is None else self._bot_token_override
        return (raw or "").strip()

    @property
    def chat_id(self) -> str:
        raw = settings.TELEGRAM_CHAT_ID if self._chat_id_override is None else self._chat_id_override
        return (raw or "").strip()

    def is_configured(self) -> bool:
        """Return True when both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are non-empty."""
        return bool(self.bot_token and self.chat_id)

    async def send_message(self, html_message: str) -> tuple[bool, str]:
        """Dispatch an HTML message to Telegram, never raising on network or API errors."""
        if not self.is_configured():
            return False, "Telegram not configured (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is blank)."

        safe_text = truncate_html_safe(html_message, MAX_TELEGRAM_HTML_CHARS)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": safe_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(url, json=payload)
            if 200 <= resp.status_code < 300:
                return True, "Alert delivered."
            reason = f"Telegram API HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning("Telegram alert delivery failed: %s", reason)
            return False, reason
        except Exception as exc:
            reason = f"Telegram request error: {exc}"
            logger.warning("Telegram alert exception (non-fatal): %s", reason)
            return False, reason

    async def send_test_alert(self) -> tuple[bool, str]:
        """Send a diagnostic test alert for POST /api/alerts/test."""
        mode = escape_html(settings.TRADING_MODE)
        msg = (
            "<b>APEX-PULSE Test Alert</b>\n"
            f"Mode: <code>{mode}</code>\n"
            "Status: Telegram HTML notifications are connected and operational."
        )
        return await self.send_message(msg)

    async def notify_trade_opened(
        self,
        *,
        symbol: str,
        side: str,
        market: str,
        qty: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str,
        reason: str = "",
    ) -> bool:
        """Notify when a new trade is opened."""
        msg = (
            "<b>APEX-PULSE Trade Opened</b>\n"
            f"Pair: <code>{escape_html(symbol)}</code> ({escape_html(market)})\n"
            f"Side: <b>{escape_html(side)}</b> | Qty: <code>{escape_html(qty)}</code>\n"
            f"Strategy: <code>{escape_html(strategy_name)}</code>\n"
            f"Entry: <code>{escape_html(entry_price)}</code>\n"
            f"SL: <code>{escape_html(stop_loss)}</code> | TP: <code>{escape_html(take_profit)}</code>"
        )
        if reason:
            msg += f"\nReason: {escape_html(reason)}"
        sent, _ = await self.send_message(msg)
        return sent

    async def notify_partial_scale_out(
        self,
        *,
        symbol: str,
        side: str,
        trade_id: int,
        partial_pnl_usd: float,
        new_stop_loss: float,
        remaining_qty: Optional[float] = None,
    ) -> bool:
        """Notify when a +1.0R partial take-profit scale-out locks profit."""
        pnl_str = f"{partial_pnl_usd:+.2f}"
        msg = (
            "<b>APEX-PULSE +1.0R Partial Scale-Out</b>\n"
            f"Pair: <code>{escape_html(symbol)}</code> #{escape_html(trade_id)} ({escape_html(side)})\n"
            f"Locked PnL (50%): <b>${escape_html(pnl_str)}</b>\n"
            f"Runner SL Moved To: <code>{escape_html(new_stop_loss)}</code>"
        )
        if remaining_qty is not None:
            msg += f"\nRemaining Qty: <code>{escape_html(remaining_qty)}</code>"
        sent, _ = await self.send_message(msg)
        return sent

    async def notify_trade_closed(
        self,
        *,
        symbol: str,
        side: str,
        market: str,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        strategy_name: str,
        reason: str = "",
    ) -> bool:
        """Notify when an open position closes."""
        pnl_str = f"{pnl_usd:+.2f}"
        pct_str = f"{pnl_pct:+.2f}%"
        outcome = "WIN" if pnl_usd >= 0 else "LOSS"
        msg = (
            f"<b>APEX-PULSE Trade Closed ({outcome})</b>\n"
            f"Pair: <code>{escape_html(symbol)}</code> ({escape_html(market)}) | {escape_html(side)}\n"
            f"Strategy: <code>{escape_html(strategy_name)}</code>\n"
            f"Exit: <code>{escape_html(exit_price)}</code>\n"
            f"PnL: <b>${escape_html(pnl_str)}</b> (<code>{escape_html(pct_str)}</code>)"
        )
        if reason:
            msg += f"\nNote: {escape_html(reason)}"
        sent, _ = await self.send_message(msg)
        return sent

    async def notify_kill_switch(
        self,
        *,
        reason: str,
        closed_positions: int = 0,
    ) -> bool:
        """Notify immediately when the Emergency Kill Switch or 3% circuit breaker triggers."""
        msg = (
            "<b>CRITICAL: APEX-PULSE KILL SWITCH ACTIVATED</b>\n"
            f"Reason: {escape_html(reason)}\n"
            f"Positions Closed: <code>{escape_html(closed_positions)}</code>\n"
            "Trading is now HALTED until manually resumed from the dashboard."
        )
        sent, _ = await self.send_message(msg)
        return sent

    async def notify_daily_summary(self, summary_dict: dict[str, Any]) -> bool:
        """Send a formatted daily performance & risk shield digest."""
        grade = escape_html(summary_dict.get("health_grade", "CALIBRATING"))
        market = escape_html(summary_dict.get("market", "ALL"))
        total_trades = escape_html(summary_dict.get("total_closed_trades", 0))
        wins = escape_html(summary_dict.get("winning_trades", 0))
        losses = escape_html(summary_dict.get("losing_trades", 0))
        win_rate = escape_html(f"{float(summary_dict.get('win_rate_pct', 0.0)):.1f}%")
        pf = escape_html(f"{float(summary_dict.get('profit_factor', 0.0)):.2f}")
        net_pnl = float(summary_dict.get("net_pnl_usd", 0.0))
        net_pnl_str = escape_html(f"${net_pnl:+.2f}")
        max_dd = escape_html(f"{float(summary_dict.get('max_drawdown_pct', 0.0)):.2f}%")

        plain = summary_dict.get("plain_english") or {}
        headline = escape_html(plain.get("headline", "Daily summary ready."))
        shield = summary_dict.get("risk_shield_report") or {}
        ks_status = "LOCKED" if shield.get("kill_switch_active") else "ARMED"
        skips = escape_html(shield.get("total_protected_skips", 0))

        lines = [
            f"<b>APEX-PULSE Daily Summary ({market})</b>",
            f"Health Grade: <b>{grade}</b> | Net PnL: <b>{net_pnl_str}</b>",
            f"Trades: <code>{total_trades}</code> ({wins}W / {losses}L) | Win Rate: <code>{win_rate}</code>",
            f"Profit Factor: <code>{pf}</code> | Max DD: <code>{max_dd}</code>",
            f"Risk Shield: <code>{ks_status}</code> | Blocked Setups: <code>{skips}</code>",
            f"Summary: {headline}",
        ]
        best = summary_dict.get("best_strategy")
        if isinstance(best, dict) and best.get("name"):
            lines.append(
                f"Top Strategy: <code>{escape_html(best.get('name'))}</code> "
                f"({escape_html(best.get('display_name', best.get('name')))})"
            )

        sent, _ = await self.send_message("\n".join(lines))
        return sent
