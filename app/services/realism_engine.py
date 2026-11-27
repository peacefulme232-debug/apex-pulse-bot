"""Exchange-accurate Realism Engine (exact Decimal math) for Bybit UTA Linear & Forex MT5 ECN/STP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP
import logging
import threading
import time
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger("apex_pulse.realism_engine")

# ---------------------------------------------------------------------------
# Exact Decimal Constants & Quantization Helpers
# ---------------------------------------------------------------------------

ZERO = Decimal("0")
ONE = Decimal("1")
TWO = Decimal("2")
HUNDRED = Decimal("100")
BPS_DIVISOR = Decimal("10000")

Q6 = Decimal("0.000001")
Q8 = Decimal("0.00000001")

BYBIT_MAKER_FEE_RATE = Decimal("0.00020")  # 0.020% VIP 0 Limit / TP Limit
BYBIT_TAKER_FEE_RATE = Decimal("0.00055")  # 0.055% VIP 0 Market / SL
BYBIT_DEFAULT_FUNDING_RATE = Decimal("0.0001")  # +0.01% per 8h
BYBIT_FUNDING_HOURS_UTC: tuple[int, ...] = (0, 8, 16)

BYBIT_INSTRUMENTS_URL = "https://api.bybit.com/v5/market/instruments-info?category=linear"
BYBIT_SPEC_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours
BYBIT_HTTP_TIMEOUT_SECONDS = 5.0

FOREX_RAW_ECN_COMMISSION_PER_LOT_SIDE = Decimal("3.50")  # $7.00 round-turn ($0.07 per 0.01 lot)
FOREX_STANDARD_STP_COMMISSION_PER_LOT_SIDE = Decimal("0.00")
FOREX_MARGIN_CALL_LEVEL_PCT = Decimal("100")
FOREX_STOP_OUT_LEVEL_PCT = Decimal("50")
NY_TZ = ZoneInfo("America/New_York")


def to_decimal(value: Decimal | int | float | str | None, default: Decimal = ZERO) -> Decimal:
    """Safely convert any numeric input to exact Decimal without float binary drift."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def q6(value: Decimal) -> Decimal:
    """Quantize monetary values to NUMERIC(18, 6)."""
    return value.quantize(Q6, rounding=ROUND_HALF_UP)


def q8(value: Decimal) -> Decimal:
    """Quantize price values to NUMERIC(18, 8)."""
    return value.quantize(Q8, rounding=ROUND_HALF_UP)


def ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round value UP to the nearest multiple of step."""
    if step <= ZERO:
        return value
    units = (value / step).to_integral_value(rounding=ROUND_CEILING)
    return units * step


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round value DOWN to the nearest multiple of step."""
    if step <= ZERO:
        return value
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return units * step


def round_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round value to the nearest multiple of step (ROUND_HALF_UP)."""
    if step <= ZERO:
        return value
    units = (value / step).to_integral_value(rounding=ROUND_HALF_UP)
    return units * step


def quantize_price(price: Decimal, tick_size: Decimal) -> Decimal:
    """Quantize a fill or trigger price to the instrument's tick_size."""
    if tick_size <= ZERO:
        return q8(price)
    return round_to_step(price, tick_size)


def normalize_side(side: str) -> str:
    """Normalize BUY/LONG -> LONG and SELL/SHORT -> SHORT."""
    cleaned = (side or "").strip().upper()
    if cleaned in {"BUY", "LONG"}:
        return "LONG"
    if cleaned in {"SELL", "SHORT"}:
        return "SHORT"
    raise ValueError(f"Unsupported side: {side!r}")


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def calculate_net_pnl(
    gross_pnl: Decimal | int | float | str,
    entry_fee: Decimal | int | float | str,
    exit_fee: Decimal | int | float | str,
    funding_or_swap_fee: Decimal | int | float | str,
) -> Decimal:
    """Exact net PnL: gross_pnl - entry_fee - exit_fee - funding_or_swap_fee."""
    return q6(
        to_decimal(gross_pnl)
        - to_decimal(entry_fee)
        - to_decimal(exit_fee)
        - to_decimal(funding_or_swap_fee)
    )


def apply_negative_balance_protection(
    active_balance: Decimal | int | float | str,
    realized_delta: Decimal | int | float | str,
) -> Decimal:
    """Apply Bybit negative balance protection: floor active_balance >= 0."""
    updated = to_decimal(active_balance) + to_decimal(realized_delta)
    return q6(max(ZERO, updated))


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BybitContractSpec:
    """Bybit USDT Linear Perpetual contract specification (VIP 0 UTA)."""

    symbol: str
    min_qty: Decimal
    qty_step: Decimal
    tick_size: Decimal
    min_notional: Decimal
    mmr: Decimal
    mm_deduction: Decimal
    max_lev: int
    spread_bps: Decimal
    funding_interval_hours: int = 8


@dataclass(frozen=True)
class ForexContractSpec:
    """Forex / MT5 ECN & STP instrument specification."""

    symbol: str
    contract_size: Decimal
    pip: Decimal
    tick_size: Decimal
    min_lot: Decimal
    lot_step: Decimal
    raw_spread_pips: Decimal
    stp_spread_pips: Decimal
    mkt_slip_pips: Decimal
    sl_slip_pips: Decimal
    swap_long: Decimal   # USD per 1.00 lot per night (negative = debit, positive = credit)
    swap_short: Decimal  # USD per 1.00 lot per night (negative = debit, positive = credit)


@dataclass(frozen=True)
class FillSimulation:
    """Deterministic simulated fill result with explicit spread, slippage, and fee breakdown."""

    requested_price: Decimal
    filled_price: Decimal
    fee: Decimal
    spread_cost: Decimal
    slippage_cost: Decimal
    order_type: str


@dataclass(frozen=True)
class BybitMarginMetrics:
    """Exact Bybit UTA Isolated Margin & Liquidation calculation output."""

    notional: Decimal
    fee_to_close: Decimal
    initial_margin: Decimal
    entry_fee: Decimal
    required_cash: Decimal
    maintenance_margin: Decimal
    liquidation_price: Decimal
    bankruptcy_price: Decimal


@dataclass(frozen=True)
class FundingResult:
    """Idempotent Bybit 8h funding fee calculation result."""

    fee_amount: Decimal  # positive = trader pays, negative = trader receives
    periods_applied: int
    new_last_funding_applied_at: Optional[datetime]


@dataclass(frozen=True)
class SwapResult:
    """Idempotent Forex 17:00 America/New_York rollover swap calculation result."""

    fee_amount: Decimal  # positive = trader pays, negative = trader receives
    rollover_days_applied: int
    total_multiplier: int
    new_last_swap_applied_date: Optional[date]

    @property
    def applied_date_str(self) -> Optional[str]:
        """Return YYYY-MM-DD date string for idempotency persistence."""
        return (
            self.new_last_swap_applied_date.isoformat()
            if self.new_last_swap_applied_date is not None
            else None
        )


@dataclass(frozen=True)
class ForexPositionSnapshot:
    """Lightweight representation of an open Forex position for margin/stop-out evaluation."""

    trade_id: int
    symbol: str
    side: str
    lots: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    used_margin: Decimal


@dataclass(frozen=True)
class ForexMarginStatus:
    """Forex account margin level, margin call flag, and stop-out liquidation order."""

    forex_equity: Decimal
    used_margin: Decimal
    free_margin: Decimal
    margin_level_pct: Optional[Decimal]
    margin_call: bool
    stop_out: bool
    positions_to_close: tuple[int, ...]


@dataclass(frozen=True)
class OrderSizingResult:
    """Result of realism-aware position sizing and pre-trade validation."""

    approved: bool
    qty: Decimal  # Bybit base qty or Forex units (lots * contract_size)
    lots: Optional[Decimal]  # Forex lots (None for Bybit)
    effective_min_qty: Decimal
    required_margin: Decimal
    available_free_margin: Decimal
    estimated_risk_usd: Decimal
    reason_code: Optional[str] = None
    details: str = ""


@dataclass(frozen=True)
class RoundTurnFrictionEstimate:
    """Pre-trade round-turn friction and net R:R estimate for strategy vetoes."""

    entry_fee: Decimal
    est_exit_fee_tp: Decimal
    est_exit_fee_sl: Decimal
    est_spread_cost: Decimal
    est_slippage_cost: Decimal
    total_friction_tp: Decimal
    gross_tp_profit: Decimal
    net_tp_profit: Decimal
    gross_sl_loss: Decimal
    net_sl_loss: Decimal
    net_rr: Decimal
    fee_drag_pct_of_tp: Decimal


# ---------------------------------------------------------------------------
# A. Bybit USDT Linear Perpetuals (VIP 0 — Official UTA Rules)
# ---------------------------------------------------------------------------

BYBIT_STATIC_SPECS: dict[str, BybitContractSpec] = {
    "BTCUSDT": BybitContractSpec(
        symbol="BTCUSDT",
        min_qty=Decimal("0.001"),
        qty_step=Decimal("0.001"),
        tick_size=Decimal("0.1"),
        min_notional=Decimal("100"),
        mmr=Decimal("0.005"),
        mm_deduction=Decimal("0"),
        max_lev=100,
        spread_bps=Decimal("1.0"),
        funding_interval_hours=8,
    ),
    "ETHUSDT": BybitContractSpec(
        symbol="ETHUSDT",
        min_qty=Decimal("0.01"),
        qty_step=Decimal("0.01"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("20"),
        mmr=Decimal("0.005"),
        mm_deduction=Decimal("0"),
        max_lev=100,
        spread_bps=Decimal("1.5"),
        funding_interval_hours=8,
    ),
    "SOLUSDT": BybitContractSpec(
        symbol="SOLUSDT",
        min_qty=Decimal("0.1"),
        qty_step=Decimal("0.1"),
        tick_size=Decimal("0.01"),
        min_notional=Decimal("5"),
        mmr=Decimal("0.010"),
        mm_deduction=Decimal("0"),
        max_lev=50,
        spread_bps=Decimal("2.0"),
        funding_interval_hours=8,
    ),
}

BYBIT_GENERIC_ALTCOIN_SPEC = BybitContractSpec(
    symbol="ALTUSDT",
    min_qty=Decimal("0.1"),
    qty_step=Decimal("0.1"),
    tick_size=Decimal("0.001"),
    min_notional=Decimal("5"),
    mmr=Decimal("0.010"),
    mm_deduction=Decimal("0"),
    max_lev=25,
    spread_bps=Decimal("3.0"),
    funding_interval_hours=8,
)


def _static_bybit_spec(symbol: str) -> BybitContractSpec:
    sym = (symbol or "").strip().upper()
    if sym in BYBIT_STATIC_SPECS:
        return BYBIT_STATIC_SPECS[sym]
    return BybitContractSpec(
        symbol=sym or "ALTUSDT",
        min_qty=BYBIT_GENERIC_ALTCOIN_SPEC.min_qty,
        qty_step=BYBIT_GENERIC_ALTCOIN_SPEC.qty_step,
        tick_size=BYBIT_GENERIC_ALTCOIN_SPEC.tick_size,
        min_notional=BYBIT_GENERIC_ALTCOIN_SPEC.min_notional,
        mmr=BYBIT_GENERIC_ALTCOIN_SPEC.mmr,
        mm_deduction=BYBIT_GENERIC_ALTCOIN_SPEC.mm_deduction,
        max_lev=BYBIT_GENERIC_ALTCOIN_SPEC.max_lev,
        spread_bps=BYBIT_GENERIC_ALTCOIN_SPEC.spread_bps,
        funding_interval_hours=BYBIT_GENERIC_ALTCOIN_SPEC.funding_interval_hours,
    )


class BybitContractSpecLoader:
    """Non-blocking live Bybit V5 linear contract spec loader with 6h TTL cache & static fallbacks."""

    def __init__(
        self,
        ttl_seconds: int = BYBIT_SPEC_CACHE_TTL_SECONDS,
        timeout_seconds: float = BYBIT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._cache: dict[str, BybitContractSpec] = {}
        self._last_loaded_monotonic: float = 0.0
        self._refresh_in_progress: bool = False

    def clear_cache(self) -> None:
        """Reset cached specs (useful for unit tests)."""
        with self._lock:
            self._cache.clear()
            self._last_loaded_monotonic = 0.0
            self._refresh_in_progress = False

    def is_cache_fresh(self) -> bool:
        with self._lock:
            if not self._cache or self._last_loaded_monotonic <= 0.0:
                return False
            return (time.monotonic() - self._last_loaded_monotonic) < self._ttl_seconds

    @staticmethod
    def parse_instruments_payload(payload: dict[str, Any]) -> dict[str, BybitContractSpec]:
        """Parse Bybit V5 /v5/market/instruments-info?category=linear JSON into exact Decimal specs."""
        parsed: dict[str, BybitContractSpec] = {}
        result = payload.get("result") or {}
        items = result.get("list") or []
        for item in items:
            sym = str(item.get("symbol") or "").strip().upper()
            if not sym:
                continue
            fallback = _static_bybit_spec(sym)
            lot_filter = item.get("lotSizeFilter") or {}
            price_filter = item.get("priceFilter") or {}
            lev_filter = item.get("leverageFilter") or {}

            min_qty = to_decimal(lot_filter.get("minOrderQty"), fallback.min_qty)
            qty_step = to_decimal(lot_filter.get("qtyStep"), fallback.qty_step)
            min_notional = to_decimal(lot_filter.get("minNotionalValue"), fallback.min_notional)
            tick_size = to_decimal(price_filter.get("tickSize"), fallback.tick_size)
            try:
                max_lev = int(to_decimal(lev_filter.get("maxLeverage"), Decimal(fallback.max_lev)))
            except Exception:
                max_lev = fallback.max_lev

            raw_funding = item.get("fundingInterval")
            funding_hours = fallback.funding_interval_hours
            if raw_funding is not None:
                try:
                    raw_int = int(raw_funding)
                    funding_hours = max(1, raw_int // 60) if raw_int >= 60 else max(1, raw_int)
                except Exception:
                    funding_hours = fallback.funding_interval_hours

            parsed[sym] = BybitContractSpec(
                symbol=sym,
                min_qty=min_qty if min_qty > ZERO else fallback.min_qty,
                qty_step=qty_step if qty_step > ZERO else fallback.qty_step,
                tick_size=tick_size if tick_size > ZERO else fallback.tick_size,
                min_notional=min_notional if min_notional > ZERO else fallback.min_notional,
                mmr=fallback.mmr,
                mm_deduction=fallback.mm_deduction,
                max_lev=max(1, max_lev),
                spread_bps=fallback.spread_bps,
                funding_interval_hours=funding_hours,
            )
        return parsed

    def fetch_live_specs(self, client: Optional[httpx.Client] = None) -> dict[str, BybitContractSpec]:
        """Synchronously fetch Bybit linear instruments with 5s timeout and update cache."""
        owns_client = client is None
        http = client or httpx.Client(timeout=self._timeout_seconds)
        try:
            resp = http.get(BYBIT_INSTRUMENTS_URL, timeout=self._timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            specs = self.parse_instruments_payload(data)
            if specs:
                with self._lock:
                    self._cache.update(specs)
                    self._last_loaded_monotonic = time.monotonic()
            return specs
        except Exception as exc:
            logger.debug("Bybit live contract spec fetch skipped (using fallback): %s", exc)
            return {}
        finally:
            if owns_client:
                http.close()

    def trigger_nonblocking_refresh(self) -> None:
        """Spawn a daemon thread to refresh live Bybit specs without blocking callers."""
        with self._lock:
            if self._refresh_in_progress:
                return
            if self._cache and (time.monotonic() - self._last_loaded_monotonic) < self._ttl_seconds:
                return
            self._refresh_in_progress = True

        def _worker() -> None:
            try:
                self.fetch_live_specs()
            finally:
                with self._lock:
                    self._refresh_in_progress = False

        thread = threading.Thread(target=_worker, name="bybit-spec-refresh", daemon=True)
        thread.start()

    def get_spec(self, symbol: str, trigger_refresh: bool = False) -> BybitContractSpec:
        """Return cached live spec if available, otherwise verified static fallback."""
        sym = (symbol or "").strip().upper()
        if trigger_refresh and not self.is_cache_fresh():
            self.trigger_nonblocking_refresh()
        with self._lock:
            if sym in self._cache:
                return self._cache[sym]
        return _static_bybit_spec(sym)


bybit_spec_loader = BybitContractSpecLoader()


def get_bybit_contract_spec(
    symbol: str,
    trigger_refresh: bool = False,
) -> BybitContractSpec:
    """Return Bybit linear contract spec (cached live spec or verified static fallback)."""
    return bybit_spec_loader.get_spec(symbol, trigger_refresh=trigger_refresh)


def bybit_fee_rate(order_type: str) -> Decimal:
    """Return VIP 0 Bybit fee rate: Maker 0.020% for Limit/TP, Taker 0.055% for Market/SL, 0% extra on Liquidation."""
    cleaned = (order_type or "MARKET").strip().upper()
    if cleaned in {"LIMIT", "TAKE_PROFIT"}:
        return BYBIT_MAKER_FEE_RATE
    if cleaned == "LIQUIDATION":
        return ZERO
    return BYBIT_TAKER_FEE_RATE


def bybit_effective_min_qty(
    symbol: str,
    price: Decimal | int | float | str,
    spec: Optional[BybitContractSpec] = None,
    is_closing: bool = False,
) -> Decimal:
    """effective_min_qty = max(min_qty, ceil_to_step(min_notional / price, qty_step)).

    Closing orders bypass min_notional and only check min_qty.
    """
    resolved = spec or get_bybit_contract_spec(symbol)
    if is_closing:
        return resolved.min_qty
    px = to_decimal(price)
    if px <= ZERO:
        return resolved.min_qty
    notional_min_qty = ceil_to_step(resolved.min_notional / px, resolved.qty_step)
    return max(resolved.min_qty, notional_min_qty)


def bybit_simulate_fill(
    symbol: str,
    side: str,
    is_entry: bool,
    mid_price: Decimal | int | float | str,
    order_type: str,
    qty: Decimal | int | float | str,
    volatility_pct: Decimal | int | float | str = Decimal("0.0015"),
    spec: Optional[BybitContractSpec] = None,
) -> FillSimulation:
    """Simulate Bybit fill with half-spread + deterministic volatility-scaled slippage quantized to tick_size."""
    resolved = spec or get_bybit_contract_spec(symbol)
    norm_side = normalize_side(side)
    otype = (order_type or "MARKET").strip().upper()
    mid = to_decimal(mid_price)
    q = to_decimal(qty)
    vol = max(ZERO, to_decimal(volatility_pct))

    # Buying happens on LONG entry or SHORT exit; selling happens on SHORT entry or LONG exit.
    is_buy = (norm_side == "LONG" and is_entry) or (norm_side == "SHORT" and not is_entry)

    if otype in {"LIMIT", "TAKE_PROFIT"}:
        filled = quantize_price(mid, resolved.tick_size)
        spread_cost = ZERO
        slippage_cost = ZERO
    elif otype == "LIQUIDATION":
        filled = quantize_price(mid, resolved.tick_size)
        spread_cost = ZERO
        slippage_cost = ZERO
    else:
        half_spread = mid * (resolved.spread_bps / BPS_DIVISOR) / TWO
        vol_ratio = max(ONE, vol / Decimal("0.002"))
        if otype == "STOP_LOSS":
            raw_slip_bps = resolved.spread_bps * Decimal("1.5") * vol_ratio
            slip_bps = min(Decimal("8.0"), max(Decimal("1.5"), raw_slip_bps))
        else:
            raw_slip_bps = resolved.spread_bps * Decimal("0.5") * vol_ratio
            slip_bps = min(Decimal("5.0"), max(Decimal("0.5"), raw_slip_bps))
        slippage = mid * (slip_bps / BPS_DIVISOR)

        raw_fill = (mid + half_spread + slippage) if is_buy else max(resolved.tick_size, mid - half_spread - slippage)
        filled = quantize_price(raw_fill, resolved.tick_size)
        spread_cost = q6(half_spread * q)
        slippage_cost = q6(slippage * q)

    rate = bybit_fee_rate(otype)
    fee = q6(filled * q * rate)
    return FillSimulation(
        requested_price=q8(mid),
        filled_price=q8(filled),
        fee=fee,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        order_type=otype,
    )


def bybit_calculate_margin_and_liquidation(
    side: str,
    entry_price: Decimal | int | float | str,
    qty: Decimal | int | float | str,
    leverage: int | Decimal,
    mark_price: Optional[Decimal | int | float | str] = None,
    extra_margin: Decimal | int | float | str = ZERO,
    mmr: Optional[Decimal | int | float | str] = None,
    mm_deduction: Optional[Decimal | int | float | str] = None,
    taker_rate: Decimal | int | float | str = BYBIT_TAKER_FEE_RATE,
    entry_order_type: str = "MARKET",
    symbol: Optional[str] = None,
    spec: Optional[BybitContractSpec] = None,
) -> BybitMarginMetrics:
    """Exact Bybit UTA Isolated Margin, Maintenance Margin, Liquidation Price, and Bankruptcy Price."""
    norm_side = normalize_side(side)
    entry = to_decimal(entry_price)
    q = to_decimal(qty)
    lev = to_decimal(leverage)
    if lev <= ZERO:
        raise ValueError("leverage must be positive")
    if q <= ZERO or entry <= ZERO:
        raise ValueError("entry_price and qty must be positive")

    resolved_spec = spec or (get_bybit_contract_spec(symbol) if symbol else None)
    eff_mmr = to_decimal(mmr) if mmr is not None else (resolved_spec.mmr if resolved_spec else Decimal("0.005"))
    eff_deduction = (
        to_decimal(mm_deduction)
        if mm_deduction is not None
        else (resolved_spec.mm_deduction if resolved_spec else ZERO)
    )
    taker = to_decimal(taker_rate)
    extra = to_decimal(extra_margin)
    mark = to_decimal(mark_price) if mark_price is not None else entry

    notional = entry * q
    inv_lev = ONE / lev

    if norm_side == "LONG":
        fee_to_close = notional * (ONE - inv_lev) * taker
        lp = (
            notional
            - (notional / lev)
            - (extra / (ONE - taker))
            - eff_deduction
        ) / (q * (ONE - eff_mmr))
        bp = max(ZERO, entry * (ONE - inv_lev) - (extra / q))
    else:
        fee_to_close = notional * (ONE + inv_lev) * taker
        lp = (
            notional
            + (notional / lev)
            + (extra / (ONE + taker))
            + eff_deduction
        ) / (q * (ONE + eff_mmr))
        bp = entry * (ONE + inv_lev) + (extra / q)

    initial_margin = (notional / lev) + fee_to_close
    entry_fee = notional * bybit_fee_rate(entry_order_type)
    required_cash = initial_margin + entry_fee
    maintenance_margin = max(ZERO, (q * mark * eff_mmr) - eff_deduction) + fee_to_close

    return BybitMarginMetrics(
        notional=q6(notional),
        fee_to_close=q6(fee_to_close),
        initial_margin=q6(initial_margin),
        entry_fee=q6(entry_fee),
        required_cash=q6(required_cash),
        maintenance_margin=q6(maintenance_margin),
        liquidation_price=q8(max(ZERO, lp)),
        bankruptcy_price=q8(max(ZERO, bp)),
    )


def bybit_should_liquidate(
    side: str,
    mark_price: Decimal | int | float | str,
    liquidation_price: Optional[Decimal | int | float | str],
) -> bool:
    """Return True if mark_price has reached or breached the UTA isolated liquidation price."""
    if liquidation_price is None:
        return False
    norm_side = normalize_side(side)
    mark = to_decimal(mark_price)
    lp = to_decimal(liquidation_price)
    if lp <= ZERO:
        return False
    if norm_side == "LONG":
        return mark <= lp
    return mark >= lp


def bybit_calculate_funding(
    side: str,
    qty: Decimal | int | float | str,
    mark_price: Decimal | int | float | str,
    opened_at: datetime,
    now_utc: datetime,
    last_funding_applied_at: Optional[datetime] = None,
    funding_rate: Decimal | int | float | str = BYBIT_DEFAULT_FUNDING_RATE,
) -> FundingResult:
    """Idempotently compute Bybit 8h funding (00:00, 08:00, 16:00 UTC).

    Positive fee_amount means the trader pays funding; negative means the trader receives funding.
    """
    norm_side = normalize_side(side)
    q = to_decimal(qty)
    mark = to_decimal(mark_price)
    rate = to_decimal(funding_rate)

    now_dt = ensure_utc(now_utc)
    open_dt = ensure_utc(opened_at)
    anchor = ensure_utc(last_funding_applied_at) if last_funding_applied_at is not None else open_dt
    if anchor < open_dt:
        anchor = open_dt

    if now_dt <= anchor:
        return FundingResult(
            fee_amount=ZERO,
            periods_applied=0,
            new_last_funding_applied_at=last_funding_applied_at,
        )

    # Start scanning from the next 8h boundary strictly after anchor
    start_day = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    candidate_times: list[datetime] = []
    cursor_day = start_day
    end_day = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor_day <= end_day:
        for h in BYBIT_FUNDING_HOURS_UTC:
            ts = cursor_day.replace(hour=h)
            if anchor < ts <= now_dt:
                candidate_times.append(ts)
        cursor_day += timedelta(days=1)

    if not candidate_times:
        return FundingResult(
            fee_amount=ZERO,
            periods_applied=0,
            new_last_funding_applied_at=last_funding_applied_at,
        )

    direction = ONE if norm_side == "LONG" else -ONE
    per_period_fee = q * mark * rate * direction
    total_fee = q6(per_period_fee * Decimal(len(candidate_times)))
    return FundingResult(
        fee_amount=total_fee,
        periods_applied=len(candidate_times),
        new_last_funding_applied_at=candidate_times[-1],
    )


def bybit_size_and_validate_order(
    symbol: str,
    side: str,
    entry_price: Decimal | int | float | str,
    stop_loss: Decimal | int | float | str,
    active_balance: Decimal | int | float | str,
    used_margin: Decimal | int | float | str,
    max_risk_per_trade_pct: Decimal | int | float | str,
    bybit_leverage: int = 5,
    order_entry_type: str = "MARKET",
    allow_min_lot_override: bool = False,
    volatility_pct: Decimal | int | float | str = Decimal("0.0015"),
    spec: Optional[BybitContractSpec] = None,
) -> OrderSizingResult:
    """Size a Bybit order using ONLY active_balance, factoring round-turn fees + spread into risk."""
    resolved = spec or get_bybit_contract_spec(symbol)
    norm_side = normalize_side(side)
    entry = to_decimal(entry_price)
    sl = to_decimal(stop_loss)
    balance = to_decimal(active_balance)
    used = to_decimal(used_margin)
    free_margin = q6(max(ZERO, balance - used))
    risk_pct = to_decimal(max_risk_per_trade_pct)
    eff_lev = max(1, min(int(bybit_leverage), resolved.max_lev))

    # Simulate entry and SL exit prices per unit to capture spread + slippage + round-turn fees
    entry_sim = bybit_simulate_fill(
        symbol=symbol,
        side=norm_side,
        is_entry=True,
        mid_price=entry,
        order_type=order_entry_type,
        qty=ONE,
        volatility_pct=volatility_pct,
        spec=resolved,
    )
    sl_sim = bybit_simulate_fill(
        symbol=symbol,
        side=norm_side,
        is_entry=False,
        mid_price=sl,
        order_type="STOP_LOSS",
        qty=ONE,
        volatility_pct=volatility_pct,
        spec=resolved,
    )

    fill_entry = entry_sim.filled_price
    fill_sl = sl_sim.filled_price
    stop_distance = abs(fill_entry - fill_sl)
    round_turn_fee_per_unit = (
        fill_entry * bybit_fee_rate(order_entry_type)
        + fill_sl * BYBIT_TAKER_FEE_RATE
    )
    risk_per_unit = stop_distance + round_turn_fee_per_unit

    eff_min_qty = bybit_effective_min_qty(symbol, fill_entry, spec=resolved)
    min_metrics = bybit_calculate_margin_and_liquidation(
        side=norm_side,
        entry_price=fill_entry,
        qty=eff_min_qty,
        leverage=eff_lev,
        entry_order_type=order_entry_type,
        spec=resolved,
    )

    if balance <= ZERO or risk_per_unit <= ZERO:
        return OrderSizingResult(
            approved=False,
            qty=ZERO,
            lots=None,
            effective_min_qty=eff_min_qty,
            required_margin=min_metrics.required_cash,
            available_free_margin=free_margin,
            estimated_risk_usd=ZERO,
            reason_code="INSUFFICIENT_MARGIN",
            details="Active balance or stop distance is non-positive.",
        )

    risk_budget = balance * (risk_pct / HUNDRED)
    raw_qty = risk_budget / risk_per_unit
    sized_qty = floor_to_step(raw_qty, resolved.qty_step)

    if sized_qty < eff_min_qty:
        if not allow_min_lot_override:
            # Determine most specific rejection code
            if min_metrics.required_cash > free_margin:
                reason_code = "INSUFFICIENT_MARGIN"
                details = (
                    f"{symbol} min order qty {eff_min_qty} (notional ${min_metrics.notional:.2f}) "
                    f"requires ${min_metrics.required_cash:.2f} margin at {eff_lev}x, "
                    f"exceeding free margin ${free_margin:.2f}."
                )
            elif (sized_qty * fill_entry) < resolved.min_notional:
                reason_code = "BELOW_MIN_NOTIONAL"
                details = (
                    f"{symbol} sized qty {sized_qty} (${sized_qty * fill_entry:.2f}) is below "
                    f"min notional ${resolved.min_notional:.2f} (effective_min_qty={eff_min_qty})."
                )
            else:
                reason_code = "BELOW_MIN_QTY"
                details = (
                    f"{symbol} sized qty {sized_qty} is below min order qty {eff_min_qty}."
                )
            return OrderSizingResult(
                approved=False,
                qty=sized_qty,
                lots=None,
                effective_min_qty=eff_min_qty,
                required_margin=min_metrics.required_cash,
                available_free_margin=free_margin,
                estimated_risk_usd=q6(sized_qty * risk_per_unit),
                reason_code=reason_code,
                details=details,
            )

        # allow_min_lot_override is True: still enforce margin and risk cap!
        if min_metrics.required_cash > free_margin:
            return OrderSizingResult(
                approved=False,
                qty=eff_min_qty,
                lots=None,
                effective_min_qty=eff_min_qty,
                required_margin=min_metrics.required_cash,
                available_free_margin=free_margin,
                estimated_risk_usd=q6(eff_min_qty * risk_per_unit),
                reason_code="INSUFFICIENT_MARGIN",
                details=(
                    f"Min-lot override blocked: {symbol} qty {eff_min_qty} requires "
                    f"${min_metrics.required_cash:.2f} > free margin ${free_margin:.2f}."
                ),
            )
        override_risk = q6(eff_min_qty * risk_per_unit)
        max_override_risk = q6(risk_budget * TWO)
        if override_risk > max_override_risk:
            return OrderSizingResult(
                approved=False,
                qty=eff_min_qty,
                lots=None,
                effective_min_qty=eff_min_qty,
                required_margin=min_metrics.required_cash,
                available_free_margin=free_margin,
                estimated_risk_usd=override_risk,
                reason_code="RISK_CAP_EXCEEDED",
                details=(
                    f"Min-lot override blocked: {symbol} risk ${override_risk:.2f} exceeds "
                    f"max cap ${max_override_risk:.2f}."
                ),
            )
        sized_qty = eff_min_qty

    # Check margin for sized_qty; scale down to fit free_margin if possible
    metrics = bybit_calculate_margin_and_liquidation(
        side=norm_side,
        entry_price=fill_entry,
        qty=sized_qty,
        leverage=eff_lev,
        entry_order_type=order_entry_type,
        spec=resolved,
    )
    if metrics.required_cash > free_margin:
        # Attempt to scale down qty to fit available free_margin
        per_unit_cash = metrics.required_cash / sized_qty
        max_affordable_qty = floor_to_step(free_margin / per_unit_cash, resolved.qty_step)
        if max_affordable_qty < eff_min_qty:
            return OrderSizingResult(
                approved=False,
                qty=sized_qty,
                lots=None,
                effective_min_qty=eff_min_qty,
                required_margin=min_metrics.required_cash,
                available_free_margin=free_margin,
                estimated_risk_usd=q6(sized_qty * risk_per_unit),
                reason_code="INSUFFICIENT_MARGIN",
                details=(
                    f"{symbol} requires ${metrics.required_cash:.2f} margin "
                    f"(min qty requires ${min_metrics.required_cash:.2f}) > free margin ${free_margin:.2f}."
                ),
            )
        sized_qty = max_affordable_qty
        metrics = bybit_calculate_margin_and_liquidation(
            side=norm_side,
            entry_price=fill_entry,
            qty=sized_qty,
            leverage=eff_lev,
            entry_order_type=order_entry_type,
            spec=resolved,
        )

    return OrderSizingResult(
        approved=True,
        qty=sized_qty,
        lots=None,
        effective_min_qty=eff_min_qty,
        required_margin=metrics.required_cash,
        available_free_margin=free_margin,
        estimated_risk_usd=q6(sized_qty * risk_per_unit),
        reason_code=None,
        details="Approved",
    )


# ---------------------------------------------------------------------------
# B. Forex / MT5 ECN & STP (EUR_USD, GBP_USD, XAU_USD)
# ---------------------------------------------------------------------------

FOREX_STATIC_SPECS: dict[str, ForexContractSpec] = {
    "EUR_USD": ForexContractSpec(
        symbol="EUR_USD",
        contract_size=Decimal("100000"),
        pip=Decimal("0.0001"),
        tick_size=Decimal("0.00001"),
        min_lot=Decimal("0.01"),
        lot_step=Decimal("0.01"),
        raw_spread_pips=Decimal("0.2"),
        stp_spread_pips=Decimal("1.2"),
        mkt_slip_pips=Decimal("0.2"),
        sl_slip_pips=Decimal("0.3"),
        swap_long=Decimal("-6.50"),
        swap_short=Decimal("1.20"),
    ),
    "GBP_USD": ForexContractSpec(
        symbol="GBP_USD",
        contract_size=Decimal("100000"),
        pip=Decimal("0.0001"),
        tick_size=Decimal("0.00001"),
        min_lot=Decimal("0.01"),
        lot_step=Decimal("0.01"),
        raw_spread_pips=Decimal("0.4"),
        stp_spread_pips=Decimal("1.5"),
        mkt_slip_pips=Decimal("0.2"),
        sl_slip_pips=Decimal("0.4"),
        swap_long=Decimal("-5.80"),
        swap_short=Decimal("0.80"),
    ),
    "XAU_USD": ForexContractSpec(
        symbol="XAU_USD",
        contract_size=Decimal("100"),
        pip=Decimal("0.10"),
        tick_size=Decimal("0.01"),
        min_lot=Decimal("0.01"),
        lot_step=Decimal("0.01"),
        raw_spread_pips=Decimal("1.8"),
        stp_spread_pips=Decimal("3.0"),
        mkt_slip_pips=Decimal("0.4"),
        sl_slip_pips=Decimal("0.8"),
        swap_long=Decimal("-28.00"),
        swap_short=Decimal("12.00"),
    ),
}


def get_forex_contract_spec(symbol: str) -> ForexContractSpec:
    """Return Forex contract specification for EUR_USD, GBP_USD, XAU_USD (or EUR_USD fallback)."""
    sym = (symbol or "").strip().upper()
    if sym in FOREX_STATIC_SPECS:
        return FOREX_STATIC_SPECS[sym]
    base = FOREX_STATIC_SPECS["EUR_USD"]
    return ForexContractSpec(
        symbol=sym or "EUR_USD",
        contract_size=base.contract_size,
        pip=base.pip,
        tick_size=base.tick_size,
        min_lot=base.min_lot,
        lot_step=base.lot_step,
        raw_spread_pips=base.raw_spread_pips,
        stp_spread_pips=base.stp_spread_pips,
        mkt_slip_pips=base.mkt_slip_pips,
        sl_slip_pips=base.sl_slip_pips,
        swap_long=base.swap_long,
        swap_short=base.swap_short,
    )


def forex_units_to_lots(symbol: str, units: Decimal | int | float | str) -> Decimal:
    """Convert raw Forex units (e.g. 1000 EUR_USD or 1 oz XAU_USD) to standard lots."""
    spec = get_forex_contract_spec(symbol)
    return q6(to_decimal(units) / spec.contract_size)


def forex_lots_to_units(symbol: str, lots: Decimal | int | float | str) -> Decimal:
    """Convert standard lots (e.g. 0.01) to raw units (lots * contract_size)."""
    spec = get_forex_contract_spec(symbol)
    return q6(to_decimal(lots) * spec.contract_size)


def forex_commission_per_side(
    lots: Decimal | int | float | str,
    account_profile: str = "mt5_raw_ecn",
) -> Decimal:
    """MT5 Raw ECN = $3.50/lot/side ($7.00 round-turn; $0.07 per 0.01 lot); Standard STP = $0.00."""
    profile = (account_profile or "mt5_raw_ecn").strip().lower()
    if profile == "mt5_standard_stp":
        return ZERO
    return q6(to_decimal(lots) * FOREX_RAW_ECN_COMMISSION_PER_LOT_SIDE)


def forex_round_turn_commission(
    lots: Decimal | int | float | str,
    account_profile: str = "mt5_raw_ecn",
) -> Decimal:
    """Total entry + exit Forex commission for the given lot size."""
    return q6(forex_commission_per_side(lots, account_profile) * TWO)


def forex_gross_pnl(
    symbol: str,
    side: str,
    lots: Decimal | int | float | str,
    entry_price: Decimal | int | float | str,
    exit_price: Decimal | int | float | str,
) -> Decimal:
    """Exact USD gross PnL for USD-quoted Forex pairs & XAU_USD: (exit - entry) * lots * contract_size * dir."""
    spec = get_forex_contract_spec(symbol)
    norm_side = normalize_side(side)
    direction = ONE if norm_side == "LONG" else -ONE
    units = to_decimal(lots) * spec.contract_size
    return q6((to_decimal(exit_price) - to_decimal(entry_price)) * units * direction)


def forex_simulate_fill(
    symbol: str,
    side: str,
    is_entry: bool,
    mid_price: Decimal | int | float | str,
    order_type: str,
    lots: Decimal | int | float | str,
    account_profile: str = "mt5_raw_ecn",
    volatility_pct: Decimal | int | float | str = Decimal("0.0005"),
) -> FillSimulation:
    """Simulate Forex MT5 ECN/STP fill with profile-specific spread + deterministic slippage."""
    spec = get_forex_contract_spec(symbol)
    norm_side = normalize_side(side)
    otype = (order_type or "MARKET").strip().upper()
    profile = (account_profile or "mt5_raw_ecn").strip().lower()
    mid = to_decimal(mid_price)
    lot_qty = to_decimal(lots)
    units = lot_qty * spec.contract_size
    vol = max(ZERO, to_decimal(volatility_pct))

    is_buy = (norm_side == "LONG" and is_entry) or (norm_side == "SHORT" and not is_entry)
    spread_pips = spec.stp_spread_pips if profile == "mt5_standard_stp" else spec.raw_spread_pips

    if otype in {"LIMIT", "TAKE_PROFIT"}:
        filled = quantize_price(mid, spec.tick_size)
        spread_cost = ZERO
        slippage_cost = ZERO
    else:
        half_spread = (spread_pips * spec.pip) / TWO
        slip_pips = spec.sl_slip_pips if otype in {"STOP_LOSS", "STOP_OUT"} else spec.mkt_slip_pips
        vol_scale = min(Decimal("3"), max(ONE, vol / Decimal("0.001")))
        slippage = slip_pips * spec.pip * vol_scale

        raw_fill = (mid + half_spread + slippage) if is_buy else max(spec.tick_size, mid - half_spread - slippage)
        filled = quantize_price(raw_fill, spec.tick_size)
        spread_cost = q6(half_spread * units)
        slippage_cost = q6(slippage * units)

    commission = forex_commission_per_side(lot_qty, profile)
    return FillSimulation(
        requested_price=q8(mid),
        filled_price=q8(filled),
        fee=commission,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        order_type=otype,
    )


def forex_required_margin(
    symbol: str,
    lots: Decimal | int | float | str,
    fill_price: Decimal | int | float | str,
    forex_leverage: int = 500,
) -> Decimal:
    """required_margin = (lots * contract_size * fill_price) / forex_leverage."""
    spec = get_forex_contract_spec(symbol)
    lev = to_decimal(forex_leverage)
    if lev <= ZERO:
        raise ValueError("forex_leverage must be positive")
    notional = to_decimal(lots) * spec.contract_size * to_decimal(fill_price)
    return q6(notional / lev)


def forex_margin_level_pct(
    forex_equity: Decimal | int | float | str,
    used_margin: Decimal | int | float | str,
) -> Optional[Decimal]:
    """margin_level_pct = (forex_equity / used_margin) * 100 (None if used_margin <= 0)."""
    used = to_decimal(used_margin)
    if used <= ZERO:
        return None
    eq = to_decimal(forex_equity)
    return q6((eq / used) * HUNDRED)


def select_stop_out_targets(
    open_positions: Sequence[ForexPositionSnapshot],
    equity: Decimal | int | float | str,
    stop_out_pct: Decimal | int | float | str = FOREX_STOP_OUT_LEVEL_PCT,
) -> list[ForexPositionSnapshot]:
    """Return open Forex positions ordered by largest unrealized loss first until margin_level_pct > stop_out_pct."""
    if not open_positions:
        return []
    eq = to_decimal(equity)
    threshold = to_decimal(stop_out_pct)
    total_used_margin = q6(sum((to_decimal(p.used_margin) for p in open_positions), ZERO))
    level = forex_margin_level_pct(eq, total_used_margin)
    if level is None or level > threshold:
        return []

    ordered = sorted(
        open_positions,
        key=lambda p: (to_decimal(p.unrealized_pnl), p.trade_id),
    )
    targets: list[ForexPositionSnapshot] = []
    remaining_margin = total_used_margin
    for pos in ordered:
        targets.append(pos)
        remaining_margin = max(ZERO, remaining_margin - to_decimal(pos.used_margin))
        if remaining_margin <= ZERO:
            break
        projected_level = forex_margin_level_pct(eq, remaining_margin)
        if projected_level is not None and projected_level > threshold:
            break
    return targets


def forex_evaluate_margin_status(
    forex_equity: Decimal | int | float | str,
    open_positions: Sequence[ForexPositionSnapshot],
    forex_leverage: int = 500,
) -> ForexMarginStatus:
    """Evaluate Forex margin level: Margin Call <= 100%, Stop-Out <= 50% (largest loser first)."""
    eq = to_decimal(forex_equity)
    total_used_margin = q6(sum((to_decimal(p.used_margin) for p in open_positions), ZERO))
    free_margin = q6(eq - total_used_margin)
    level = forex_margin_level_pct(eq, total_used_margin)

    if level is None:
        return ForexMarginStatus(
            forex_equity=q6(eq),
            used_margin=ZERO,
            free_margin=q6(eq),
            margin_level_pct=None,
            margin_call=False,
            stop_out=False,
            positions_to_close=(),
        )

    margin_call = level <= FOREX_MARGIN_CALL_LEVEL_PCT
    stop_out = level <= FOREX_STOP_OUT_LEVEL_PCT
    targets = select_stop_out_targets(open_positions, eq, FOREX_STOP_OUT_LEVEL_PCT) if stop_out else []
    to_close = [pos.trade_id for pos in targets]

    return ForexMarginStatus(
        forex_equity=q6(eq),
        used_margin=total_used_margin,
        free_margin=free_margin,
        margin_level_pct=level,
        margin_call=margin_call,
        stop_out=stop_out,
        positions_to_close=tuple(to_close),
    )


def forex_calculate_rollover_swap(
    symbol: str,
    side: str,
    lots: Decimal | int | float | str,
    opened_at: datetime,
    now_utc: datetime,
    last_swap_applied_date: Optional[date | str] = None,
) -> SwapResult:
    """Idempotently compute daily Forex rollover swap at 17:00 America/New_York (3x on Wednesdays).

    Positive fee_amount means the trader pays swap; negative means the trader receives swap.
    """
    spec = get_forex_contract_spec(symbol)
    norm_side = normalize_side(side)
    lot_qty = to_decimal(lots)

    parsed_last_date: Optional[date]
    if isinstance(last_swap_applied_date, str) and last_swap_applied_date.strip():
        parsed_last_date = date.fromisoformat(last_swap_applied_date.strip())
    elif isinstance(last_swap_applied_date, date):
        parsed_last_date = last_swap_applied_date
    else:
        parsed_last_date = None

    now_ny = ensure_utc(now_utc).astimezone(NY_TZ)
    open_ny = ensure_utc(opened_at).astimezone(NY_TZ)

    # Latest rollover date that has already occurred at or before now_ny
    latest_eligible_date = now_ny.date() if now_ny.hour >= 17 else (now_ny.date() - timedelta(days=1))

    # First rollover date strictly after position opened
    first_eligible_date = open_ny.date() if open_ny.hour < 17 else (open_ny.date() + timedelta(days=1))

    start_date = (
        max(first_eligible_date, parsed_last_date + timedelta(days=1))
        if parsed_last_date is not None
        else first_eligible_date
    )

    if start_date > latest_eligible_date:
        return SwapResult(
            fee_amount=ZERO,
            rollover_days_applied=0,
            total_multiplier=0,
            new_last_swap_applied_date=parsed_last_date,
        )

    days_applied = 0
    total_multiplier = 0
    last_applied: Optional[date] = parsed_last_date

    cursor = start_date
    while cursor <= latest_eligible_date:
        weekday = cursor.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
        if weekday < 5:
            mult = 3 if weekday == 2 else 1  # Triple swap on Wednesday for weekend rollover
            days_applied += 1
            total_multiplier += mult
        last_applied = cursor
        cursor += timedelta(days=1)

    if days_applied == 0:
        return SwapResult(
            fee_amount=ZERO,
            rollover_days_applied=0,
            total_multiplier=0,
            new_last_swap_applied_date=last_applied,
        )

    # swap_long / swap_short are negative for debits and positive for credits.
    # Negate to convert into fee_amount (positive = cost deducted from net_pnl).
    daily_swap_per_lot = spec.swap_long if norm_side == "LONG" else spec.swap_short
    fee_amount = q6(-daily_swap_per_lot * lot_qty * Decimal(total_multiplier))

    return SwapResult(
        fee_amount=fee_amount,
        rollover_days_applied=days_applied,
        total_multiplier=total_multiplier,
        new_last_swap_applied_date=last_applied,
    )


def forex_size_and_validate_order(
    symbol: str,
    side: str,
    entry_price: Decimal | int | float | str,
    stop_loss: Decimal | int | float | str,
    active_balance: Decimal | int | float | str,
    used_margin: Decimal | int | float | str,
    max_risk_per_trade_pct: Decimal | int | float | str,
    forex_leverage: int = 500,
    account_profile: str = "mt5_raw_ecn",
    order_entry_type: str = "MARKET",
    allow_min_lot_override: bool = False,
    volatility_pct: Decimal | int | float | str = Decimal("0.0005"),
) -> OrderSizingResult:
    """Size a Forex order using ONLY active_balance, factoring round-turn commission + spread into risk."""
    spec = get_forex_contract_spec(symbol)
    norm_side = normalize_side(side)
    entry = to_decimal(entry_price)
    sl = to_decimal(stop_loss)
    balance = to_decimal(active_balance)
    used = to_decimal(used_margin)
    free_margin = q6(max(ZERO, balance - used))
    risk_pct = to_decimal(max_risk_per_trade_pct)

    entry_sim = forex_simulate_fill(
        symbol=symbol,
        side=norm_side,
        is_entry=True,
        mid_price=entry,
        order_type=order_entry_type,
        lots=ONE,
        account_profile=account_profile,
        volatility_pct=volatility_pct,
    )
    sl_sim = forex_simulate_fill(
        symbol=symbol,
        side=norm_side,
        is_entry=False,
        mid_price=sl,
        order_type="STOP_LOSS",
        lots=ONE,
        account_profile=account_profile,
        volatility_pct=volatility_pct,
    )

    fill_entry = entry_sim.filled_price
    fill_sl = sl_sim.filled_price
    stop_distance = abs(fill_entry - fill_sl)
    rt_comm_per_lot = forex_round_turn_commission(ONE, account_profile)
    risk_per_lot = (stop_distance * spec.contract_size) + rt_comm_per_lot

    min_margin = forex_required_margin(symbol, spec.min_lot, fill_entry, forex_leverage)
    min_units = forex_lots_to_units(symbol, spec.min_lot)

    if balance <= ZERO or risk_per_lot <= ZERO:
        return OrderSizingResult(
            approved=False,
            qty=ZERO,
            lots=ZERO,
            effective_min_qty=min_units,
            required_margin=min_margin,
            available_free_margin=free_margin,
            estimated_risk_usd=ZERO,
            reason_code="INSUFFICIENT_MARGIN",
            details="Active balance or stop distance is non-positive.",
        )

    risk_budget = balance * (risk_pct / HUNDRED)
    raw_lots = risk_budget / risk_per_lot
    sized_lots = floor_to_step(raw_lots, spec.lot_step)

    if sized_lots < spec.min_lot:
        if not allow_min_lot_override:
            if min_margin > free_margin:
                return OrderSizingResult(
                    approved=False,
                    qty=forex_lots_to_units(symbol, sized_lots),
                    lots=sized_lots,
                    effective_min_qty=min_units,
                    required_margin=min_margin,
                    available_free_margin=free_margin,
                    estimated_risk_usd=q6(sized_lots * risk_per_lot),
                    reason_code="INSUFFICIENT_MARGIN",
                    details=(
                        f"{symbol} min lot {spec.min_lot} requires ${min_margin:.2f} margin "
                        f"> free margin ${free_margin:.2f}."
                    ),
                )
            return OrderSizingResult(
                approved=False,
                qty=forex_lots_to_units(symbol, sized_lots),
                lots=sized_lots,
                effective_min_qty=min_units,
                required_margin=min_margin,
                available_free_margin=free_margin,
                estimated_risk_usd=q6(sized_lots * risk_per_lot),
                reason_code="BELOW_MIN_QTY",
                details=(
                    f"{symbol} sized lots {sized_lots} is below min_lot {spec.min_lot} "
                    f"(risk budget ${risk_budget:.2f})."
                ),
            )

        if min_margin > free_margin:
            return OrderSizingResult(
                approved=False,
                qty=min_units,
                lots=spec.min_lot,
                effective_min_qty=min_units,
                required_margin=min_margin,
                available_free_margin=free_margin,
                estimated_risk_usd=q6(spec.min_lot * risk_per_lot),
                reason_code="INSUFFICIENT_MARGIN",
                details=(
                    f"Min-lot override blocked: {symbol} {spec.min_lot} lot requires "
                    f"${min_margin:.2f} > free margin ${free_margin:.2f}."
                ),
            )
        override_risk = q6(spec.min_lot * risk_per_lot)
        max_override_risk = q6(risk_budget * TWO)
        if override_risk > max_override_risk:
            return OrderSizingResult(
                approved=False,
                qty=min_units,
                lots=spec.min_lot,
                effective_min_qty=min_units,
                required_margin=min_margin,
                available_free_margin=free_margin,
                estimated_risk_usd=override_risk,
                reason_code="RISK_CAP_EXCEEDED",
                details=(
                    f"Min-lot override blocked: {symbol} risk ${override_risk:.2f} exceeds "
                    f"max cap ${max_override_risk:.2f}."
                ),
            )
        sized_lots = spec.min_lot

    req_margin = forex_required_margin(symbol, sized_lots, fill_entry, forex_leverage)
    if req_margin > free_margin:
        margin_per_lot = forex_required_margin(symbol, ONE, fill_entry, forex_leverage)
        max_lots = floor_to_step(free_margin / margin_per_lot, spec.lot_step) if margin_per_lot > ZERO else ZERO
        if max_lots < spec.min_lot:
            return OrderSizingResult(
                approved=False,
                qty=forex_lots_to_units(symbol, sized_lots),
                lots=sized_lots,
                effective_min_qty=min_units,
                required_margin=min_margin,
                available_free_margin=free_margin,
                estimated_risk_usd=q6(sized_lots * risk_per_lot),
                reason_code="INSUFFICIENT_MARGIN",
                details=(
                    f"{symbol} requires ${req_margin:.2f} margin > free margin ${free_margin:.2f}."
                ),
            )
        sized_lots = max_lots
        req_margin = forex_required_margin(symbol, sized_lots, fill_entry, forex_leverage)

    units = forex_lots_to_units(symbol, sized_lots)
    return OrderSizingResult(
        approved=True,
        qty=units,
        lots=sized_lots,
        effective_min_qty=min_units,
        required_margin=req_margin,
        available_free_margin=free_margin,
        estimated_risk_usd=q6(sized_lots * risk_per_lot),
        reason_code=None,
        details="Approved",
    )


# ---------------------------------------------------------------------------
# C. Round-Turn Friction Estimator (for Strategy Vetoes)
# ---------------------------------------------------------------------------


def estimate_round_turn_friction(
    market: str,
    symbol: str,
    side: str,
    qty: Decimal | int | float | str,
    entry_price: Decimal | int | float | str,
    sl_price: Decimal | int | float | str,
    tp_price: Decimal | int | float | str,
    leverage: int | Decimal = 5,
    profile: str = "bybit_vip0_linear",
    entry_order_type: str = "MARKET",
    tp_order_type: str = "TAKE_PROFIT",
    sl_order_type: str = "STOP_LOSS",
    volatility_pct: Optional[Decimal | int | float | str] = None,
) -> RoundTurnFrictionEstimate:
    """Estimate round-turn fees, spread, slippage, net TP/SL PnL, net R:R, and TP fee drag.

    For Bybit, `qty` is base asset quantity (e.g. 0.002 BTC or 0.1 SOL).
    For Forex, `qty` is standard lots (e.g. 0.01 lot).
    """
    mkt = (market or "").strip().lower()
    norm_side = normalize_side(side)
    direction = ONE if norm_side == "LONG" else -ONE
    q = to_decimal(qty)
    entry = to_decimal(entry_price)
    sl = to_decimal(sl_price)
    tp = to_decimal(tp_price)
    _lev = to_decimal(leverage)
    if _lev <= ZERO:
        raise ValueError("leverage must be positive")
    if q <= ZERO or entry <= ZERO:
        raise ValueError("qty and entry_price must be positive")

    if mkt == "bybit":
        vol = to_decimal(volatility_pct) if volatility_pct is not None else Decimal("0.0015")
        entry_fill = bybit_simulate_fill(
            symbol=symbol,
            side=norm_side,
            is_entry=True,
            mid_price=entry,
            order_type=entry_order_type,
            qty=q,
            volatility_pct=vol,
        )
        tp_fill = bybit_simulate_fill(
            symbol=symbol,
            side=norm_side,
            is_entry=False,
            mid_price=tp,
            order_type=tp_order_type,
            qty=q,
            volatility_pct=vol,
        )
        sl_fill = bybit_simulate_fill(
            symbol=symbol,
            side=norm_side,
            is_entry=False,
            mid_price=sl,
            order_type=sl_order_type,
            qty=q,
            volatility_pct=vol,
        )
        gross_tp_profit = q6(max(ZERO, (tp - entry) * q * direction))
        gross_sl_loss = q6(max(ZERO, (entry - sl) * q * direction))
    elif mkt == "forex":
        vol = to_decimal(volatility_pct) if volatility_pct is not None else Decimal("0.0005")
        resolved_profile = (profile or "mt5_raw_ecn").strip().lower()
        entry_fill = forex_simulate_fill(
            symbol=symbol,
            side=norm_side,
            is_entry=True,
            mid_price=entry,
            order_type=entry_order_type,
            lots=q,
            account_profile=resolved_profile,
            volatility_pct=vol,
        )
        tp_fill = forex_simulate_fill(
            symbol=symbol,
            side=norm_side,
            is_entry=False,
            mid_price=tp,
            order_type=tp_order_type,
            lots=q,
            account_profile=resolved_profile,
            volatility_pct=vol,
        )
        sl_fill = forex_simulate_fill(
            symbol=symbol,
            side=norm_side,
            is_entry=False,
            mid_price=sl,
            order_type=sl_order_type,
            lots=q,
            account_profile=resolved_profile,
            volatility_pct=vol,
        )
        gross_tp_profit = q6(max(ZERO, forex_gross_pnl(symbol, norm_side, q, entry, tp)))
        gross_sl_loss = q6(max(ZERO, -forex_gross_pnl(symbol, norm_side, q, entry, sl)))
    else:
        raise ValueError(f"Unsupported market '{market}'. Expected 'bybit' or 'forex'.")

    entry_fee = q6(entry_fill.fee)
    est_exit_fee_tp = q6(tp_fill.fee)
    est_exit_fee_sl = q6(sl_fill.fee)
    est_spread_cost = q6(entry_fill.spread_cost + tp_fill.spread_cost)
    est_slippage_cost = q6(entry_fill.slippage_cost + tp_fill.slippage_cost)

    total_friction_tp = q6(entry_fee + est_exit_fee_tp + est_spread_cost + est_slippage_cost)
    total_friction_sl = q6(
        entry_fee
        + est_exit_fee_sl
        + entry_fill.spread_cost
        + sl_fill.spread_cost
        + entry_fill.slippage_cost
        + sl_fill.slippage_cost
    )

    net_tp_profit = q6(gross_tp_profit - total_friction_tp)
    net_sl_loss = q6(gross_sl_loss + total_friction_sl)
    net_rr = q6(net_tp_profit / net_sl_loss) if net_sl_loss > ZERO else ZERO
    fee_drag_pct_of_tp = (
        q6((total_friction_tp / gross_tp_profit) * HUNDRED)
        if gross_tp_profit > ZERO
        else q6(HUNDRED)
    )

    return RoundTurnFrictionEstimate(
        entry_fee=entry_fee,
        est_exit_fee_tp=est_exit_fee_tp,
        est_exit_fee_sl=est_exit_fee_sl,
        est_spread_cost=est_spread_cost,
        est_slippage_cost=est_slippage_cost,
        total_friction_tp=total_friction_tp,
        gross_tp_profit=gross_tp_profit,
        net_tp_profit=net_tp_profit,
        gross_sl_loss=gross_sl_loss,
        net_sl_loss=net_sl_loss,
        net_rr=net_rr,
        fee_drag_pct_of_tp=fee_drag_pct_of_tp,
    )

