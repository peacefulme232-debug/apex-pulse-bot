"""Zero-repaint technical indicators, confirmed swing structure, and S/R zones.

NON-NEGOTIABLE ZERO-REPAINT GUARANTEES:
1. Every indicator operates strictly on CLOSED candles only. Any candle flagged
   as incomplete/forming (`is_closed=False`, `closed=False`, `complete=False`,
   or `is_forming=True`) is dropped before computation.
2. Confirmed Swing Highs & Swing Lows use strictly lagging pivots (left=3, right=3):
   a pivot at candle index `i` is invisible until candle index `i + right` has closed.
3. Every series computation is strictly causal (index `i` depends ONLY on `0..i`),
   guaranteeing 100% bit-for-bit identical values on `candles[:k]` vs `candles[:k+N]`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class Candle:
    """Immutable OHLCV candle with explicit closed-bar flag."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: Optional[Any] = None
    is_closed: bool = True


CandleLike = Union[Candle, Mapping[str, Any]]


@dataclass(frozen=True)
class Pivot:
    """Strictly lagging confirmed swing pivot (visible only at `confirmed_at_index`)."""

    index: int
    confirmed_at_index: int
    kind: Literal["HIGH", "LOW"]
    price: float


@dataclass(frozen=True)
class SRZone:
    """Clustered horizontal support/resistance zone built from confirmed pivots."""

    zone_low: float
    zone_high: float
    touches: int
    last_touch_index: int

    @property
    def midpoint(self) -> float:
        return (self.zone_low + self.zone_high) * 0.5


@dataclass(frozen=True)
class CandleMetrics:
    """Single- and two-candle price-action anatomy metrics."""

    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    is_bullish_rejection: bool
    is_bearish_rejection: bool
    is_bullish_engulfing: bool
    is_bearish_engulfing: bool


def _is_candle_closed(raw: Any) -> bool:
    """Return False if a candle explicitly signals it is still forming/incomplete."""
    if isinstance(raw, Candle):
        return bool(raw.is_closed)
    if isinstance(raw, Mapping):
        if "is_closed" in raw and raw["is_closed"] is not None:
            return bool(raw["is_closed"])
        if "closed" in raw and raw["closed"] is not None:
            return bool(raw["closed"])
        if "complete" in raw and raw["complete"] is not None:
            return bool(raw["complete"])
        if "is_forming" in raw and raw["is_forming"] is not None:
            return not bool(raw["is_forming"])
        if "forming" in raw and raw["forming"] is not None:
            return not bool(raw["forming"])
        return True
    # Generic object with attributes
    for attr in ("is_closed", "closed", "complete"):
        val = getattr(raw, attr, None)
        if val is not None:
            return bool(val)
    for attr in ("is_forming", "forming"):
        val = getattr(raw, attr, None)
        if val is not None:
            return not bool(val)
    return True


def _to_candle(raw: Any) -> Candle:
    """Convert a Candle, mapping, or attribute-bearing object into a `Candle`."""
    if isinstance(raw, Candle):
        return raw
    if isinstance(raw, Mapping):
        return Candle(
            open=float(raw["open"]),
            high=float(raw["high"]),
            low=float(raw["low"]),
            close=float(raw["close"]),
            volume=float(raw.get("volume", 0.0)),
            timestamp=raw.get("timestamp", raw.get("ts")),
            is_closed=_is_candle_closed(raw),
        )
    return Candle(
        open=float(getattr(raw, "open")),
        high=float(getattr(raw, "high")),
        low=float(getattr(raw, "low")),
        close=float(getattr(raw, "close")),
        volume=float(getattr(raw, "volume", 0.0)),
        timestamp=getattr(raw, "timestamp", getattr(raw, "ts", None)),
        is_closed=_is_candle_closed(raw),
    )


def filter_closed_candles(candles: Any) -> list[Candle]:
    """Return strictly closed `Candle` objects, dropping any forming/incomplete candle.

    Accepts a sequence of `Candle`, dicts/mappings, objects, or a pandas DataFrame.
    """
    if candles is None:
        return []

    # Support pandas DataFrame without requiring pandas as a hard runtime dependency.
    if hasattr(candles, "to_dict") and hasattr(candles, "columns"):
        rows = candles.to_dict(orient="records")
        return [_to_candle(row) for row in rows if _is_candle_closed(row)]

    closed: list[Candle] = []
    for item in candles:
        if _is_candle_closed(item):
            closed.append(_to_candle(item))
    return closed


def extract_ohlcv(
    candles: Any,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Filter strictly to closed candles and unpack `(opens, highs, lows, closes, volumes)`."""
    closed = filter_closed_candles(candles)
    opens = [c.open for c in closed]
    highs = [c.high for c in closed]
    lows = [c.low for c in closed]
    closes = [c.close for c in closed]
    volumes = [c.volume for c in closed]
    return opens, highs, lows, closes, volumes


# ---------------------------------------------------------------------------
# Pure Deterministic Causal Indicator Functions
# ---------------------------------------------------------------------------


def ema(closes: Sequence[float], period: int) -> list[Optional[float]]:
    """Compute SMA-seeded Exponential Moving Average matching TradingView `ta.ema`.

    Strictly causal: index `i` depends only on `closes[0..i]`.
    Returns `None` for warmup indices `0 .. period - 2`.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")

    n = len(closes)
    if n == 0:
        return []

    result: list[Optional[float]] = [None] * n
    if n < period:
        return result

    seed = 0.0
    for i in range(period):
        seed += float(closes[i])
    prev = seed / float(period)
    result[period - 1] = prev

    alpha = 2.0 / float(period + 1)
    one_minus_alpha = 1.0 - alpha

    for i in range(period, n):
        prev = alpha * float(closes[i]) + one_minus_alpha * prev
        result[i] = prev

    return result


def ema_slope(
    ema_values: Sequence[Optional[float]],
    lookback: int = 5,
) -> list[Optional[float]]:
    """Compute strictly causal per-bar EMA slope over `lookback` closed bars."""
    if lookback < 1:
        raise ValueError(f"EMA slope lookback must be >= 1, got {lookback}")

    n = len(ema_values)
    result: list[Optional[float]] = [None] * n
    inv_lookback = 1.0 / float(lookback)

    for i in range(lookback, n):
        cur = ema_values[i]
        prev = ema_values[i - lookback]
        if cur is not None and prev is not None:
            result[i] = (float(cur) - float(prev)) * inv_lookback

    return result


def rsi_wilder(closes: Sequence[float], period: int = 14) -> list[Optional[float]]:
    """Compute Wilder's Relative Strength Index matching TradingView `ta.rsi`.

    Strictly causal:
    - Warmup indices `0 .. period - 1` are `None`.
    - Index `period` uses the simple mean of the first `period` gains/losses.
    - Subsequent indices use Wilder's RMA: `(prev * (period - 1) + current) / period`.
    - Guards flat markets (`avg_gain == 0` and `avg_loss == 0` -> `50.0`) and
      pure uptrends (`avg_loss == 0` -> `100.0`).
    """
    if period < 1:
        raise ValueError(f"RSI period must be >= 1, got {period}")

    n = len(closes)
    if n == 0:
        return []

    result: list[Optional[float]] = [None] * n
    if n <= period:
        return result

    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, period + 1):
        delta = float(closes[i]) - float(closes[i - 1])
        if delta > 0.0:
            gain_sum += delta
        elif delta < 0.0:
            loss_sum += -delta

    avg_gain = gain_sum / float(period)
    avg_loss = loss_sum / float(period)

    def _calc_rsi(ag: float, al: float) -> float:
        if ag <= 1e-12 and al <= 1e-12:
            return 50.0
        if al <= 1e-12:
            return 100.0
        if ag <= 1e-12:
            return 0.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    result[period] = _calc_rsi(avg_gain, avg_loss)

    p_minus_1 = float(period - 1)
    p_float = float(period)
    for i in range(period + 1, n):
        delta = float(closes[i]) - float(closes[i - 1])
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        avg_gain = (avg_gain * p_minus_1 + gain) / p_float
        avg_loss = (avg_loss * p_minus_1 + loss) / p_float
        result[i] = _calc_rsi(avg_gain, avg_loss)

    return result


def atr_wilder(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[Optional[float]]:
    """Compute Wilder's Average True Range matching TradingView `ta.atr`.

    Strictly causal:
    - `TR[0] = high[0] - low[0]`
    - `TR[i] = max(high[i] - low[i], |high[i] - close[i-1]|, |low[i] - close[i-1]|)`
    - Warmup indices `0 .. period - 2` are `None`.
    - Index `period - 1` is the SMA of `TR[0..period-1]`.
    - Subsequent indices use Wilder's RMA: `(prev * (period - 1) + TR[i]) / period`.
    """
    if period < 1:
        raise ValueError(f"ATR period must be >= 1, got {period}")

    n = len(closes)
    if len(highs) != n or len(lows) != n:
        raise ValueError("highs, lows, and closes must have equal length")
    if n == 0:
        return []

    result: list[Optional[float]] = [None] * n
    if n < period:
        return result

    tr_sum = float(highs[0]) - float(lows[0])
    if tr_sum < 0.0:
        tr_sum = abs(tr_sum)

    for i in range(1, period):
        h = float(highs[i])
        l = float(lows[i])
        pc = float(closes[i - 1])
        tr = max(abs(h - l), abs(h - pc), abs(l - pc))
        tr_sum += tr

    prev_atr = tr_sum / float(period)
    result[period - 1] = prev_atr

    p_minus_1 = float(period - 1)
    p_float = float(period)
    for i in range(period, n):
        h = float(highs[i])
        l = float(lows[i])
        pc = float(closes[i - 1])
        tr = max(abs(h - l), abs(h - pc), abs(l - pc))
        prev_atr = (prev_atr * p_minus_1 + tr) / p_float
        result[i] = prev_atr

    return result


def atr_percentile(
    atr_values: Sequence[Optional[float]],
    window: int = 100,
) -> list[Optional[float]]:
    """Compute strictly causal rolling percentile rank (0..100) of ATR values.

    Uses mid-rank `(count_less + 0.5 * count_equal) / count * 100.0` over the
    trailing `window` valid ATR observations up to and including index `i`.
    """
    if window < 1:
        raise ValueError(f"ATR percentile window must be >= 1, got {window}")

    n = len(atr_values)
    result: list[Optional[float]] = [None] * n
    valid_history: list[float] = []

    for i in range(n):
        val = atr_values[i]
        if val is None or not math.isfinite(float(val)):
            continue
        fval = float(val)
        valid_history.append(fval)
        start = max(0, len(valid_history) - window)
        sample = valid_history[start:]
        count = len(sample)
        less = 0
        equal = 0
        for s in sample:
            if s < fval:
                less += 1
            elif s == fval:
                equal += 1
        result[i] = ((less + 0.5 * equal) / float(count)) * 100.0

    return result


def adx_wilder(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """Compute Wilder's ADX, +DI, and -DI matching TradingView `ta.dmi`.

    Strictly causal:
    - `+DI` and `-DI` are valid starting at index `period`.
    - `ADX` is valid starting at index `2 * period - 1` (SMA of first `period` DX values,
      then Wilder smoothed).
    - Zero-denominator guards prevent division by zero on flat candles.
    """
    if period < 1:
        raise ValueError(f"ADX period must be >= 1, got {period}")

    n = len(closes)
    if len(highs) != n or len(lows) != n:
        raise ValueError("highs, lows, and closes must have equal length")

    adx_out: list[Optional[float]] = [None] * n
    plus_di_out: list[Optional[float]] = [None] * n
    minus_di_out: list[Optional[float]] = [None] * n

    if n <= period:
        return adx_out, plus_di_out, minus_di_out

    tr_list: list[float] = [0.0] * n
    plus_dm_list: list[float] = [0.0] * n
    minus_dm_list: list[float] = [0.0] * n

    for i in range(1, n):
        h = float(highs[i])
        l = float(lows[i])
        ph = float(highs[i - 1])
        pl = float(lows[i - 1])
        pc = float(closes[i - 1])

        tr_list[i] = max(abs(h - l), abs(h - pc), abs(l - pc))

        up_move = h - ph
        down_move = pl - l
        if up_move > down_move and up_move > 0.0:
            plus_dm_list[i] = up_move
        if down_move > up_move and down_move > 0.0:
            minus_dm_list[i] = down_move

    smooth_tr = 0.0
    smooth_plus = 0.0
    smooth_minus = 0.0
    for i in range(1, period + 1):
        smooth_tr += tr_list[i]
        smooth_plus += plus_dm_list[i]
        smooth_minus += minus_dm_list[i]

    p_float = float(period)
    p_minus_1 = float(period - 1)
    dx_values: list[float] = []
    prev_adx: Optional[float] = None

    for i in range(period, n):
        if i > period:
            smooth_tr = smooth_tr - (smooth_tr / p_float) + tr_list[i]
            smooth_plus = smooth_plus - (smooth_plus / p_float) + plus_dm_list[i]
            smooth_minus = smooth_minus - (smooth_minus / p_float) + minus_dm_list[i]

        if smooth_tr > 1e-12:
            pdi = 100.0 * smooth_plus / smooth_tr
            mdi = 100.0 * smooth_minus / smooth_tr
        else:
            pdi = 0.0
            mdi = 0.0

        plus_di_out[i] = pdi
        minus_di_out[i] = mdi

        di_sum = pdi + mdi
        dx = (100.0 * abs(pdi - mdi) / di_sum) if di_sum > 1e-12 else 0.0
        dx_values.append(dx)

        if len(dx_values) == period:
            prev_adx = sum(dx_values) / p_float
            adx_out[i] = prev_adx
        elif len(dx_values) > period and prev_adx is not None:
            prev_adx = (prev_adx * p_minus_1 + dx) / p_float
            adx_out[i] = prev_adx

    return adx_out, plus_di_out, minus_di_out


# ---------------------------------------------------------------------------
# Strictly Lagging Confirmed Pivots & S/R Zone Clustering
# ---------------------------------------------------------------------------


def confirmed_pivots(
    highs: Sequence[float],
    lows: Sequence[float],
    left: int = 3,
    right: int = 3,
) -> list[Pivot]:
    """Detect strictly lagging confirmed swing highs and swing lows.

    A pivot at index `i` requires `left` prior bars (`i-left .. i-1`) AND `right`
    subsequent closed bars (`i+1 .. i+right`). Therefore, a pivot at `i` is ONLY
    visible once candle `i + right` has closed (`confirmed_at_index == i + right`).
    """
    if left < 1 or right < 1:
        raise ValueError(f"left and right must be >= 1, got left={left}, right={right}")

    n = len(highs)
    if len(lows) != n:
        raise ValueError("highs and lows must have equal length")

    if n < left + right + 1:
        return []

    pivots: list[Pivot] = []
    # Only indices `i` with `i + right < n` are confirmed on the closed series.
    for i in range(left, n - right):
        hi = float(highs[i])
        lo = float(lows[i])

        is_swing_high = True
        for j in range(i - left, i):
            if hi <= float(highs[j]):
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(i + 1, i + right + 1):
                if hi < float(highs[j]):
                    is_swing_high = False
                    break

        if is_swing_high:
            pivots.append(
                Pivot(
                    index=i,
                    confirmed_at_index=i + right,
                    kind="HIGH",
                    price=hi,
                )
            )

        is_swing_low = True
        for j in range(i - left, i):
            if lo >= float(lows[j]):
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(i + 1, i + right + 1):
                if lo > float(lows[j]):
                    is_swing_low = False
                    break

        if is_swing_low:
            pivots.append(
                Pivot(
                    index=i,
                    confirmed_at_index=i + right,
                    kind="LOW",
                    price=lo,
                )
            )

    return pivots


def key_sr_zones(
    pivots: Sequence[Pivot],
    atr: Optional[float],
    merge_tolerance_atr: float = 0.5,
    min_touches: int = 2,
) -> list[SRZone]:
    """Cluster confirmed pivots into horizontal Support/Resistance zones.

    Pivots are processed in chronological confirmation order; a pivot merges into
    the nearest cluster whose center is within `merge_tolerance_atr * atr`.
    """
    if not pivots or atr is None or not math.isfinite(float(atr)) or float(atr) <= 0.0:
        return []
    if merge_tolerance_atr < 0.0:
        raise ValueError(f"merge_tolerance_atr must be >= 0, got {merge_tolerance_atr}")
    if min_touches < 1:
        raise ValueError(f"min_touches must be >= 1, got {min_touches}")

    tolerance = float(merge_tolerance_atr) * float(atr)
    ordered_pivots = sorted(pivots, key=lambda p: (p.confirmed_at_index, p.index, p.kind))

    # Each cluster: [prices_list, last_touch_index]
    clusters: list[tuple[list[float], int]] = []

    for pivot in ordered_pivots:
        best_idx: Optional[int] = None
        best_dist = float("inf")
        for idx, (prices, _) in enumerate(clusters):
            center = sum(prices) / float(len(prices))
            dist = abs(pivot.price - center)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            clusters.append(([pivot.price], pivot.index))
        else:
            prices, last_touch = clusters[best_idx]
            prices.append(pivot.price)
            clusters[best_idx] = (prices, max(last_touch, pivot.index))

    zones: list[SRZone] = []
    for prices, last_touch in clusters:
        if len(prices) >= min_touches:
            zones.append(
                SRZone(
                    zone_low=min(prices),
                    zone_high=max(prices),
                    touches=len(prices),
                    last_touch_index=last_touch,
                )
            )

    zones.sort(key=lambda z: (z.zone_low, z.zone_high))
    return zones


# ---------------------------------------------------------------------------
# Candle Price-Action Metrics
# ---------------------------------------------------------------------------


def candle_metrics(
    candle: CandleLike,
    prev_candle: Optional[CandleLike] = None,
) -> CandleMetrics:
    """Compute wick/body ratios and rejection/engulfing price-action flags."""
    cur = _to_candle(candle)
    candle_range = cur.high - cur.low

    if candle_range <= 1e-12:
        return CandleMetrics(
            body_ratio=0.0,
            upper_wick_ratio=0.0,
            lower_wick_ratio=0.0,
            is_bullish_rejection=False,
            is_bearish_rejection=False,
            is_bullish_engulfing=False,
            is_bearish_engulfing=False,
        )

    body = abs(cur.close - cur.open)
    upper_wick = max(0.0, cur.high - max(cur.open, cur.close))
    lower_wick = max(0.0, min(cur.open, cur.close) - cur.low)

    body_ratio = body / candle_range
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range
    close_pos = (cur.close - cur.low) / candle_range

    is_bullish_rejection = (
        lower_wick > 0.0
        and lower_wick >= 1.5 * body
        and close_pos >= 0.60
        and (cur.close >= cur.open or close_pos >= 0.65)
    )

    is_bearish_rejection = (
        upper_wick > 0.0
        and upper_wick >= 1.5 * body
        and close_pos <= 0.40
        and (cur.close <= cur.open or close_pos <= 0.35)
    )

    is_bullish_engulfing = False
    is_bearish_engulfing = False
    if prev_candle is not None:
        prev = _to_candle(prev_candle)
        prev_body = abs(prev.close - prev.open)
        if body > 0.0 and prev_body > 0.0:
            if (
                prev.close < prev.open
                and cur.close > cur.open
                and cur.open <= prev.close
                and cur.close >= prev.open
            ):
                is_bullish_engulfing = True
            elif (
                prev.close > prev.open
                and cur.close < cur.open
                and cur.open >= prev.close
                and cur.close <= prev.open
            ):
                is_bearish_engulfing = True

    return CandleMetrics(
        body_ratio=body_ratio,
        upper_wick_ratio=upper_wick_ratio,
        lower_wick_ratio=lower_wick_ratio,
        is_bullish_rejection=is_bullish_rejection,
        is_bearish_rejection=is_bearish_rejection,
        is_bullish_engulfing=is_bullish_engulfing,
        is_bearish_engulfing=is_bearish_engulfing,
    )
