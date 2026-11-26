"""Pure pandas/numpy vectorized technical indicators with zero-lookahead guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CANDLES = 205
REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def has_sufficient_candles(df: pd.DataFrame, min_candles: int = MIN_CANDLES) -> bool:
    """Return True if the DataFrame has enough bars for full indicator warmup."""
    return df is not None and len(df) >= min_candles


def ema(series: pd.Series, span: int) -> pd.Series:
    """Compute Exponential Moving Average using pandas ewm (strictly causal)."""
    return series.astype(float).ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index using Wilder's smoothing with zero-division guards.

    Explicitly guards:
    - avg_gain == 0 and avg_loss == 0 -> 50.0 (flat market)
    - avg_loss == 0 and avg_gain > 0  -> 100.0 (pure upward move)
    """
    prices = series.astype(float)
    delta = prices.diff()
    gain = delta.clip(lower=0.0).fillna(0.0)
    loss = (-delta.clip(upper=0.0)).fillna(0.0)

    alpha = 1.0 / float(period)
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    safe_loss = avg_loss.where(avg_loss > 1e-12, np.nan)
    rs = avg_gain / safe_loss
    raw_rsi = 100.0 - (100.0 / (1.0 + rs))

    both_zero = (avg_gain <= 1e-12) & (avg_loss <= 1e-12)
    loss_zero_only = (avg_loss <= 1e-12) & (avg_gain > 1e-12)

    result = np.where(both_zero, 50.0, np.where(loss_zero_only, 100.0, raw_rsi))
    return pd.Series(result, index=series.index, dtype=float)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range using True Range and Wilder's smoothing."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.ewm(alpha=1.0 / float(period), adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, +DI, and -DI using Wilder's smoothing with zero-denominator guards."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    alpha = 1.0 / float(period)
    atr_series = atr(df, period=period)
    safe_atr = atr_series.where(atr_series > 1e-12, np.nan)

    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = (100.0 * smoothed_plus_dm / safe_atr).fillna(0.0)
    minus_di = (100.0 * smoothed_minus_dm / safe_atr).fillna(0.0)

    di_sum = plus_di + minus_di
    safe_di_sum = di_sum.where(di_sum > 1e-12, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / safe_di_sum).fillna(0.0)
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)

    return adx_series, plus_di, minus_di


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
    percentile_window: int = 50,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands, bandwidth, and rolling 50-bar bandwidth percentile (0-100)."""
    prices = series.astype(float)
    bb_mid = prices.rolling(window=period, min_periods=period).mean()
    bb_std = prices.rolling(window=period, min_periods=period).std(ddof=0)

    bb_upper = bb_mid + std_dev * bb_std
    bb_lower = bb_mid - std_dev * bb_std

    safe_mid = bb_mid.where(bb_mid.abs() > 1e-12, np.nan)
    bb_width = (bb_upper - bb_lower) / safe_mid

    bbw_percentile = bb_width.rolling(window=percentile_window, min_periods=1).rank(pct=True) * 100.0
    return bb_upper, bb_mid, bb_lower, bb_width, bbw_percentile


def donchian_channels(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series]:
    """Compute Donchian Channels shifted by 1 bar so candle i compares strictly to PRIOR bars."""
    dc_high = df["high"].astype(float).rolling(window=period, min_periods=period).max().shift(1)
    dc_low = df["low"].astype(float).rolling(window=period, min_periods=period).min().shift(1)
    return dc_high, dc_low


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a non-mutated copy of `df` enriched with all Phase 2 indicators.

    All calculations are strictly causal (zero lookahead) and guarded against
    zero/NaN edge cases. Sets `result.attrs["sufficient_history"]` based on
    `MIN_CANDLES` (205 bars).
    """
    out = df.copy(deep=True)
    out.attrs["min_candles"] = MIN_CANDLES
    out.attrs["sufficient_history"] = has_sufficient_candles(out, MIN_CANDLES)

    if out.empty:
        for col in (
            "ema9",
            "ema21",
            "ema50",
            "ema200",
            "rsi",
            "atr",
            "atr_pct",
            "adx",
            "plus_di",
            "minus_di",
            "bb_upper",
            "bb_mid",
            "bb_lower",
            "bb_width",
            "bbw_percentile",
            "dc_high",
            "dc_low",
        ):
            out[col] = pd.Series(dtype=float)
        return out

    close = out["close"].astype(float)

    out["ema9"] = ema(close, 9)
    out["ema21"] = ema(close, 21)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)

    out["rsi"] = rsi(close, period=14)

    atr_series = atr(out, period=14)
    out["atr"] = atr_series
    safe_close = close.replace(0.0, np.nan)
    out["atr_pct"] = (atr_series / safe_close) * 100.0

    adx_series, plus_di, minus_di = adx(out, period=14)
    out["adx"] = adx_series
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    bb_upper, bb_mid, bb_lower, bb_width, bbw_pct = bollinger_bands(close, period=20, std_dev=2.0)
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width
    out["bbw_percentile"] = bbw_pct

    dc_high, dc_low = donchian_channels(out, period=20)
    out["dc_high"] = dc_high
    out["dc_low"] = dc_low

    return out
