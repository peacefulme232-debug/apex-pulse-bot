"""Stockfish-grade 0-100 Confluence Evaluator & Hard Veto Engine.

Every CandidateSignal emitted by the 3 synced strategies MUST pass through
`ConfluenceEvaluator.evaluate(...)` before sizing or execution.

5 Deterministic Scoring Pillars (0-100 total):
1. HTF Alignment (0-25 pts)
2. Level / Rejection Precision (0-25 pts)
3. Momentum & Divergence (0-20 pts)
4. Volatility & Liquidity Session (0-15 pts)
5. Post-Fee Net R:R (0-15 pts, queried via realism_engine.estimate_round_turn_friction)

6 Hard Veto Gates (any failure = instant reject + log to execution_rejections):
1. FEE_DRAG_TOO_HIGH (fee_drag_pct_of_tp > max_fee_drag_pct OR net_rr < 1.85)
2. OPPOSING_DIVERGENCE_VETO (Bearish H4/D1 div blocks Longs; Bullish blocks Shorts)
3. MICRO_CORRELATION_LIMIT (active_balance <= $100 -> max 1 pos/market + 0 correlated bucket overlap)
4. LOSS_STREAK_CIRCUIT_BREAKER (2 consecutive losses OR daily trading loss >= 4.0% from equity_baseline_day)
5. SYMBOL_COOLDOWN (3 closed H1 candles after a Stop-Loss hit on same symbol)
6. LOW_CONFLUENCE_SCORE (total_score < min_confluence_score, default 75, configurable 65-90)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

from app.models import ExecutionRejection, Trade, WalletAccount
from app.services.indicators import (
    adx_wilder,
    atr_percentile,
    atr_wilder,
    candle_metrics,
    ema,
    filter_closed_candles,
    rsi_wilder,
)
from app.services.realism_engine import (
    ZERO,
    estimate_round_turn_friction,
    q6,
    to_decimal,
)
from app.services.regime_router import RegimeDecision
from app.services.strategies.base import CandidateSignal
from app.services.strategies.rsi_divergence_sniper import (
    ActiveDivergence,
    RsiDivergenceSniperStrategy,
)
from app.services.wallet_service import compute_daily_pnl_isolated, lock_wallet_account


CORRELATION_BUCKETS: tuple[frozenset[str], ...] = (
    frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"}),
    frozenset({"EUR_USD", "GBP_USD", "EURUSD", "GBPUSD"}),
)


def _norm_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _symbol_bucket(symbol: str) -> Optional[frozenset[str]]:
    sym = _norm_symbol(symbol)
    for bucket in CORRELATION_BUCKETS:
        if sym in bucket:
            return bucket
    return None


@dataclass
class BrainConfig:
    """Mutable runtime configuration for the Strategy Brain, Evaluator, and Trade Manager."""

    ema_50_pullback_enabled: bool = True
    breakout_retest_enabled: bool = True
    rsi_divergence_sniper_enabled: bool = True
    min_confluence_score: int = 75
    max_fee_drag_pct: float = 15.0
    min_net_rr: float = 1.85
    breakeven_r_trigger: float = 1.0
    max_consecutive_losses: int = 2
    max_daily_loss_pct: float = 4.0
    symbol_cooldown_bars: int = 3
    micro_shield_enabled: bool = True
    micro_shield_threshold_usd: float = 100.0

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        name = (strategy_name or "").strip().upper()
        if name == "EMA_50_PULLBACK":
            return bool(self.ema_50_pullback_enabled)
        if name == "BREAKOUT_RETEST":
            return bool(self.breakout_retest_enabled)
        if name == "RSI_DIVERGENCE_SNIPER":
            return bool(self.rsi_divergence_sniper_enabled)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ema_50_pullback_enabled": self.ema_50_pullback_enabled,
            "breakout_retest_enabled": self.breakout_retest_enabled,
            "rsi_divergence_sniper_enabled": self.rsi_divergence_sniper_enabled,
            "min_confluence_score": self.min_confluence_score,
            "max_fee_drag_pct": self.max_fee_drag_pct,
            "min_net_rr": self.min_net_rr,
            "breakeven_r_trigger": self.breakeven_r_trigger,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "symbol_cooldown_bars": self.symbol_cooldown_bars,
            "micro_shield_enabled": self.micro_shield_enabled,
            "micro_shield_threshold_usd": self.micro_shield_threshold_usd,
        }


@dataclass(frozen=True)
class ConfluenceBreakdown:
    """Deterministic 5-pillar breakdown summing to 0-100."""

    htf_alignment: int
    level_precision: int
    momentum_divergence: int
    volatility_liquidity: int
    post_fee_rr: int

    @property
    def total_score(self) -> int:
        return (
            self.htf_alignment
            + self.level_precision
            + self.momentum_divergence
            + self.volatility_liquidity
            + self.post_fee_rr
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "htf_alignment": self.htf_alignment,
            "level_precision": self.level_precision,
            "momentum_divergence": self.momentum_divergence,
            "volatility_liquidity": self.volatility_liquidity,
            "post_fee_rr": self.post_fee_rr,
            "total_score": self.total_score,
        }


@dataclass(frozen=True)
class ConfluenceEvaluation:
    """Result of evaluating a CandidateSignal through all 5 pillars and 6 hard vetoes."""

    approved: bool
    total_score: int
    breakdown: ConfluenceBreakdown
    vetoes: list[str]
    primary_veto: Optional[str]
    details: str
    fee_drag_pct_of_tp: float
    net_rr: float
    estimated_total_friction_tp: float
    opposing_divergence: Optional[str] = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "total_score": self.total_score,
            "breakdown": self.breakdown.to_dict(),
            "vetoes": list(self.vetoes),
            "primary_veto": self.primary_veto,
            "details": self.details,
            "fee_drag_pct_of_tp": round(self.fee_drag_pct_of_tp, 4),
            "net_rr": round(self.net_rr, 4),
            "estimated_total_friction_tp": round(self.estimated_total_friction_tp, 6),
            "opposing_divergence": self.opposing_divergence,
            "telemetry": dict(self.telemetry),
        }


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ConfluenceEvaluator:
    """Evaluates CandidateSignals on 5 deterministic pillars (0-100) + 6 hard veto gates."""

    def __init__(
        self,
        config: Optional[BrainConfig] = None,
        divergence_detector: Optional[RsiDivergenceSniperStrategy] = None,
    ) -> None:
        self.config = config or BrainConfig()
        self.divergence_detector = divergence_detector or RsiDivergenceSniperStrategy()

    def evaluate(
        self,
        signal: CandidateSignal,
        *,
        db: Optional[Session] = None,
        h1_candles: Optional[Any] = None,
        h4_candles: Optional[Any] = None,
        d1_candles: Optional[Any] = None,
        regime_decision: Optional[RegimeDecision] = None,
        active_divergence: Optional[ActiveDivergence] = None,
        estimated_qty: Optional[Decimal | float] = None,
        leverage: Optional[int] = None,
        account_profile: Optional[str] = None,
        active_balance: Optional[Decimal | float] = None,
        current_active_equity: Optional[Decimal | float] = None,
        open_positions: Optional[Sequence[Any]] = None,
        bars_since_last_sl: Optional[int] = None,
        now_utc: Optional[datetime] = None,
        log_rejection: bool = True,
    ) -> ConfluenceEvaluation:
        """Score a CandidateSignal (0-100), enforce all 6 hard vetoes, and audit rejections."""
        now = _as_utc(now_utc) or datetime.now(timezone.utc)
        norm_market = (signal.market or "bybit").strip().lower()
        side = signal.side.upper()

        wallet: Optional[WalletAccount] = None
        if db is not None:
            wallet = lock_wallet_account(db, norm_market)

        resolved_active_balance = (
            to_decimal(active_balance)
            if active_balance is not None
            else (to_decimal(wallet.active_balance) if wallet is not None else Decimal("1000.0"))
        )
        resolved_equity = (
            to_decimal(current_active_equity)
            if current_active_equity is not None
            else resolved_active_balance
        )
        resolved_leverage = leverage
        if resolved_leverage is None:
            if wallet is not None:
                resolved_leverage = (
                    int(wallet.bybit_leverage or 5)
                    if norm_market == "bybit"
                    else int(wallet.forex_leverage or 500)
                )
            else:
                resolved_leverage = 5 if norm_market == "bybit" else 500

        resolved_profile = account_profile
        if resolved_profile is None:
            if wallet is not None and wallet.account_profile:
                resolved_profile = str(wallet.account_profile)
            else:
                resolved_profile = "bybit_vip0_linear" if norm_market == "bybit" else "mt5_raw_ecn"

        # Resolve qty for round-turn friction estimation
        if estimated_qty is not None and to_decimal(estimated_qty) > ZERO:
            friction_qty = to_decimal(estimated_qty)
        elif norm_market == "forex":
            friction_qty = Decimal("0.01")
        else:
            # Estimate 1% risk qty (with small positive floor) for accurate Bybit friction ratio
            risk_per_unit = abs(signal.entry_price - signal.sl_price)
            risk_budget = max(float(resolved_active_balance) * 0.01, 0.20)
            raw_qty = (risk_budget / risk_per_unit) if risk_per_unit > 1e-12 else 0.01
            friction_qty = to_decimal(max(raw_qty, 0.001))

        # --- Closed candles & indicator context ---
        h1 = filter_closed_candles(h1_candles) if h1_candles is not None else []
        h4 = filter_closed_candles(h4_candles) if h4_candles is not None else []
        d1 = filter_closed_candles(d1_candles) if d1_candles is not None else []

        # --- Detect active HTF divergence (for Pillar 3 & Hard Veto #2) ---
        resolved_div = active_divergence
        if resolved_div is None:
            if d1:
                resolved_div = self.divergence_detector.detect_active_divergence(d1)
            if resolved_div is None and h4:
                resolved_div = self.divergence_detector.detect_active_divergence(h4)

        opposing_div_kind: Optional[str] = None
        if side == "LONG":
            if (resolved_div is not None and resolved_div.kind == "BEARISH") or (
                regime_decision is not None and regime_decision.veto_longs
            ):
                opposing_div_kind = "BEARISH"
        else:
            if (resolved_div is not None and resolved_div.kind == "BULLISH") or (
                regime_decision is not None and regime_decision.veto_shorts
            ):
                opposing_div_kind = "BULLISH"

        # -------------------------------------------------------------------
        # PILLAR 1: HTF Alignment (0-25 pts)
        # -------------------------------------------------------------------
        htf_score = self._score_htf_alignment(signal, h4, d1)

        # -------------------------------------------------------------------
        # PILLAR 2: Level / Rejection Precision (0-25 pts)
        # -------------------------------------------------------------------
        level_score = self._score_level_precision(signal, h1)

        # -------------------------------------------------------------------
        # PILLAR 3: Momentum & Divergence (0-20 pts)
        # -------------------------------------------------------------------
        momentum_score, mom_meta = self._score_momentum_divergence(
            signal=signal,
            h1=h1,
            opposing_div_active=(opposing_div_kind is not None),
        )

        # -------------------------------------------------------------------
        # PILLAR 4: Volatility & Liquidity Session (0-15 pts)
        # -------------------------------------------------------------------
        vol_score, vol_meta = self._score_volatility_liquidity(
            market=norm_market,
            h1=h1,
            now_utc=now,
        )

        # -------------------------------------------------------------------
        # PILLAR 5: Post-Fee Net R:R (0-15 pts) via realism_engine
        # -------------------------------------------------------------------
        friction = estimate_round_turn_friction(
            market=norm_market,
            symbol=signal.symbol,
            side=side,
            qty=friction_qty,
            entry_price=signal.entry_price,
            sl_price=signal.sl_price,
            tp_price=signal.tp_price,
            leverage=resolved_leverage,
            profile=resolved_profile,
            entry_order_type=signal.order_entry_type or "MARKET",
            tp_order_type=signal.order_exit_type or "TAKE_PROFIT",
            sl_order_type="STOP_LOSS",
        )
        fee_drag_pct = float(friction.fee_drag_pct_of_tp)
        net_rr = float(friction.net_rr)

        if net_rr >= 2.5:
            rr_score = 15
        elif net_rr >= 2.2:
            rr_score = 12
        elif net_rr >= 2.0:
            rr_score = 8
        elif net_rr >= self.config.min_net_rr:
            rr_score = 4
        else:
            rr_score = 0

        breakdown = ConfluenceBreakdown(
            htf_alignment=htf_score,
            level_precision=level_score,
            momentum_divergence=momentum_score,
            volatility_liquidity=vol_score,
            post_fee_rr=rr_score,
        )
        total_score = breakdown.total_score

        # -------------------------------------------------------------------
        # HARD VETO GATES (Any failure = instant reject)
        # -------------------------------------------------------------------
        vetoes: list[str] = []
        veto_reasons: list[str] = []

        # Gate 1: FEE_DRAG_TOO_HIGH (CRITICAL FOR $20)
        if fee_drag_pct > self.config.max_fee_drag_pct or net_rr < self.config.min_net_rr:
            vetoes.append("FEE_DRAG_TOO_HIGH")
            veto_reasons.append(
                f"Fee drag {fee_drag_pct:.2f}% (max {self.config.max_fee_drag_pct:.1f}%) "
                f"or post-fee Net R:R {net_rr:.2f} < {self.config.min_net_rr:.2f}."
            )

        # Gate 2: OPPOSING_DIVERGENCE_VETO
        if opposing_div_kind is not None:
            vetoes.append("OPPOSING_DIVERGENCE_VETO")
            veto_reasons.append(
                f"Active HTF {opposing_div_kind} RSI(14) divergence vetoes {side} entry."
            )

        # Gate 3: MICRO_CORRELATION_LIMIT (active_balance <= $100)
        if (
            self.config.micro_shield_enabled
            and resolved_active_balance <= to_decimal(self.config.micro_shield_threshold_usd)
        ):
            open_list = self._resolve_open_positions(db, open_positions)
            same_market_open = [
                p for p in open_list if str(p["market"]).strip().lower() == norm_market
            ]
            target_bucket = _symbol_bucket(signal.symbol)
            correlated_open = [
                p
                for p in open_list
                if target_bucket is not None and _norm_symbol(p["symbol"]) in target_bucket
            ]
            if len(same_market_open) >= 1 or len(correlated_open) >= 1:
                vetoes.append("MICRO_CORRELATION_LIMIT")
                open_syms = ", ".join(p["symbol"] for p in (correlated_open or same_market_open))
                veto_reasons.append(
                    f"Micro-Account Shield (${float(resolved_active_balance):.2f} <= "
                    f"${self.config.micro_shield_threshold_usd:.0f}): blocked {signal.symbol} "
                    f"while open position(s) exist ({open_syms})."
                )

        # Gate 4: LOSS_STREAK_CIRCUIT_BREAKER
        breaker_triggered, breaker_msg = self._check_circuit_breaker(
            db=db,
            wallet=wallet,
            current_active_equity=resolved_equity,
            now_utc=now,
        )
        if breaker_triggered:
            vetoes.append("LOSS_STREAK_CIRCUIT_BREAKER")
            veto_reasons.append(breaker_msg)

        # Gate 5: SYMBOL_COOLDOWN (3 closed H1 candles after SL hit)
        cooldown_blocked, cooldown_msg = self._check_symbol_cooldown(
            db=db,
            symbol=signal.symbol,
            bars_since_last_sl=bars_since_last_sl,
            now_utc=now,
        )
        if cooldown_blocked:
            vetoes.append("SYMBOL_COOLDOWN")
            veto_reasons.append(cooldown_msg)

        # Gate 6: LOW_CONFLUENCE_SCORE
        if total_score < self.config.min_confluence_score:
            vetoes.append("LOW_CONFLUENCE_SCORE")
            veto_reasons.append(
                f"Confluence score {total_score}/100 < minimum {self.config.min_confluence_score}."
            )

        approved = len(vetoes) == 0
        primary_veto = vetoes[0] if vetoes else None
        details = (
            f"Approved ({total_score}/100, net_rr={net_rr:.2f}, fee_drag={fee_drag_pct:.2f}%)"
            if approved
            else " | ".join(veto_reasons)
        )

        # Log rejection to execution_rejections table
        if not approved and log_rejection and db is not None and primary_veto is not None:
            rejection = ExecutionRejection(
                market=norm_market,
                symbol=signal.symbol,
                side=side,
                strategy=signal.strategy,
                confluence_score=Decimal(str(total_score)),
                reason_code=primary_veto,
                requested_qty=q6(friction_qty),
                available_free_margin=q6(resolved_active_balance),
                details=details,
            )
            db.add(rejection)
            db.flush()

        return ConfluenceEvaluation(
            approved=approved,
            total_score=total_score,
            breakdown=breakdown,
            vetoes=vetoes,
            primary_veto=primary_veto,
            details=details,
            fee_drag_pct_of_tp=fee_drag_pct,
            net_rr=net_rr,
            estimated_total_friction_tp=float(friction.total_friction_tp),
            opposing_divergence=opposing_div_kind,
            telemetry={
                **mom_meta,
                **vol_meta,
                "fee_drag_pct_of_tp": round(fee_drag_pct, 4),
                "net_rr": round(net_rr, 4),
                "friction_qty": float(friction_qty),
            },
        )

    # ------------------------------------------------------------------
    # Pillar Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tf_aligned(candles: Sequence[Any], side: str) -> Optional[bool]:
        if not candles:
            return None
        closes = [c.close for c in candles]
        if len(closes) >= 200:
            ref = ema(closes, 200)[-1]
        elif len(closes) >= 50:
            ref = ema(closes, 50)[-1]
        else:
            ref = closes[0]
        if ref is None:
            return None
        return (closes[-1] > ref) if side == "LONG" else (closes[-1] < ref)

    def _score_htf_alignment(
        self,
        signal: CandidateSignal,
        h4: Sequence[Any],
        d1: Sequence[Any],
    ) -> int:
        if signal.strategy == "RSI_DIVERGENCE_SNIPER":
            return 15
        side = signal.side.upper()
        h4_ok = self._is_tf_aligned(h4, side)
        d1_ok = self._is_tf_aligned(d1, side)
        if h4_ok and d1_ok:
            return 25
        if h4_ok:
            return 18
        # Fallback to signal metadata if H4 candles were not passed directly
        if h4_ok is None and "h4_close" in signal.metadata and "h4_ema200" in signal.metadata:
            h4_c = float(signal.metadata["h4_close"])
            h4_e = float(signal.metadata["h4_ema200"])
            if (side == "LONG" and h4_c > h4_e) or (side == "SHORT" and h4_c < h4_e):
                return 18
        return 0

    @staticmethod
    def _score_level_precision(signal: CandidateSignal, h1: Sequence[Any]) -> int:
        side = signal.side.upper()
        pattern = str(signal.metadata.get("pattern", "")).lower()
        if "rejection" in pattern or "engulfing" in pattern:
            return 25
        if len(h1) >= 2:
            m = candle_metrics(h1[-1], h1[-2])
            if side == "LONG":
                if m.is_bullish_rejection or m.is_bullish_engulfing:
                    return 25
                if h1[-1].close > h1[-1].open:
                    return 15
            else:
                if m.is_bearish_rejection or m.is_bearish_engulfing:
                    return 25
                if h1[-1].close < h1[-1].open:
                    return 15
        if signal.strategy == "BREAKOUT_RETEST":
            return 25
        return 15

    @staticmethod
    def _score_momentum_divergence(
        signal: CandidateSignal,
        h1: Sequence[Any],
        opposing_div_active: bool,
    ) -> tuple[int, dict[str, Any]]:
        side = signal.side.upper()
        meta: dict[str, Any] = {
            "h1_rsi": None,
            "h1_plus_di": None,
            "h1_minus_di": None,
        }
        if opposing_div_active:
            return 0, meta

        h1_rsi: Optional[float] = None
        plus_di: Optional[float] = None
        minus_di: Optional[float] = None

        if len(h1) >= 29:
            highs = [c.high for c in h1]
            lows = [c.low for c in h1]
            closes = [c.close for c in h1]
            h1_rsi = rsi_wilder(closes, 14)[-1]
            _, pdi_seq, mdi_seq = adx_wilder(highs, lows, closes, 14)
            plus_di = pdi_seq[-1]
            minus_di = mdi_seq[-1]
        elif "h1_rsi" in signal.metadata:
            h1_rsi = float(signal.metadata["h1_rsi"])

        meta["h1_rsi"] = h1_rsi
        meta["h1_plus_di"] = plus_di
        meta["h1_minus_di"] = minus_di

        rsi_ok = False
        if h1_rsi is not None:
            rsi_ok = (35.0 <= h1_rsi <= 62.0) if side == "LONG" else (38.0 <= h1_rsi <= 65.0)
        elif signal.strategy == "RSI_DIVERGENCE_SNIPER":
            rsi_ok = True

        di_ok = False
        if plus_di is not None and minus_di is not None:
            di_ok = (plus_di > minus_di) if side == "LONG" else (minus_di > plus_di)
        elif signal.strategy == "RSI_DIVERGENCE_SNIPER":
            di_ok = True

        if rsi_ok and di_ok:
            return 20, meta
        if rsi_ok or di_ok:
            return 12, meta
        return 6, meta

    @staticmethod
    def _score_volatility_liquidity(
        market: str,
        h1: Sequence[Any],
        now_utc: datetime,
    ) -> tuple[int, dict[str, Any]]:
        atr_pct: Optional[float] = None
        vol_ratio: Optional[float] = None

        if len(h1) >= 15:
            highs = [c.high for c in h1]
            lows = [c.low for c in h1]
            closes = [c.close for c in h1]
            atr_seq = atr_wilder(highs, lows, closes, 14)
            atr_pct = atr_percentile(atr_seq, window=100)[-1]

            vols = [c.volume for c in h1[-20:]]
            avg_vol = sum(vols) / len(vols) if vols else 0.0
            vol_ratio = (h1[-1].volume / avg_vol) if avg_vol > 1e-12 else 1.0

        atr_ok = atr_pct is not None and 20.0 <= atr_pct <= 85.0
        if market == "forex":
            liquidity_ok = 7 <= now_utc.hour < 20
        else:
            liquidity_ok = vol_ratio is None or vol_ratio >= 1.0

        meta = {
            "h1_atr_percentile": atr_pct,
            "volume_ratio": vol_ratio,
            "session_hour_utc": now_utc.hour,
        }
        if atr_ok and liquidity_ok:
            return 15, meta
        if atr_ok or liquidity_ok:
            return 8, meta
        return 0, meta

    # ------------------------------------------------------------------
    # Veto Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_open_positions(
        db: Optional[Session],
        open_positions: Optional[Sequence[Any]],
    ) -> list[dict[str, str]]:
        if open_positions is not None:
            resolved: list[dict[str, str]] = []
            for p in open_positions:
                if isinstance(p, dict):
                    resolved.append({"symbol": str(p["symbol"]), "market": str(p["market"])})
                else:
                    resolved.append({"symbol": str(p.symbol), "market": str(p.market)})
            return resolved
        if db is None:
            return []
        rows = db.query(Trade).filter(Trade.status == "OPEN").all()
        return [{"symbol": str(t.symbol), "market": str(t.market)} for t in rows]

    def _check_circuit_breaker(
        self,
        *,
        db: Optional[Session],
        wallet: Optional[WalletAccount],
        current_active_equity: Decimal,
        now_utc: datetime,
    ) -> tuple[bool, str]:
        if wallet is None:
            return False, ""

        until = _as_utc(wallet.circuit_breaker_until)
        if until is not None and now_utc < until:
            return (
                True,
                f"Circuit breaker active on {wallet.market} until {until.isoformat()} "
                f"(consecutive_losses={wallet.consecutive_losses}).",
            )

        if int(wallet.consecutive_losses or 0) >= self.config.max_consecutive_losses:
            wallet.circuit_breaker_until = now_utc + timedelta(hours=24)
            if db is not None:
                db.flush()
            return (
                True,
                f"Loss streak circuit breaker triggered on {wallet.market}: "
                f"{wallet.consecutive_losses} >= {self.config.max_consecutive_losses} consecutive losses.",
            )

        # Daily trading loss % computed strictly from equity_baseline_day (NEVER vault withdrawals)
        _daily_usd, daily_pnl_pct_dec = compute_daily_pnl_isolated(
            current_active_equity=current_active_equity,
            equity_baseline_day=to_decimal(wallet.equity_baseline_day),
            fallback_capital=to_decimal(wallet.active_balance),
        )
        daily_pnl_pct = float(daily_pnl_pct_dec)
        if daily_pnl_pct <= -abs(self.config.max_daily_loss_pct):
            wallet.circuit_breaker_until = now_utc + timedelta(hours=24)
            if db is not None:
                db.flush()
            return (
                True,
                f"Daily loss circuit breaker triggered on {wallet.market}: "
                f"daily trading loss {daily_pnl_pct:.2f}% <= -{self.config.max_daily_loss_pct:.2f}%.",
            )

        return False, ""

    def _check_symbol_cooldown(
        self,
        *,
        db: Optional[Session],
        symbol: str,
        bars_since_last_sl: Optional[int],
        now_utc: datetime,
    ) -> tuple[bool, str]:
        if bars_since_last_sl is not None:
            if bars_since_last_sl < self.config.symbol_cooldown_bars:
                return (
                    True,
                    f"Symbol cooldown on {symbol}: {bars_since_last_sl} < "
                    f"{self.config.symbol_cooldown_bars} closed H1 candles since SL hit.",
                )
            return False, ""

        if db is None:
            return False, ""

        last_sl = (
            db.query(Trade)
            .filter(
                Trade.symbol == symbol,
                Trade.status == "CLOSED",
                Trade.order_exit_type == "STOP_LOSS",
            )
            .order_by(Trade.closed_at.desc(), Trade.id.desc())
            .first()
        )
        if last_sl is None or last_sl.closed_at is None:
            return False, ""

        closed_at = _as_utc(last_sl.closed_at)
        if closed_at is None:
            return False, ""

        elapsed_hours = (now_utc - closed_at).total_seconds() / 3600.0
        if elapsed_hours < float(self.config.symbol_cooldown_bars):
            return (
                True,
                f"Symbol cooldown on {symbol}: {elapsed_hours:.2f}h < "
                f"{self.config.symbol_cooldown_bars} closed H1 candles since SL hit.",
            )
        return False, ""
