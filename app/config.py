"""Typed application configuration with PAPER mode safety lock."""

from functools import lru_cache
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_LEVERAGE = 3


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    TRADING_MODE: str = "PAPER"
    DATABASE_URL: str = "sqlite:///./apex_pulse.db"
    LOOP_INTERVAL_SECONDS: int = 10
    MAX_RISK_PER_TRADE_PCT: float = 1.0
    MAX_DAILY_LOSS_PCT: float = 3.0

    # Phase 3: Bybit V5 & Institutional Confluence Settings
    BYBIT_MODE: str = "paper"  # paper | demo (mainnet live blocked in Phase 3)
    BYBIT_CATEGORY: str = "linear"  # linear | spot
    BYBIT_API_KEY: str = ""
    BYBIT_API_SECRET: str = ""
    BYBIT_SYMBOLS: str = "BTCUSDT,ETHUSDT,SOLUSDT"
    BYBIT_PRIMARY_INTERVAL: str = "15m"
    BYBIT_HTF_INTERVAL: str = "4h"
    BYBIT_LEVERAGE: int = 2
    CONFLUENCE_MIN_SCORE: int = 65

    # Phase 4: Cloud Forex (OANDA Practice / Deriv Demo) & Session/News Guard
    FOREX_PROVIDER: str = "paper"  # paper | deriv | oanda
    FOREX_SYMBOLS: str = "EUR_USD,GBP_USD,XAU_USD"
    FOREX_PRIMARY_INTERVAL: str = "15m"
    FOREX_HTF_INTERVAL: str = "4h"
    FOREX_KILLZONE_ONLY: bool = True
    NEWS_BLACKOUT_MINUTES: int = 30
    OANDA_API_KEY: str = ""
    OANDA_ACCOUNT_ID: str = ""
    OANDA_BASE_URL: str = "https://api-fxpractice.oanda.com/v3"
    DERIV_API_TOKEN: str = ""
    DERIV_APP_ID: str = "1089"
    DERIV_ACCOUNT_ID: str = ""
    DERIV_AUTH_MODE: str = "auto"  # auto | otp | legacy

    # Phase 5: +1.0R Partial Take-Profit Scale-Out
    PARTIAL_TP_ENABLED: bool = True

    # Phase 6: Security (Phone PIN Gate), 2-Way Cron Engine, & Telegram Sleep Alerts
    DASHBOARD_PIN: str = ""
    SESSION_SECRET: str = ""
    SESSION_TTL_HOURS: int = 24
    ENABLE_BACKGROUND_LOOP: bool = True
    CRON_SECRET: str = ""
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        """Convert legacy postgres:// scheme to postgresql:// for SQLAlchemy 2.x."""
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql://", 1)
        return value

    @field_validator("TRADING_MODE")
    @classmethod
    def normalize_trading_mode(cls, value: str) -> str:
        """Ensure trading mode is uppercase and defaults safely."""
        cleaned = (value or "PAPER").strip().upper()
        return cleaned if cleaned in {"PAPER", "LIVE"} else "PAPER"

    @field_validator("BYBIT_MODE")
    @classmethod
    def validate_bybit_mode(cls, value: str) -> str:
        """Only allow 'paper' or 'demo' in Phase 3; explicitly block live mainnet."""
        cleaned = (value or "paper").strip().lower()
        if cleaned not in {"paper", "demo"}:
            raise ValueError(
                f"BYBIT_MODE='{value}' is blocked in Phase 3. Only 'paper' or 'demo' is permitted."
            )
        return cleaned

    @field_validator("BYBIT_CATEGORY")
    @classmethod
    def validate_bybit_category(cls, value: str) -> str:
        """Validate Bybit V5 category (linear or spot)."""
        cleaned = (value or "linear").strip().lower()
        if cleaned not in {"linear", "spot"}:
            raise ValueError(f"BYBIT_CATEGORY='{value}' must be 'linear' or 'spot'.")
        return cleaned

    @field_validator("BYBIT_LEVERAGE")
    @classmethod
    def validate_bybit_leverage(cls, value: int) -> int:
        """Hard-cap linear perpetual leverage at MAX_LEVERAGE=3."""
        if value < 1 or value > MAX_LEVERAGE:
            raise ValueError(f"BYBIT_LEVERAGE must be between 1 and {MAX_LEVERAGE}, got {value}.")
        return value

    @field_validator("CONFLUENCE_MIN_SCORE")
    @classmethod
    def validate_confluence_min_score(cls, value: int) -> int:
        """Validate confluence minimum score is within 0-100."""
        if value < 0 or value > 100:
            raise ValueError(f"CONFLUENCE_MIN_SCORE must be between 0 and 100, got {value}.")
        return value

    @field_validator("FOREX_PROVIDER")
    @classmethod
    def validate_forex_provider(cls, value: str) -> str:
        """Validate Forex provider is one of paper, deriv, or oanda."""
        cleaned = (value or "paper").strip().lower()
        if cleaned not in {"paper", "deriv", "oanda"}:
            raise ValueError(
                f"FOREX_PROVIDER='{value}' must be 'paper', 'deriv', or 'oanda'."
            )
        return cleaned

    @field_validator("DERIV_AUTH_MODE")
    @classmethod
    def validate_deriv_auth_mode(cls, value: str) -> str:
        """Validate Deriv auth mode is one of auto, otp, or legacy."""
        cleaned = (value or "auto").strip().lower()
        if cleaned not in {"auto", "otp", "legacy"}:
            raise ValueError(
                f"DERIV_AUTH_MODE='{value}' must be 'auto', 'otp', or 'legacy'."
            )
        return cleaned

    @field_validator("OANDA_BASE_URL")
    @classmethod
    def validate_oanda_base_url(cls, value: str) -> str:
        """Restrict OANDA strictly to Practice/Demo URL in Phase 4."""
        cleaned = (value or "https://api-fxpractice.oanda.com/v3").strip().rstrip("/")
        if cleaned != "https://api-fxpractice.oanda.com/v3":
            raise ValueError(
                "OANDA_BASE_URL is restricted strictly to Practice/Demo "
                "'https://api-fxpractice.oanda.com/v3' in Phase 4."
            )
        return cleaned

    @field_validator("NEWS_BLACKOUT_MINUTES")
    @classmethod
    def validate_news_blackout_minutes(cls, value: int) -> int:
        """Ensure news blackout window is non-negative."""
        if value < 0 or value > 240:
            raise ValueError(f"NEWS_BLACKOUT_MINUTES must be between 0 and 240, got {value}.")
        return value

    @field_validator("SESSION_TTL_HOURS")
    @classmethod
    def validate_session_ttl_hours(cls, value: int) -> int:
        """Ensure session cookie TTL is between 1 hour and 30 days."""
        if value < 1 or value > 720:
            raise ValueError(f"SESSION_TTL_HOURS must be between 1 and 720, got {value}.")
        return value

    @model_validator(mode="after")
    def enforce_keyless_paper_fallback(self) -> "Settings":
        """Safety Lock: Default BYBIT_MODE=paper and FOREX_PROVIDER=paper if credentials are blank."""
        if not (self.BYBIT_API_KEY or "").strip():
            object.__setattr__(self, "BYBIT_MODE", "paper")
        if self.FOREX_PROVIDER == "oanda":
            if not (self.OANDA_API_KEY or "").strip() or not (self.OANDA_ACCOUNT_ID or "").strip():
                object.__setattr__(self, "FOREX_PROVIDER", "paper")
        elif self.FOREX_PROVIDER == "deriv":
            if not (self.DERIV_API_TOKEN or "").strip():
                object.__setattr__(self, "FOREX_PROVIDER", "paper")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
