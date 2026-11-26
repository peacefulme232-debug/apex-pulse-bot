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

    @model_validator(mode="after")
    def enforce_keyless_paper_fallback(self) -> "Settings":
        """Safety Lock: Default BYBIT_MODE=paper if BYBIT_API_KEY is blank."""
        if not (self.BYBIT_API_KEY or "").strip():
            object.__setattr__(self, "BYBIT_MODE", "paper")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
