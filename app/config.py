"""Typed application configuration with PAPER mode safety lock."""

from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    TRADING_MODE: str = "PAPER"
    DATABASE_URL: str = "sqlite:///./apex_pulse.db"
    LOOP_INTERVAL_SECONDS: int = 10
    MAX_RISK_PER_TRADE_PCT: float = 1.0
    MAX_DAILY_LOSS_PCT: float = 3.0

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


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
