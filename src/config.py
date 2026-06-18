from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ENV_PATH = PROJECT_ROOT / ".env"

SOURCE_API = "API"
SOURCE_CACHE = "cache"
SOURCE_LOCAL = "local CSV"
SOURCE_FALLBACK = "fallback"


@dataclass(frozen=True)
class APIConfig:
    football_api_url: str | None
    football_api_key: str | None
    ratings_api_url: str | None
    weather_api_url: str | None
    weather_api_key: str | None
    odds_api_url: str | None
    odds_api_key: str | None
    polymarket_api_url: str | None
    polymarket_clob_api_url: str | None
    polymarket_gamma_api_url: str | None

    @property
    def football_configured(self) -> bool:
        return bool(self.football_api_url and self.football_api_key)

    @property
    def ratings_configured(self) -> bool:
        return bool(self.ratings_api_url)

    @property
    def weather_configured(self) -> bool:
        # Open-Meteo does not need a key, so URL is sufficient.
        return bool(self.weather_api_url)

    @property
    def odds_configured(self) -> bool:
        return bool(self.odds_api_url)

    @property
    def polymarket_configured(self) -> bool:
        return bool(self.polymarket_api_url or self.polymarket_gamma_api_url or self.polymarket_clob_api_url)


def load_env() -> None:
    load_dotenv(ENV_PATH, override=False)


def env_value(name: str, default: str | None = None) -> str | None:
    load_env()
    value = os.getenv(name, default)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def get_config() -> APIConfig:
    return APIConfig(
        football_api_url=env_value("FOOTBALL_API_URL"),
        football_api_key=env_value("FOOTBALL_API_KEY"),
        ratings_api_url=env_value("RATINGS_API_URL"),
        weather_api_url=env_value("WEATHER_API_URL"),
        weather_api_key=env_value("WEATHER_API_KEY"),
        odds_api_url=env_value("ODDS_API_URL"),
        odds_api_key=env_value("ODDS_API_KEY"),
        polymarket_api_url=env_value("POLYMARKET_API_URL"),
        polymarket_clob_api_url=env_value("POLYMARKET_CLOB_API_URL"),
        polymarket_gamma_api_url=env_value("POLYMARKET_GAMMA_API_URL"),
    )


def api_summary() -> dict[str, bool]:
    cfg = get_config()
    return {
        "fixtures": cfg.football_configured,
        "ratings": cfg.ratings_configured,
        "weather": cfg.weather_configured,
        "odds": cfg.odds_configured,
        "polymarket": cfg.polymarket_configured,
    }
