from __future__ import annotations

from typing import Any

import pandas as pd

from src.ratings import get_team_ratings
from src.utils import coerce_float
from src.weather import (
    ENVIRONMENT_COLUMNS,
    VENUE_COLUMNS,
    environment_from_venue_row,
    get_venue_environment,
    load_venues,
    summarize_environment_adjustment,
)


CAPITAL_CLIMATE_DEFAULTS = {
    "Argentina": (18, 68),
    "Belgium": (12, 78),
    "Bosnia and Herzegovina": (14, 70),
    "Brazil": (24, 70),
    "Canada": (10, 65),
    "Colombia": (19, 72),
    "Croatia": (18, 65),
    "Czechia": (12, 70),
    "DR Congo": (26, 75),
    "England": (12, 75),
    "France": (13, 70),
    "Germany": (11, 72),
    "Ghana": (27, 78),
    "Mexico": (22, 60),
    "Netherlands": (12, 78),
    "Panama": (28, 80),
    "Portugal": (17, 68),
    "Qatar": (31, 60),
    "South Africa": (20, 55),
    "South Korea": (15, 65),
    "Spain": (18, 60),
    "Switzerland": (10, 70),
    "Uzbekistan": (23, 45),
}


def get_team_training_climate(team: str) -> dict[str, Any]:
    """Return approximate usual training climate for a national team.

    Confidence hierarchy:
    1. manual values in `team_ratings.csv` -> high;
    2. country/team climate defaults -> moderate;
    3. neutral defaults -> low.
    """
    ratings = get_team_ratings()
    row = ratings.loc[ratings["team"].astype(str).str.lower() == str(team).lower()]
    if not row.empty:
        temp = row["training_temp_c"].iloc[0]
        humidity = row["training_humidity_pct"].iloc[0]
        if pd.notna(temp) and pd.notna(humidity):
            return {
                "training_temp_c": coerce_float(temp, 20.0),
                "training_humidity_pct": coerce_float(humidity, 60.0),
                "source": "team_ratings_csv",
                "confidence": "high",
            }

    if team in CAPITAL_CLIMATE_DEFAULTS:
        temp, humidity = CAPITAL_CLIMATE_DEFAULTS[team]
        return {
            "training_temp_c": temp,
            "training_humidity_pct": humidity,
            "source": "national_capital_climate_proxy",
            "confidence": "moderate",
        }

    return {
        "training_temp_c": 20.0,
        "training_humidity_pct": 60.0,
        "source": "neutral_default",
        "confidence": "low",
    }
