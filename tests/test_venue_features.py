from __future__ import annotations

import pandas as pd

from src.model import ModelConfig, expected_goals
from src.venue_features import classify_venue_context


def _team(name: str) -> pd.Series:
    return pd.Series(
        {
            "team": name,
            "elo": 1700,
            "attack": 0.60,
            "defense": 0.60,
            "recent_form": 0.60,
            "training_temp_c": 20,
            "training_humidity_pct": 55,
        }
    )


def test_azteca_mexico_host_high_altitude_is_not_neutral() -> None:
    env = {
        "venue": "Estadio Azteca",
        "country": "Mexico",
        "altitude_m": 2240,
        "temp_c": 20,
        "humidity_pct": 45,
        "wind_kmh": 8,
        "precipitation_mm": 0,
        "roof_expected_closed": 0,
        "neutral_site": 0,
    }

    hxg_host, axg_host, components = expected_goals(_team("Mexico"), _team("Ecuador"), env, ModelConfig())
    neutral_env = {**env, "country": "USA", "neutral_site": 1}
    hxg_neutral, _axg_neutral, neutral_components = expected_goals(_team("Mexico"), _team("Ecuador"), neutral_env, ModelConfig())

    assert classify_venue_context("Mexico", "Ecuador", env) == "home_host"
    assert components["venue_context"] == "home_host"
    assert components["altitude_category"] == "high_altitude"
    assert components["home_venue_log_adj"] > 0
    assert neutral_components["venue_context"] == "neutral"
    assert hxg_host > hxg_neutral
    assert hxg_host > axg_host
