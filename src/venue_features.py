from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import clamp, coerce_bool, coerce_float


ALTITUDE_FAMILIARITY_DEFAULTS_M = {
    "bolivia": 3000.0,
    "colombia": 1600.0,
    "ecuador": 2400.0,
    "mexico": 1900.0,
    "peru": 2300.0,
}


def normalise_location_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().replace(".", "").split())


def _as_dict(value: dict[str, Any] | pd.Series | None) -> dict[str, Any]:
    if value is None:
        return {}
    return value.to_dict() if isinstance(value, pd.Series) else dict(value)


def is_same_country_or_team(team: Any, country: Any) -> bool:
    team_key = normalise_location_name(team)
    country_key = normalise_location_name(country)
    if not team_key or not country_key:
        return False
    aliases = {
        "usa": {"united states", "united states of america", "usa"},
        "united states": {"united states", "united states of america", "usa"},
        "czechia": {"czech republic", "czechia"},
        "dr congo": {"congo dr", "dr congo", "democratic republic of congo"},
        "ivory coast": {"cote divoire", "ivory coast"},
        "south korea": {"korea republic", "south korea"},
    }
    team_values = aliases.get(team_key, {team_key})
    country_values = aliases.get(country_key, {country_key})
    return bool(team_values & country_values)


def classify_venue_context(home_team: Any, away_team: Any, env_or_row: dict[str, Any] | pd.Series | None) -> str:
    env = _as_dict(env_or_row)
    country = env.get("country", "")
    neutral_raw = env.get("neutral_site", True)
    neutral_site = True if neutral_raw is None or pd.isna(neutral_raw) else coerce_bool(neutral_raw)
    home_host = is_same_country_or_team(home_team, country)
    away_host = is_same_country_or_team(away_team, country)
    if home_host and not away_host:
        return "home_host"
    if away_host and not home_host:
        return "away_host"
    if home_host and away_host:
        return "shared_host"
    if neutral_site:
        return "neutral"
    return "designated_home"


def altitude_category(altitude_m: Any) -> str:
    altitude = coerce_float(altitude_m, 0.0)
    if altitude >= 1800.0:
        return "high_altitude"
    if altitude >= 1000.0:
        return "moderate_altitude"
    return "low_altitude"


def team_altitude_familiarity_m(team_row: pd.Series | dict[str, Any] | None, team_name: Any = "") -> float:
    row = _as_dict(team_row)
    explicit = coerce_float(row.get("training_altitude_m"), float("nan"))
    if not pd.isna(explicit):
        return max(explicit, 0.0)
    key = normalise_location_name(row.get("team", "") or team_name)
    return ALTITUDE_FAMILIARITY_DEFAULTS_M.get(key, 0.0)


def altitude_log_penalty(team_row: pd.Series | dict[str, Any] | None, altitude_m: Any, team_name: Any = "") -> float:
    altitude = coerce_float(altitude_m, 0.0)
    if altitude <= 1200.0:
        return 0.0
    familiar = team_altitude_familiarity_m(team_row, team_name)
    tolerated = max(1200.0, familiar)
    gap = max(altitude - tolerated, 0.0)
    return clamp(-0.000050 * gap, -0.10, 0.0)


def venue_log_adjustments(home_team: Any, away_team: Any, env_or_row: dict[str, Any] | pd.Series | None) -> dict[str, Any]:
    env = _as_dict(env_or_row)
    context = classify_venue_context(home_team, away_team, env)
    home_adj = 0.0
    away_adj = 0.0
    if context == "home_host":
        home_adj = 0.09
    elif context == "away_host":
        away_adj = 0.09
    elif context == "designated_home":
        home_adj = 0.03
    return {
        "venue_context": context,
        "home_venue_log_adj": home_adj,
        "away_venue_log_adj": away_adj,
        "altitude_category": altitude_category(env.get("altitude_m", 0.0)),
        "venue_country": env.get("country", ""),
        "neutral_site": bool(True if env.get("neutral_site", True) is None or pd.isna(env.get("neutral_site", True)) else coerce_bool(env.get("neutral_site", True))),
    }
