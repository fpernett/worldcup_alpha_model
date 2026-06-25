from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.historical_binding import audit_historical_binding_for_match
from src.utils import coerce_bool, coerce_float


LEAGUE_BASELINES = {
    "goals_per_match": 2.56,
    "home_win": 0.471,
    "draw": 0.251,
    "away_win": 0.278,
    "btts": 0.443,
    "over_2_5": 0.468,
}


def build_data_support(match_row: pd.Series, historical_matches_df: pd.DataFrame | None) -> dict[str, Any]:
    home = str(match_row.get("home", ""))
    away = str(match_row.get("away", ""))
    as_of_date = str(match_row.get("date_utc", "") or "") or None
    matches = _normalise_history(historical_matches_df)

    if matches.empty:
        return {
            "home_canonical": home,
            "away_canonical": away,
            "home_team_match_count": 0,
            "away_team_match_count": 0,
            "h2h_match_count": 0,
            "last_h2h_date": "",
            "h2h_note": "No historical match data available.",
            "training_data_start": "",
            "training_data_end": "",
            "training_match_count": 0,
            "warnings": ["Historical match data missing; report uses baseline context where needed."],
        }

    binding = audit_historical_binding_for_match(
        home,
        away,
        historical_matches_df if historical_matches_df is not None else pd.DataFrame(),
        as_of_date=as_of_date,
    )

    h2h_count = int(binding["h2h_rows_before_asof"] if as_of_date else binding["h2h_rows_total"])
    if h2h_count == 0:
        h2h_note = "First meeting in training data"
        last_h2h_date = ""
    else:
        h2h_rows = _canonical_h2h_rows(matches, binding["home_canonical"], binding["away_canonical"])
        if as_of_date:
            h2h_rows = h2h_rows.loc[h2h_rows["date_utc"] < pd.Timestamp(as_of_date)]
        last_h2h_date = _date_label(h2h_rows["date_utc"].max()) if not h2h_rows.empty else ""
        h2h_note = f"{h2h_count} long-format meeting row(s) in training data"

    team_rows_before = int(binding["home_historical_rows_before_asof"]) + int(binding["away_historical_rows_before_asof"])
    team_dates = matches.loc[
        _team_mask(matches, binding["home_canonical"]) | _team_mask(matches, binding["away_canonical"])
    ].copy()
    if as_of_date and not team_dates.empty:
        team_dates = team_dates.loc[team_dates["date_utc"] < pd.Timestamp(as_of_date)]

    return {
        "home_canonical": binding["home_canonical"],
        "away_canonical": binding["away_canonical"],
        "home_team_match_count": int(binding["home_historical_rows_before_asof"]),
        "away_team_match_count": int(binding["away_historical_rows_before_asof"]),
        "h2h_match_count": h2h_count,
        "last_h2h_date": last_h2h_date,
        "h2h_note": h2h_note,
        "training_data_start": _date_label(team_dates["date_utc"].min()) if not team_dates.empty else "",
        "training_data_end": _date_label(team_dates["date_utc"].max()) if not team_dates.empty else "",
        "training_match_count": team_rows_before,
        "warnings": [binding["warning"]] if binding.get("warning") else [],
    }


def calculate_goal_distribution(score_matrix: pd.DataFrame | None, max_goals: int = 14) -> pd.DataFrame:
    rows = []
    matrix = score_matrix.copy() if score_matrix is not None else pd.DataFrame()
    for col in ["home_goals", "away_goals", "prob"]:
        if col not in matrix.columns:
            matrix[col] = pd.NA
    matrix["home_goals"] = pd.to_numeric(matrix["home_goals"], errors="coerce")
    matrix["away_goals"] = pd.to_numeric(matrix["away_goals"], errors="coerce")
    matrix["prob"] = pd.to_numeric(matrix["prob"], errors="coerce").fillna(0.0)

    for goals in range(max_goals + 1):
        rows.append(
            {
                "goals": goals,
                "home_probability": float(matrix.loc[matrix["home_goals"] == goals, "prob"].sum()),
                "away_probability": float(matrix.loc[matrix["away_goals"] == goals, "prob"].sum()),
            }
        )
    return pd.DataFrame(rows)


def calculate_league_context(historical_matches_df: pd.DataFrame | None, model_result: dict[str, Any]) -> pd.DataFrame:
    metrics = _league_metrics(historical_matches_df)
    probs = model_result.get("probs", {}) if isinstance(model_result, dict) else {}
    this_match = {
        "goals_per_match": coerce_float(model_result.get("hxg"), 0.0) + coerce_float(model_result.get("axg"), 0.0),
        "home_win": coerce_float(probs.get("home_win"), 0.0),
        "draw": coerce_float(probs.get("draw"), 0.0),
        "away_win": coerce_float(probs.get("away_win"), 0.0),
        "btts": coerce_float(probs.get("btts_yes"), 0.0),
        "over_2_5": coerce_float(probs.get("over_2_5"), 0.0),
    }
    labels = [
        ("Goals per match", "goals_per_match", "goals"),
        ("Home/team1 win %", "home_win", "probability"),
        ("Draw %", "draw", "probability"),
        ("Away/team2 win %", "away_win", "probability"),
        ("Both teams to score %", "btts", "probability"),
        ("Over 2.5 goals %", "over_2_5", "probability"),
    ]

    rows = []
    for label, key, unit in labels:
        league = float(metrics[key])
        match_value = float(this_match[key])
        rows.append(
            {
                "Metric": label,
                "League": league,
                "This match": match_value,
                "Delta": match_value - league,
                "unit": unit,
                "source": metrics.get("source", "fallback baseline"),
            }
        )
    return pd.DataFrame(rows)


def build_expected_goals_df(model_result: dict[str, Any]) -> pd.DataFrame:
    home = str(model_result.get("home", "Home"))
    away = str(model_result.get("away", "Away"))
    hxg = coerce_float(model_result.get("hxg"), 0.0)
    axg = coerce_float(model_result.get("axg"), 0.0)
    df = pd.DataFrame(
        [
            {"team": home, "base_xg": hxg, "adjusted_xg": hxg, "lower": hxg * 0.65, "upper": hxg * 1.45},
            {"team": away, "base_xg": axg, "adjusted_xg": axg, "lower": axg * 0.65, "upper": axg * 1.45},
        ]
    )
    df.attrs["note"] = "TODO: base xG is not separately stored yet; adjusted xG is used as base for this v1 report."
    return df


def calculate_team_rating_percentiles(teams_df: pd.DataFrame | None, home: str, away: str) -> pd.DataFrame:
    teams = teams_df.copy() if teams_df is not None else pd.DataFrame()
    for col in ["team", "attack", "defense"]:
        if col not in teams.columns:
            teams[col] = pd.NA
    teams["attack"] = pd.to_numeric(teams["attack"], errors="coerce")
    teams["defense"] = pd.to_numeric(teams["defense"], errors="coerce")

    rows = []
    for team in [home, away]:
        row = teams.loc[teams["team"].astype(str).str.lower() == str(team).lower()]
        attack = coerce_float(row["attack"].iloc[0], 0.55) if not row.empty else 0.55
        defense = coerce_float(row["defense"].iloc[0], 0.55) if not row.empty else 0.55
        attack_pct = _percentile(teams["attack"], attack)
        defense_pct = _percentile(teams["defense"], defense)
        rows.append(
            {
                "team": team,
                "attack": attack,
                "defense": defense,
                "attack_percentile": attack_pct,
                "defense_percentile": defense_pct,
                "attack_label": _top_label(attack_pct),
                "defense_label": _top_label(defense_pct),
            }
        )
    return pd.DataFrame(rows)


def build_climate_factor_table(model_result: dict[str, Any], venue_environment: dict[str, Any] | None) -> pd.DataFrame:
    env = venue_environment or {}
    components = model_result.get("components", {}) if isinstance(model_result, dict) else {}
    home = str(model_result.get("home", "Home"))
    away = str(model_result.get("away", "Away"))
    roof_closed = coerce_bool(env.get("roof_expected_closed", components.get("roof_expected_closed", 0)))
    weather_multiplier = 0.15 if roof_closed else 1.0

    home_adj = coerce_float(components.get("home_environment_log_adj"), 0.0)
    away_adj = coerce_float(components.get("away_environment_log_adj"), 0.0)
    total_adj = coerce_float(components.get("total_environment_log_adj"), 0.0)
    favours = _favours(home, away, home_adj, away_adj)

    altitude = coerce_float(env.get("altitude_m", components.get("altitude_m")), 0.0)
    temp = coerce_float(env.get("effective_temp_c", env.get("temp_c", components.get("effective_temp_c"))), 22.0)
    humidity = coerce_float(env.get("effective_humidity_pct", env.get("humidity_pct", components.get("effective_humidity_pct"))), 55.0)
    precipitation = coerce_float(
        env.get("effective_precipitation_mm", env.get("precipitation_mm", components.get("effective_precipitation_mm"))),
        0.0,
    )
    wind = coerce_float(env.get("effective_wind_kmh", env.get("wind_kmh", components.get("effective_wind_kmh"))), 0.0)

    rows = [
        {
            "factor": "Altitude",
            "value": altitude,
            "unit": "m",
            "category": _altitude_category(altitude),
            "favours": favours,
            "multiplier": _altitude_multiplier(altitude),
        },
        {
            "factor": "Temperature",
            "value": temp,
            "unit": "C",
            "category": _temperature_category(temp),
            "favours": favours,
            "multiplier": _factor_multiplier(max(temp - 30.0, 0.0) * -0.0035 * weather_multiplier),
        },
        {
            "factor": "Humidity",
            "value": humidity,
            "unit": "%",
            "category": _humidity_category(humidity),
            "favours": favours,
            "multiplier": _factor_multiplier(max(humidity - 75.0, 0.0) * -0.0015 * weather_multiplier),
        },
        {
            "factor": "Precipitation",
            "value": precipitation,
            "unit": "mm",
            "category": _precipitation_category(precipitation),
            "favours": "Lower scoring" if precipitation > 0 else "Neutral",
            "multiplier": _factor_multiplier(-0.008 * precipitation * weather_multiplier),
        },
        {
            "factor": "Wind",
            "value": wind,
            "unit": "km/h",
            "category": _wind_category(wind),
            "favours": "Lower scoring" if wind > 12 else "Neutral",
            "multiplier": _factor_multiplier(-0.0045 * max(wind - 12.0, 0.0) * weather_multiplier),
        },
    ]
    out = pd.DataFrame(rows)
    out.attrs["team_factor_note"] = (
        f"{home} environment factor: x{math.exp(home_adj):.3f}; "
        f"{away} environment factor: x{math.exp(away_adj):.3f}; "
        f"total-goals factor: x{math.exp(total_adj):.3f}."
    )
    if roof_closed:
        out.attrs["roof_note"] = "Roof expected closed: outdoor weather multipliers are heavily dampened."
    return out


def _normalise_history(historical_matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if historical_matches_df is None or historical_matches_df.empty:
        return pd.DataFrame(columns=["date_utc", "home", "away", "home_goals", "away_goals"])
    out = historical_matches_df.copy()
    if "home" not in out.columns and "team" in out.columns:
        out["home"] = out["team"]
    if "away" not in out.columns and "opponent" in out.columns:
        out["away"] = out["opponent"]
    if "home_goals" not in out.columns and "team_goals" in out.columns:
        out["home_goals"] = out["team_goals"]
    if "away_goals" not in out.columns and "opponent_goals" in out.columns:
        out["away_goals"] = out["opponent_goals"]
    for col in ["date_utc", "home", "away", "home_goals", "away_goals"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce")
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"]).copy()
    return out.sort_values("date_utc").reset_index(drop=True)


def _team_mask(matches: pd.DataFrame, team: str) -> pd.Series:
    team_l = str(team).lower()
    return (matches["home"].astype(str).str.lower() == team_l) | (matches["away"].astype(str).str.lower() == team_l)


def _canonical_h2h_rows(matches: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    home_l = str(home).lower()
    away_l = str(away).lower()
    return matches.loc[
        (
            (matches["home"].astype(str).str.lower() == home_l)
            & (matches["away"].astype(str).str.lower() == away_l)
        )
        | (
            (matches["home"].astype(str).str.lower() == away_l)
            & (matches["away"].astype(str).str.lower() == home_l)
        )
    ].copy()


def _date_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _league_metrics(historical_matches_df: pd.DataFrame | None) -> dict[str, Any]:
    matches = _normalise_history(historical_matches_df)
    if matches.empty:
        return {**LEAGUE_BASELINES, "source": "fallback baseline"}

    total_goals = matches["home_goals"] + matches["away_goals"]
    home_goals = matches["home_goals"]
    away_goals = matches["away_goals"]
    return {
        "goals_per_match": float(total_goals.mean()),
        "home_win": float((home_goals > away_goals).mean()),
        "draw": float((home_goals == away_goals).mean()),
        "away_win": float((home_goals < away_goals).mean()),
        "btts": float(((home_goals > 0) & (away_goals > 0)).mean()),
        "over_2_5": float((total_goals >= 3).mean()),
        "source": "historical matches",
    }


def _percentile(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).mean() * 100.0)


def _top_label(percentile: float) -> str:
    top = max(1, min(100, round(100 - percentile)))
    return f"Top {top}%"


def _favours(home: str, away: str, home_adj: float, away_adj: float) -> str:
    diff = home_adj - away_adj
    if abs(diff) < 0.01:
        return "Neutral"
    return home if diff > 0 else away


def _factor_multiplier(log_adj: float) -> float:
    return float(np.clip(math.exp(log_adj), 0.80, 1.05))


def _altitude_multiplier(altitude_m: float) -> float:
    if altitude_m <= 1200.0:
        return 1.0
    return _factor_multiplier(-0.000025 * (altitude_m - 1200.0))


def _altitude_category(value: float) -> str:
    if value >= 2200:
        return "Very high"
    if value >= 1200:
        return "High"
    if value <= 100:
        return "Low"
    return "Normal"


def _temperature_category(value: float) -> str:
    if value >= 34:
        return "Very high"
    if value >= 29:
        return "High"
    if value <= 5:
        return "Low"
    return "Normal"


def _humidity_category(value: float) -> str:
    if value >= 85:
        return "Very high"
    if value >= 70:
        return "High"
    if value <= 35:
        return "Low"
    return "Normal"


def _precipitation_category(value: float) -> str:
    if value >= 8:
        return "Very high"
    if value >= 3:
        return "High"
    if value > 0:
        return "Low"
    return "Normal"


def _wind_category(value: float) -> str:
    if value >= 35:
        return "Very high"
    if value >= 20:
        return "High"
    if value <= 5:
        return "Low"
    return "Normal"
