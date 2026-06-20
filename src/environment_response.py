from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.recency import calculate_match_weight
from src.utils import clamp


ENVIRONMENT_RESPONSE_COLUMNS = [
    "team",
    "environment_sample_size",
    "hot_match_count",
    "humid_match_count",
    "altitude_match_count",
    "windy_match_count",
    "rain_match_count",
    "roof_closed_match_count",
    "hot_attack_delta",
    "hot_defense_delta",
    "humid_attack_delta",
    "humid_defense_delta",
    "altitude_attack_delta",
    "altitude_defense_delta",
    "wind_attack_delta",
    "wind_defense_delta",
    "rain_attack_delta",
    "rain_defense_delta",
    "roof_closed_attack_delta",
    "roof_closed_defense_delta",
    "environment_response_index",
    "environment_data_quality",
    "environment_warning",
]


def calculate_environment_response(matches_df: pd.DataFrame, team: str, reference_date: Any) -> dict[str, Any]:
    """Estimate conservative team response under prior environmental conditions.

    Deltas are goals-per-match differences. Attack deltas are condition goals-for
    minus non-condition goals-for. Defense deltas are non-condition goals-against
    minus condition goals-against, so positive means the team conceded less in
    the condition.
    """
    view = _team_view(matches_df, team)
    base = _neutral_response(team)
    if view.empty:
        base["environment_warning"] = "No historical environment data available; neutral response used."
        return base

    env_cols = ["temperature_c", "humidity_pct", "altitude_m", "wind_kmh", "precipitation_mm", "roof_closed"]
    for col in env_cols + ["team_goals", "opponent_goals"]:
        view[col] = pd.to_numeric(view.get(col, pd.Series(index=view.index)), errors="coerce")
    view = view.dropna(subset=["team_goals", "opponent_goals"]).copy()
    env_view = view.loc[view[env_cols].notna().any(axis=1)].copy()
    if env_view.empty:
        base["environment_warning"] = "Environment columns are missing for this team; neutral response used."
        return base

    env_view["match_weight"] = env_view.apply(lambda row: calculate_match_weight(row, reference_date), axis=1)
    env_view["match_weight"] = pd.to_numeric(env_view["match_weight"], errors="coerce").fillna(0.0)
    if float(env_view["match_weight"].sum()) <= 0:
        env_view["match_weight"] = 1.0

    conditions = {
        "hot": env_view["temperature_c"] >= 28,
        "humid": env_view["humidity_pct"] >= 70,
        "altitude": env_view["altitude_m"] >= 1000,
        "wind": env_view["wind_kmh"] >= 20,
        "rain": env_view["precipitation_mm"] > 0,
        "roof_closed": env_view["roof_closed"].fillna(0).astype(float) >= 1,
    }

    response = {
        "team": team,
        "environment_sample_size": int(len(env_view)),
        "hot_match_count": int(conditions["hot"].fillna(False).sum()),
        "humid_match_count": int(conditions["humid"].fillna(False).sum()),
        "altitude_match_count": int(conditions["altitude"].fillna(False).sum()),
        "windy_match_count": int(conditions["wind"].fillna(False).sum()),
        "rain_match_count": int(conditions["rain"].fillna(False).sum()),
        "roof_closed_match_count": int(conditions["roof_closed"].fillna(False).sum()),
    }

    all_deltas: list[float] = []
    for label, mask in conditions.items():
        attack_delta, defense_delta = _condition_deltas(env_view, mask)
        response[f"{label}_attack_delta"] = attack_delta
        response[f"{label}_defense_delta"] = defense_delta
        if not np.isnan(attack_delta):
            all_deltas.append(float(attack_delta))
        if not np.isnan(defense_delta):
            all_deltas.append(float(defense_delta))

    if response["environment_sample_size"] < 5:
        quality = "low"
        warning = "Fewer than 5 environment-tagged matches; response confidence is low."
    elif max(response["hot_match_count"], response["humid_match_count"], response["altitude_match_count"]) < 5:
        quality = "low"
        warning = "Condition-specific samples are below 5 matches; response confidence is low."
    elif response["environment_sample_size"] >= 15:
        quality = "high"
        warning = ""
    else:
        quality = "moderate"
        warning = ""

    avg_delta = float(np.nanmean(all_deltas)) if all_deltas else 0.0
    capped_delta = clamp(avg_delta, -0.35, 0.35)
    response["environment_response_index"] = round(clamp(0.5 + capped_delta * 0.35, 0.35, 0.65), 3)
    response["environment_data_quality"] = quality
    response["environment_warning"] = warning
    return response


def _condition_deltas(view: pd.DataFrame, mask: pd.Series) -> tuple[float, float]:
    condition_mask = mask.fillna(False)
    if int(condition_mask.sum()) == 0 or int((~condition_mask).sum()) == 0:
        return float("nan"), float("nan")
    condition = view.loc[condition_mask]
    baseline = view.loc[~condition_mask]
    attack_delta = _weighted_mean(condition, "team_goals") - _weighted_mean(baseline, "team_goals")
    defense_delta = _weighted_mean(baseline, "opponent_goals") - _weighted_mean(condition, "opponent_goals")
    return round(clamp(attack_delta, -0.35, 0.35), 3), round(clamp(defense_delta, -0.35, 0.35), 3)


def _weighted_mean(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df[column], errors="coerce")
    weights = pd.to_numeric(df["match_weight"], errors="coerce").fillna(0.0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def _team_view(matches_df: pd.DataFrame | None, team: str) -> pd.DataFrame:
    if matches_df is None or matches_df.empty or "team" not in matches_df.columns:
        return pd.DataFrame()
    return matches_df.loc[matches_df["team"].astype(str).str.lower() == str(team).lower()].copy()


def _neutral_response(team: str) -> dict[str, Any]:
    return {
        "team": team,
        "environment_sample_size": 0,
        "hot_match_count": 0,
        "humid_match_count": 0,
        "altitude_match_count": 0,
        "windy_match_count": 0,
        "rain_match_count": 0,
        "roof_closed_match_count": 0,
        "hot_attack_delta": 0.0,
        "hot_defense_delta": 0.0,
        "humid_attack_delta": 0.0,
        "humid_defense_delta": 0.0,
        "altitude_attack_delta": 0.0,
        "altitude_defense_delta": 0.0,
        "wind_attack_delta": 0.0,
        "wind_defense_delta": 0.0,
        "rain_attack_delta": 0.0,
        "rain_defense_delta": 0.0,
        "roof_closed_attack_delta": 0.0,
        "roof_closed_defense_delta": 0.0,
        "environment_response_index": 0.5,
        "environment_data_quality": "none",
        "environment_warning": "No historical environment data available; neutral response used.",
    }
