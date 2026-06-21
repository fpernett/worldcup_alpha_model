from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.utils import clamp, coerce_bool, coerce_float


RESIDUAL_COLUMNS = [
    "expected_result_score",
    "actual_result_score",
    "result_residual",
    "expected_goals_for",
    "expected_goals_against",
    "goals_for_residual",
    "goals_against_residual",
    "goals_for_residual_capped",
    "goals_against_residual_capped",
    "blowout_weight",
]

DEFAULT_RESIDUAL_CAP = 1.75


def calculate_expected_result_from_elo(
    team_elo: Any,
    opponent_elo: Any,
    neutral: bool = True,
    home_advantage: float = 35,
) -> float:
    team_rating = coerce_float(team_elo, float("nan"))
    opponent_rating = coerce_float(opponent_elo, float("nan"))
    if pd.isna(team_rating) or pd.isna(opponent_rating):
        return float("nan")
    adjusted_team_elo = team_rating if neutral else team_rating + float(home_advantage)
    return float(1.0 / (1.0 + 10.0 ** ((opponent_rating - adjusted_team_elo) / 400.0)))


def calculate_expected_goals_from_elo(
    team_elo: Any,
    opponent_elo: Any,
    base_goals: float = 1.35,
) -> float:
    team_rating = coerce_float(team_elo, float("nan"))
    opponent_rating = coerce_float(opponent_elo, float("nan"))
    if pd.isna(team_rating) or pd.isna(opponent_rating):
        return float("nan")
    expected_goals = float(base_goals) * math.exp((team_rating - opponent_rating) / 900.0)
    return clamp(expected_goals, 0.25, 3.50)


def add_performance_residuals(matches_df: pd.DataFrame | None, residual_cap: float = DEFAULT_RESIDUAL_CAP) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        columns = list(matches_df.columns) if isinstance(matches_df, pd.DataFrame) else []
        return pd.DataFrame(columns=columns + [col for col in RESIDUAL_COLUMNS if col not in columns])

    out = matches_df.copy()
    for col in ["team_elo_pre", "opponent_elo", "team_goals", "opponent_goals", "is_home", "is_neutral"]:
        if col not in out.columns:
            out[col] = pd.NA
    for col in ["team_elo_pre", "opponent_elo", "team_goals", "opponent_goals"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["expected_result_score"] = out.apply(_expected_result_for_row, axis=1)
    out["actual_result_score"] = out.apply(_actual_result_for_row, axis=1)
    out["result_residual"] = out["actual_result_score"] - out["expected_result_score"]
    out["expected_goals_for"] = out.apply(
        lambda row: calculate_expected_goals_from_elo(row.get("team_elo_pre"), row.get("opponent_elo")),
        axis=1,
    )
    out["expected_goals_against"] = out.apply(
        lambda row: calculate_expected_goals_from_elo(row.get("opponent_elo"), row.get("team_elo_pre")),
        axis=1,
    )
    out["goals_for_residual"] = out["team_goals"] - out["expected_goals_for"]
    out["goals_against_residual"] = out["opponent_goals"] - out["expected_goals_against"]
    out["goals_for_residual_capped"] = out["goals_for_residual"].map(lambda value: cap_residual(value, residual_cap))
    out["goals_against_residual_capped"] = out["goals_against_residual"].map(lambda value: cap_residual(value, residual_cap))
    out["blowout_weight"] = out.apply(lambda row: blowout_weight(row.get("team_goals"), row.get("opponent_goals")), axis=1)
    return out


def cap_residual(value: Any, cap: float = DEFAULT_RESIDUAL_CAP) -> float:
    residual = coerce_float(value, float("nan"))
    if pd.isna(residual):
        return float("nan")
    cap_value = abs(float(cap))
    return clamp(residual, -cap_value, cap_value)


def blowout_weight(team_goals: Any = None, opponent_goals: Any = None, goal_difference: Any = None) -> float:
    if goal_difference is None:
        goals_for = coerce_float(team_goals, float("nan"))
        goals_against = coerce_float(opponent_goals, float("nan"))
        if pd.isna(goals_for) or pd.isna(goals_against):
            return 1.0
        margin = abs(goals_for - goals_against)
    else:
        margin = abs(coerce_float(goal_difference, float("nan")))
        if pd.isna(margin):
            return 1.0
    if margin <= 2:
        return 1.0
    return clamp(1.0 / (1.0 + 0.35 * (margin - 2.0)), 0.55, 1.0)


def attack_index_from_goal_residual(weighted_goal_for_residual: Any) -> float:
    residual = coerce_float(weighted_goal_for_residual, float("nan"))
    if pd.isna(residual):
        return 0.5
    return clamp(0.5 + residual / 3.0, 0.0, 1.0)


def defense_index_from_goal_residual(weighted_goal_against_residual: Any) -> float:
    residual = coerce_float(weighted_goal_against_residual, float("nan"))
    if pd.isna(residual):
        return 0.5
    return clamp(0.5 - residual / 3.0, 0.0, 1.0)


def final_index_from_raw_and_residual(raw_index: Any, residual_index: Any) -> float:
    raw = coerce_float(raw_index, 0.5)
    residual = coerce_float(residual_index, 0.5)
    return clamp(0.70 * raw + 0.30 * residual, 0.0, 1.0)


def _expected_result_for_row(row: pd.Series) -> float:
    team_elo = row.get("team_elo_pre")
    opponent_elo = row.get("opponent_elo")
    neutral = coerce_bool(row.get("is_neutral", True))
    is_home = coerce_bool(row.get("is_home", False))
    if neutral:
        return calculate_expected_result_from_elo(team_elo, opponent_elo, neutral=True)
    if is_home:
        return calculate_expected_result_from_elo(team_elo, opponent_elo, neutral=False)
    team_rating = coerce_float(team_elo, float("nan"))
    opponent_rating = coerce_float(opponent_elo, float("nan"))
    if pd.isna(team_rating) or pd.isna(opponent_rating):
        return float("nan")
    return float(1.0 / (1.0 + 10.0 ** (((opponent_rating + 35.0) - team_rating) / 400.0)))


def _actual_result_for_row(row: pd.Series) -> float:
    goals_for = coerce_float(row.get("team_goals"), float("nan"))
    goals_against = coerce_float(row.get("opponent_goals"), float("nan"))
    if pd.isna(goals_for) or pd.isna(goals_against):
        return float("nan")
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0
