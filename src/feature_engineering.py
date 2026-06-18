from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.cache import read_dataframe_cache, set_source_attrs
from src.config import SOURCE_CACHE, SOURCE_LOCAL
from src.utils import DATA_DIR, clamp, coerce_float, read_csv_with_columns


MATCH_COLUMNS = [
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "home_elo",
    "away_elo",
]


def _fallback_team_value(team: str, column: str, default: float) -> float:
    ratings_path = DATA_DIR / "team_ratings.csv"
    ratings = read_csv_with_columns(ratings_path, ["team", column])
    if ratings.empty or "team" not in ratings:
        return default
    row = ratings.loc[ratings["team"].astype(str).str.lower() == str(team).lower()]
    if row.empty:
        return default
    return clamp(coerce_float(row[column].iloc[0], default), 0.0, 1.0)


def _normalise_matches(matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    df = matches_df.copy()
    for col in MATCH_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    for col in ["home_goals", "away_goals", "home_elo", "away_elo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"])
    return df.sort_values("date_utc", ascending=False)


def _team_match_view(matches_df: pd.DataFrame, team: str) -> pd.DataFrame:
    df = _normalise_matches(matches_df)
    if df.empty:
        return df

    rows = []
    team_key = str(team).lower()
    for _, row in df.iterrows():
        home = str(row["home"])
        away = str(row["away"])
        if home.lower() == team_key:
            scored = row["home_goals"]
            conceded = row["away_goals"]
            opponent = away
            opponent_elo = row.get("away_elo", np.nan)
            result = _result_from_goals(scored, conceded)
        elif away.lower() == team_key:
            scored = row["away_goals"]
            conceded = row["home_goals"]
            opponent = home
            opponent_elo = row.get("home_elo", np.nan)
            result = _result_from_goals(scored, conceded)
        else:
            continue

        rows.append(
            {
                "date_utc": row["date_utc"],
                "team": team,
                "opponent": opponent,
                "goals_for": float(scored),
                "goals_against": float(conceded),
                "opponent_elo": opponent_elo,
                "result_score": result,
            }
        )

    return pd.DataFrame(rows).sort_values("date_utc", ascending=False)


def _result_from_goals(scored: float, conceded: float) -> float:
    if scored > conceded:
        return 1.0
    if scored == conceded:
        return 0.5
    return 0.0


def _recency_weights(n: int, half_life_matches: float = 5.0) -> np.ndarray:
    if n <= 0:
        return np.array([])
    ranks = np.arange(n)
    weights = np.exp(-math.log(2) * ranks / half_life_matches)
    return weights / weights.sum()


def _opponent_factor(values: pd.Series, mode: Literal["attack", "defense"]) -> np.ndarray:
    """Return a mild opponent-strength factor around 1.0.

    For attack, scoring against stronger opponents should count slightly more.
    For defense, conceding against stronger opponents should be penalized less.
    """
    elo = pd.to_numeric(values, errors="coerce").fillna(1750.0)
    raw = (elo - 1750.0) / 500.0
    if mode == "attack":
        factor = 1.0 + raw * 0.18
    else:
        factor = 1.0 - raw * 0.18
    return np.clip(factor.to_numpy(dtype=float), 0.85, 1.15)


def calculate_attack_strength(matches_df: pd.DataFrame, team: str) -> float:
    """Estimate attack strength on a 0-1 scale from recent international matches.

    Formula:
    - collect the team's recent matches;
    - weight newer matches more heavily with exponential decay;
    - calculate weighted goals scored per match;
    - give a small adjustment for opponent Elo if available;
    - scale roughly around 1.55 goals per match and clamp to 0.20-0.95.

    If match history is missing, this falls back to the team's manual
    `attack` value in `data/team_ratings.csv`.
    """
    view = _team_match_view(matches_df, team)
    if view.empty:
        return _fallback_team_value(team, "attack", 0.55)

    weights = _recency_weights(len(view))
    adjusted_goals = view["goals_for"].to_numpy(dtype=float) * _opponent_factor(view["opponent_elo"], "attack")
    weighted_gpm = float(np.sum(adjusted_goals * weights))
    score = 0.22 + (weighted_gpm / 2.7) * 0.73
    return clamp(score, 0.20, 0.95)


def calculate_defense_strength(matches_df: pd.DataFrame, team: str) -> float:
    """Estimate defense strength on a 0-1 scale from recent international matches.

    Formula:
    - collect the team's recent matches;
    - weight newer matches more heavily with exponential decay;
    - calculate weighted goals conceded per match;
    - adjust gently for opponent Elo if available;
    - invert the conceded-goals rate so higher is better.

    If match history is missing, this falls back to the team's manual
    `defense` value in `data/team_ratings.csv`.
    """
    view = _team_match_view(matches_df, team)
    if view.empty:
        return _fallback_team_value(team, "defense", 0.55)

    weights = _recency_weights(len(view))
    adjusted_conceded = view["goals_against"].to_numpy(dtype=float) * _opponent_factor(view["opponent_elo"], "defense")
    weighted_gapm = float(np.sum(adjusted_conceded * weights))
    score = 0.95 - (weighted_gapm / 2.8) * 0.73
    return clamp(score, 0.20, 0.95)


def calculate_recent_form(matches_df: pd.DataFrame, team: str, n_matches: int = 10) -> float:
    """Return recency-weighted form where win=1.0, draw=0.5, loss=0.0."""
    view = _team_match_view(matches_df, team)
    if view.empty:
        return _fallback_team_value(team, "recent_form", 0.55)

    view = view.head(n_matches)
    weights = _recency_weights(len(view))
    return clamp(float(np.sum(view["result_score"].to_numpy(dtype=float) * weights)), 0.0, 1.0)


def load_recent_matches(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        cached = read_dataframe_cache("recent_matches_latest.csv", max_age_hours=72)
        if cached is not None and not cached.empty:
            return set_source_attrs(cached, SOURCE_CACHE, "data/cache/recent_matches_latest.csv", cached.attrs.get("last_updated"))
        path = DATA_DIR / "recent_matches.csv"

    matches = read_csv_with_columns(path, MATCH_COLUMNS)
    return set_source_attrs(matches, SOURCE_LOCAL, str(path.relative_to(DATA_DIR.parent)) if path.exists() else str(path))
