from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.cache import utc_now_iso
from src.historical_data import HISTORICAL_MATCH_COLUMNS
from src.utils import clamp, coerce_bool, coerce_float


@dataclass(frozen=True)
class EloConfig:
    default_elo: float = 1500.0
    home_advantage: float = 35.0
    max_elo_change: float = 35.0
    min_elo: float = 900.0
    max_elo: float = 2300.0
    world_cup_k: float = 45.0
    qualifier_k: float = 35.0
    continental_k: float = 35.0
    nations_league_k: float = 25.0
    friendly_k: float = 15.0
    other_k: float = 20.0


DEFAULT_ELO_CONFIG = EloConfig()


ELO_OUTPUT_COLUMNS = [
    "match_id",
    "date_utc",
    "team",
    "opponent",
    "team_goals",
    "opponent_goals",
    "team_elo_pre",
    "opponent_elo",
    "team_elo_post",
    "opponent_elo_post",
    "elo_change",
    "competition_type",
    "source",
]


def calculate_rolling_elo(matches_df: pd.DataFrame | None, config: EloConfig | None = None) -> pd.DataFrame:
    """Calculate chronological pre/post Elo for long-format team-match rows."""
    cfg = config or DEFAULT_ELO_CONFIG
    matches = _one_row_per_match(matches_df)
    if matches.empty:
        return pd.DataFrame(columns=ELO_OUTPUT_COLUMNS)

    ratings: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for _, match in matches.sort_values(["date_utc", "match_key"]).iterrows():
        home = str(match["home_team"])
        away = str(match["away_team"])
        home_pre = ratings.get(home, cfg.default_elo)
        away_pre = ratings.get(away, cfg.default_elo)
        home_goals = coerce_float(match["home_goals"])
        away_goals = coerce_float(match["away_goals"])
        home_score = _result_score(home_goals, away_goals)
        is_neutral = coerce_bool(match.get("is_neutral", False))
        expected_home = expected_result(home_pre, away_pre, is_neutral=is_neutral, config=cfg)
        goal_difference = home_goals - away_goals
        k_factor = elo_k_factor(str(match.get("competition_type", "other")), cfg)
        mov_multiplier = max(1.0, math.log(abs(goal_difference) + 1.0))
        elo_delta = clamp(
            k_factor * mov_multiplier * (home_score - expected_home),
            -cfg.max_elo_change,
            cfg.max_elo_change,
        )

        home_post = clamp(home_pre + elo_delta, cfg.min_elo, cfg.max_elo)
        away_post = clamp(away_pre - elo_delta, cfg.min_elo, cfg.max_elo)
        ratings[home] = home_post
        ratings[away] = away_post

        common = {
            "match_id": match.get("match_id", ""),
            "date_utc": pd.Timestamp(match["date_utc"]).date().isoformat(),
            "competition_type": match.get("competition_type", "other"),
            "source": match.get("source", ""),
        }
        rows.append(
            {
                **common,
                "team": home,
                "opponent": away,
                "team_goals": home_goals,
                "opponent_goals": away_goals,
                "team_elo_pre": round(home_pre, 3),
                "opponent_elo": round(away_pre, 3),
                "team_elo_post": round(home_post, 3),
                "opponent_elo_post": round(away_post, 3),
                "elo_change": round(home_post - home_pre, 3),
            }
        )
        rows.append(
            {
                **common,
                "team": away,
                "opponent": home,
                "team_goals": away_goals,
                "opponent_goals": home_goals,
                "team_elo_pre": round(away_pre, 3),
                "opponent_elo": round(home_pre, 3),
                "team_elo_post": round(away_post, 3),
                "opponent_elo_post": round(home_post, 3),
                "elo_change": round(away_post - away_pre, 3),
            }
        )

    return pd.DataFrame(rows, columns=ELO_OUTPUT_COLUMNS)


def add_elo_to_historical_matches(historical_matches_df: pd.DataFrame | None, config: EloConfig | None = None) -> pd.DataFrame:
    """Fill team_elo_pre and opponent_elo on the historical long-format table."""
    if historical_matches_df is None or historical_matches_df.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)

    out = historical_matches_df.copy()
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    elo = calculate_rolling_elo(out, config)
    if elo.empty:
        return out[HISTORICAL_MATCH_COLUMNS].copy()

    out["_elo_key"] = out.apply(_long_row_key, axis=1)
    elo_lookup = elo.copy()
    elo_lookup["_elo_key"] = elo_lookup.apply(_long_row_key, axis=1)
    elo_lookup = elo_lookup.drop_duplicates("_elo_key", keep="last").set_index("_elo_key")

    matched = out["_elo_key"].isin(elo_lookup.index)
    out.loc[matched, "team_elo_pre"] = out.loc[matched, "_elo_key"].map(elo_lookup["team_elo_pre"])
    out.loc[matched, "opponent_elo"] = out.loc[matched, "_elo_key"].map(elo_lookup["opponent_elo"])
    if "last_updated" in out.columns:
        out.loc[matched, "last_updated"] = out.loc[matched, "last_updated"].fillna(utc_now_iso())
    out = out.drop(columns=["_elo_key"])
    return out[HISTORICAL_MATCH_COLUMNS].copy()


def expected_result(home_elo: float, away_elo: float, is_neutral: bool = False, config: EloConfig | None = None) -> float:
    cfg = config or DEFAULT_ELO_CONFIG
    home_rating = float(home_elo) + (0.0 if is_neutral else cfg.home_advantage)
    return 1.0 / (1.0 + 10.0 ** ((float(away_elo) - home_rating) / 400.0))


def elo_k_factor(competition_type: str, config: EloConfig | None = None) -> float:
    cfg = config or DEFAULT_ELO_CONFIG
    key = str(competition_type or "other").strip().lower().replace("-", "_")
    key = " ".join(key.replace("_", " ").split())
    if key in {"world cup", "world_cup", "fifa world cup"}:
        return cfg.world_cup_k
    if key in {"world cup qualifier", "world_cup_qualifier", "world cup qualification", "qualifier"}:
        return cfg.qualifier_k
    if key in {"continental tournament", "continental_tournament", "euro", "copa america", "afcon", "asian cup", "gold cup"}:
        return cfg.continental_k
    if key in {"nations league", "nations_league"}:
        return cfg.nations_league_k
    if key == "friendly":
        return cfg.friendly_k
    return cfg.other_k


def _one_row_per_match(matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame()
    df = matches_df.copy()
    for col in [
        "match_id",
        "date_utc",
        "competition_type",
        "team",
        "opponent",
        "team_goals",
        "opponent_goals",
        "is_home",
        "is_neutral",
        "source",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    df["team_goals"] = pd.to_numeric(df["team_goals"], errors="coerce")
    df["opponent_goals"] = pd.to_numeric(df["opponent_goals"], errors="coerce")
    df = df.dropna(subset=["date_utc", "team", "opponent", "team_goals", "opponent_goals"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["_match_key"] = df.apply(_match_group_key, axis=1)
    rows = []
    for match_key, group in df.groupby("_match_key", sort=False):
        match = _match_from_group(match_key, group)
        if match:
            rows.append(match)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _match_from_group(match_key: str, group: pd.DataFrame) -> dict[str, Any] | None:
    if group.empty:
        return None
    home_rows = group.loc[group["is_home"].map(coerce_bool)]
    row = home_rows.iloc[0] if not home_rows.empty else group.sort_values(["team", "opponent"]).iloc[0]
    team = str(row["team"]).strip()
    opponent = str(row["opponent"]).strip()
    if not team or not opponent or team == opponent:
        return None
    return {
        "match_key": match_key,
        "match_id": str(row.get("match_id", "") or ""),
        "date_utc": row["date_utc"],
        "home_team": team,
        "away_team": opponent,
        "home_goals": row["team_goals"],
        "away_goals": row["opponent_goals"],
        "competition_type": str(row.get("competition_type", "other") or "other"),
        "is_neutral": row.get("is_neutral", False),
        "source": str(row.get("source", "") or ""),
    }


def _match_group_key(row: pd.Series) -> str:
    match_id = str(row.get("match_id", "") or "").strip()
    if match_id:
        return f"id:{match_id}"
    date_label = _date_key(row.get("date_utc"))
    teams = sorted([str(row.get("team", "") or "").strip(), str(row.get("opponent", "") or "").strip()])
    goals = sorted([str(int(coerce_float(row.get("team_goals")))), str(int(coerce_float(row.get("opponent_goals"))))])
    return f"fallback:{date_label}:{teams[0]}:{teams[1]}:{goals[0]}:{goals[1]}"


def _long_row_key(row: pd.Series) -> str:
    match_id = str(row.get("match_id", "") or "").strip()
    team = str(row.get("team", "") or "").strip()
    opponent = str(row.get("opponent", "") or "").strip()
    if match_id:
        return f"id:{match_id}:{team}:{opponent}"
    return (
        f"fallback:{_date_key(row.get('date_utc'))}:{team}:{opponent}:"
        f"{int(coerce_float(row.get('team_goals')))}:{int(coerce_float(row.get('opponent_goals')))}"
    )


def _date_key(value: Any) -> str:
    date_value = pd.Timestamp(value)
    if pd.isna(date_value):
        return ""
    return date_value.date().isoformat()


def _result_score(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0
