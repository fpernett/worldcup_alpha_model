from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from src.cache import (
    cache_last_updated,
    read_dataframe_cache,
    set_source_attrs,
    utc_now_iso,
    write_dataframe_cache,
    write_json_cache,
)
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_LOCAL, get_config
from src.feature_engineering import (
    calculate_attack_strength,
    calculate_defense_strength,
    calculate_recent_form,
    load_recent_matches,
)
from src.utils import clamp, coerce_float, read_csv_with_columns, today_iso


TEAM_RATING_COLUMNS = [
    "team",
    "elo",
    "attack",
    "defense",
    "recent_form",
    "fifa_rank_proxy",
    "training_temp_c",
    "training_humidity_pct",
    "data_quality",
    "last_updated",
    "notes",
]


def get_team_ratings(force_refresh: bool = False) -> pd.DataFrame:
    """Return one row per team with transparent strength inputs.

    Hierarchy:
    1. ratings API if `RATINGS_API_URL` is configured and succeeds;
    2. cached ratings if present;
    3. manual `data/team_ratings.csv` base plus cached/local historical results;
    4. FIFA-rank proxy defaults for missing values.
    """
    cfg = get_config()
    warning = ""

    if cfg.ratings_configured and not force_refresh:
        cached = read_dataframe_cache("team_ratings_latest.csv", max_age_hours=12)
        if cached is not None and not cached.empty:
            return set_source_attrs(
                _normalise_ratings(cached, "ratings_cache"),
                SOURCE_CACHE,
                "data/cache/team_ratings_latest.csv",
                cache_last_updated("team_ratings_latest.csv"),
            )

    if cfg.ratings_configured:
        api_df, error = _fetch_ratings_from_api()
        if api_df is not None and not api_df.empty:
            ratings = _normalise_ratings(api_df, "ratings_api")
            write_dataframe_cache(ratings, "team_ratings_latest.csv", "ratings API")
            return set_source_attrs(ratings, SOURCE_API, "ratings API", utc_now_iso())
        warning = error or "Ratings API returned no usable rows."

        cached = read_dataframe_cache("team_ratings_latest.csv", max_age_hours=None)
        if cached is not None and not cached.empty:
            return set_source_attrs(
                _normalise_ratings(cached, "ratings_cache"),
                SOURCE_CACHE,
                "data/cache/team_ratings_latest.csv",
                cache_last_updated("team_ratings_latest.csv"),
                warning=f"Ratings API failed; using cached ratings. {warning}",
            )

    local = _normalise_ratings(
        read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS),
        "manual_csv",
    )
    ratings = _apply_recent_match_features(local)
    source_label = SOURCE_LOCAL
    source_detail = "data/team_ratings.csv"
    matches_source = ratings.attrs.get("match_source_label")
    if matches_source == SOURCE_CACHE:
        source_label = SOURCE_CACHE
        source_detail = "data/team_ratings.csv + data/cache/recent_matches_latest.csv"

    return set_source_attrs(
        ratings,
        source_label,
        source_detail,
        _csv_last_modified("team_ratings.csv"),
        warning=f"Ratings API unavailable; using fallback. {warning}" if warning else None,
    )


def neutral_team_rating(team: str) -> pd.Series:
    return pd.Series(
        {
            "team": team,
            "elo": 1700.0,
            "attack": 0.55,
            "defense": 0.55,
            "recent_form": 0.55,
            "fifa_rank_proxy": 70,
            "training_temp_c": 20.0,
            "training_humidity_pct": 60.0,
            "data_quality": "neutral_fallback",
            "last_updated": utc_now_iso(),
            "notes": "Neutral fallback because this team was missing from team_ratings.csv.",
        }
    )


def _fetch_ratings_from_api() -> tuple[pd.DataFrame | None, str | None]:
    cfg = get_config()
    if not cfg.ratings_api_url:
        return None, "RATINGS_API_URL is not configured."

    try:
        response = requests.get(cfg.ratings_api_url, timeout=20)
        response.raise_for_status()
        payload: Any = response.json()
    except Exception as exc:
        return None, str(exc)

    write_json_cache(payload, "ratings_raw_latest.json", "ratings API")
    if isinstance(payload, dict):
        payload = payload.get("ratings") or payload.get("teams") or payload.get("data") or []
    if not isinstance(payload, list):
        return None, "Ratings API payload did not contain a list."
    df = pd.json_normalize(payload)
    if df.empty:
        return None, "Ratings API payload was empty."
    return df, None


def _apply_recent_match_features(ratings: pd.DataFrame) -> pd.DataFrame:
    matches = load_recent_matches()
    if matches.empty:
        return ratings

    elo_by_team = _calculate_elo_from_results(matches)
    out = ratings.copy()
    for idx, row in out.iterrows():
        team = row["team"]
        if team in elo_by_team:
            out.at[idx, "elo"] = round(elo_by_team[team], 1)
        out.at[idx, "attack"] = round(calculate_attack_strength(matches, team), 3)
        out.at[idx, "defense"] = round(calculate_defense_strength(matches, team), 3)
        out.at[idx, "recent_form"] = round(calculate_recent_form(matches, team), 3)
        existing_quality = str(out.at[idx, "data_quality"])
        if "recent_match_history" not in existing_quality:
            out.at[idx, "data_quality"] = f"{existing_quality}+recent_match_history"
        out.at[idx, "last_updated"] = today_iso()

    out.attrs["match_source_label"] = matches.attrs.get("source_label", "")
    out.attrs["match_source_detail"] = matches.attrs.get("source_detail", "")
    return out


def _calculate_elo_from_results(matches: pd.DataFrame, initial: float = 1700.0, k_factor: float = 24.0) -> dict[str, float]:
    if matches.empty:
        return {}
    df = matches.copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    df = df.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"]).sort_values("date_utc")
    ratings: dict[str, float] = {}

    for _, row in df.iterrows():
        home = str(row["home"])
        away = str(row["away"])
        home_rating = ratings.get(home, initial)
        away_rating = ratings.get(away, initial)
        expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))
        score_home = _match_result_score(float(row["home_goals"]), float(row["away_goals"]))
        margin = abs(float(row["home_goals"]) - float(row["away_goals"]))
        margin_multiplier = 1.0 + min(margin, 4.0) * 0.08
        delta = k_factor * margin_multiplier * (score_home - expected_home)
        ratings[home] = clamp(home_rating + delta, 1200.0, 2200.0)
        ratings[away] = clamp(away_rating - delta, 1200.0, 2200.0)

    return ratings


def _match_result_score(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def _normalise_ratings(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "name": "team",
        "rating": "elo",
        "rank": "fifa_rank_proxy",
        "fifa_rank": "fifa_rank_proxy",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    for col in TEAM_RATING_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[TEAM_RATING_COLUMNS].copy()
    out = out.dropna(subset=["team"])
    out["team"] = out["team"].astype(str).str.strip()
    out = out[out["team"] != ""]

    for idx, row in out.iterrows():
        rank = coerce_float(row["fifa_rank_proxy"], 80.0)
        elo = coerce_float(row["elo"], _derive_elo_from_rank(rank))
        rank_based = _rank_score(rank)
        attack = coerce_float(row["attack"], rank_based)
        defense = coerce_float(row["defense"], rank_based)
        form = coerce_float(row["recent_form"], rank_based * 0.85)

        out.at[idx, "fifa_rank_proxy"] = int(round(rank))
        out.at[idx, "elo"] = round(clamp(elo, 1200.0, 2200.0), 1)
        out.at[idx, "attack"] = round(clamp(attack, 0.0, 1.0), 3)
        out.at[idx, "defense"] = round(clamp(defense, 0.0, 1.0), 3)
        out.at[idx, "recent_form"] = round(clamp(form, 0.0, 1.0), 3)
        out.at[idx, "training_temp_c"] = coerce_float(row["training_temp_c"], 20.0)
        out.at[idx, "training_humidity_pct"] = coerce_float(row["training_humidity_pct"], 60.0)
        out.at[idx, "data_quality"] = row["data_quality"] if pd.notna(row["data_quality"]) else source_label
        out.at[idx, "last_updated"] = row["last_updated"] if pd.notna(row["last_updated"]) else today_iso()
        out.at[idx, "notes"] = row["notes"] if pd.notna(row["notes"]) else "Transparent rating approximation."

    return out.sort_values("team").reset_index(drop=True)


def _derive_elo_from_rank(rank: float) -> float:
    rank = clamp(rank, 1.0, 150.0)
    return clamp(2110.0 - (rank - 1.0) * 5.3, 1320.0, 2110.0)


def _rank_score(rank: float) -> float:
    rank = clamp(rank, 1.0, 150.0)
    return clamp(0.92 - (rank - 1.0) * 0.0048, 0.24, 0.92)


def _csv_last_modified(filename: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
