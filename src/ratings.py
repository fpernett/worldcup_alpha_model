from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from src.behavior_calibration import DEFAULT_BEHAVIOR_CONFIG
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
from src.team_behavior import load_team_behavior
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

ATTACK_DELTA_CAP = 0.12
DEFENSE_DELTA_CAP = 0.12
RECENT_FORM_DELTA_CAP = 0.18


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
                _apply_behavior_blend(_normalise_ratings(cached, "ratings_cache")),
                SOURCE_CACHE,
                "data/cache/team_ratings_latest.csv",
                cache_last_updated("team_ratings_latest.csv"),
            )

    if cfg.ratings_configured:
        api_df, error = _fetch_ratings_from_api()
        if api_df is not None and not api_df.empty:
            ratings = _normalise_ratings(api_df, "ratings_api")
            write_dataframe_cache(ratings, "team_ratings_latest.csv", "ratings API")
            ratings = _apply_behavior_blend(ratings)
            return set_source_attrs(ratings, SOURCE_API, "ratings API", utc_now_iso())
        warning = error or "Ratings API returned no usable rows."

        cached = read_dataframe_cache("team_ratings_latest.csv", max_age_hours=None)
        if cached is not None and not cached.empty:
            return set_source_attrs(
                _apply_behavior_blend(_normalise_ratings(cached, "ratings_cache")),
                SOURCE_CACHE,
                "data/cache/team_ratings_latest.csv",
                cache_last_updated("team_ratings_latest.csv"),
                warning=f"Ratings API failed; using cached ratings. {warning}",
            )

    local = _normalise_ratings(
        read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS),
        "manual_csv",
    )
    ratings = _apply_recent_match_features(local.copy())
    ratings = _apply_behavior_blend(ratings, manual_base=local)
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
            "manual_attack": 0.55,
            "manual_defense": 0.55,
            "manual_recent_form": 0.55,
            "behavior_attack_index": pd.NA,
            "behavior_defense_index": pd.NA,
            "behavior_recent_form_index": pd.NA,
            "behavior_overall_data_quality": "none",
            "behavior_n_matches": 0,
            "attack_source": "neutral_fallback",
            "defense_source": "neutral_fallback",
            "recent_form_source": "neutral_fallback",
            "behavior_blend_used": False,
            "behavior_blend_cap_hit": False,
            "behavior_attack_delta": 0.0,
            "behavior_defense_delta": 0.0,
            "behavior_recent_form_delta": 0.0,
            "behavior_attack_cap_hit": False,
            "behavior_defense_cap_hit": False,
            "behavior_recent_form_cap_hit": False,
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


def _apply_behavior_blend(ratings: pd.DataFrame, manual_base: pd.DataFrame | None = None) -> pd.DataFrame:
    """Blend optional historical behavior metrics into transparent rating inputs.

    Manual/API rating rows stay as the base. Behavior rows are only allowed to
    move model inputs within quality-dependent blend weights and hard delta caps.
    """
    out = ratings.copy()
    if out.empty:
        return out

    base = manual_base.copy() if manual_base is not None else ratings.copy()
    behavior = load_team_behavior()
    out = _append_fixture_behavior_neutral_rows(out, behavior)
    base = _append_fixture_behavior_neutral_rows(base, behavior)
    for col in [
        "manual_attack",
        "manual_defense",
        "manual_recent_form",
        "behavior_attack_index",
        "behavior_defense_index",
        "behavior_recent_form_index",
        "behavior_overall_data_quality",
        "behavior_n_matches",
        "attack_source",
        "defense_source",
        "recent_form_source",
        "behavior_blend_used",
        "behavior_blend_cap_hit",
        "behavior_attack_delta",
        "behavior_defense_delta",
        "behavior_recent_form_delta",
        "behavior_attack_cap_hit",
        "behavior_defense_cap_hit",
        "behavior_recent_form_cap_hit",
    ]:
        if col not in out.columns:
            out[col] = pd.NA

    out["behavior_blend_used"] = False
    out["behavior_blend_cap_hit"] = False
    for idx, row in out.iterrows():
        base_row = _matching_base_row(base, row["team"])
        manual_attack = coerce_float(base_row.get("attack", row["attack"]), coerce_float(row["attack"], 0.55))
        manual_defense = coerce_float(base_row.get("defense", row["defense"]), coerce_float(row["defense"], 0.55))
        manual_form = coerce_float(base_row.get("recent_form", row["recent_form"]), coerce_float(row["recent_form"], 0.55))
        source = _rating_source_label(row)
        out.at[idx, "manual_attack"] = round(manual_attack, 3)
        out.at[idx, "manual_defense"] = round(manual_defense, 3)
        out.at[idx, "manual_recent_form"] = round(manual_form, 3)
        out.at[idx, "attack_source"] = source
        out.at[idx, "defense_source"] = source
        out.at[idx, "recent_form_source"] = source
        out.at[idx, "behavior_overall_data_quality"] = "none"
        out.at[idx, "behavior_n_matches"] = 0
        out.at[idx, "behavior_attack_delta"] = 0.0
        out.at[idx, "behavior_defense_delta"] = 0.0
        out.at[idx, "behavior_recent_form_delta"] = 0.0
        out.at[idx, "behavior_attack_cap_hit"] = False
        out.at[idx, "behavior_defense_cap_hit"] = False
        out.at[idx, "behavior_recent_form_cap_hit"] = False

    if behavior.empty or "team" not in behavior.columns:
        return out

    for idx, row in out.iterrows():
        behavior_row = _matching_base_row(behavior, row["team"])
        if behavior_row.empty:
            continue

        quality = str(behavior_row.get("overall_data_quality", "none")).lower()
        n_matches = coerce_float(behavior_row.get("n_matches"), float("nan"))
        if pd.isna(n_matches):
            n_matches = coerce_float(behavior_row.get("matches_used_recent"), 0.0)
        attack_index = coerce_float(behavior_row.get("attack_index"), float("nan"))
        defense_index = coerce_float(behavior_row.get("defense_index"), float("nan"))
        form_index = coerce_float(behavior_row.get("recent_form_index"), float("nan"))
        out.at[idx, "behavior_attack_index"] = round(attack_index, 3) if not pd.isna(attack_index) else pd.NA
        out.at[idx, "behavior_defense_index"] = round(defense_index, 3) if not pd.isna(defense_index) else pd.NA
        out.at[idx, "behavior_recent_form_index"] = round(form_index, 3) if not pd.isna(form_index) else pd.NA
        out.at[idx, "behavior_overall_data_quality"] = quality or "none"
        out.at[idx, "behavior_n_matches"] = int(n_matches)

        manual_attack = coerce_float(out.at[idx, "manual_attack"], coerce_float(row["attack"], 0.55))
        manual_defense = coerce_float(out.at[idx, "manual_defense"], coerce_float(row["defense"], 0.55))
        manual_form = coerce_float(out.at[idx, "manual_recent_form"], coerce_float(row["recent_form"], 0.55))

        blend_weights = _behavior_blend_weights(behavior_row)
        if blend_weights is None:
            out.at[idx, "attack"] = round(clamp(manual_attack, 0.0, 1.0), 3)
            out.at[idx, "defense"] = round(clamp(manual_defense, 0.0, 1.0), 3)
            out.at[idx, "recent_form"] = round(clamp(manual_form, 0.0, 1.0), 3)
            out.at[idx, "attack_source"] = "manual_base_insufficient_behavior"
            out.at[idx, "defense_source"] = "manual_base_insufficient_behavior"
            out.at[idx, "recent_form_source"] = "manual_base_insufficient_behavior"
            continue

        attack, attack_delta, attack_cap_hit = _capped_behavior_value(
            manual_attack,
            attack_index,
            blend_weights["attack"],
            ATTACK_DELTA_CAP,
        )
        defense, defense_delta, defense_cap_hit = _capped_behavior_value(
            manual_defense,
            defense_index,
            blend_weights["defense"],
            DEFENSE_DELTA_CAP,
        )
        form, form_delta, form_cap_hit = _capped_behavior_value(
            manual_form,
            form_index,
            blend_weights["recent_form"],
            RECENT_FORM_DELTA_CAP,
        )
        any_cap_hit = attack_cap_hit or defense_cap_hit or form_cap_hit
        source_suffix = "_capped" if any_cap_hit else ""
        quality_suffix = "_low_weight" if quality == "low" else ""

        out.at[idx, "attack"] = round(attack, 3)
        out.at[idx, "defense"] = round(defense, 3)
        out.at[idx, "recent_form"] = round(form, 3)
        out.at[idx, "attack_source"] = f"manual_base+historical_behavior{quality_suffix}{source_suffix}"
        out.at[idx, "defense_source"] = f"manual_base+historical_behavior{quality_suffix}{source_suffix}"
        out.at[idx, "recent_form_source"] = f"manual_base+historical_behavior{quality_suffix}{source_suffix}"
        out.at[idx, "behavior_blend_used"] = True
        out.at[idx, "behavior_blend_cap_hit"] = any_cap_hit
        out.at[idx, "behavior_attack_delta"] = round(attack_delta, 3)
        out.at[idx, "behavior_defense_delta"] = round(defense_delta, 3)
        out.at[idx, "behavior_recent_form_delta"] = round(form_delta, 3)
        out.at[idx, "behavior_attack_cap_hit"] = attack_cap_hit
        out.at[idx, "behavior_defense_cap_hit"] = defense_cap_hit
        out.at[idx, "behavior_recent_form_cap_hit"] = form_cap_hit
        quality_text = str(out.at[idx, "data_quality"])
        if "historical_behavior" not in quality_text:
            out.at[idx, "data_quality"] = f"{quality_text}+historical_behavior"
        notes = str(out.at[idx, "notes"])
        if "Historical behavior blend used." not in notes:
            out.at[idx, "notes"] = f"{notes} Historical behavior blend used.".strip()

    return out


def _append_fixture_behavior_neutral_rows(ratings: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """Add neutral-base rows for fixture teams that have behavior but no manual row."""
    if ratings is None or ratings.empty or behavior is None or behavior.empty or "team" not in behavior.columns:
        return ratings

    fixture_teams = _fixture_team_names()
    if not fixture_teams:
        return ratings

    existing = set(ratings["team"].dropna().astype(str).str.lower()) if "team" in ratings.columns else set()
    behavior_teams = set(behavior["team"].dropna().astype(str))
    rows = []
    for team in sorted(fixture_teams):
        if team.lower() in existing or team not in behavior_teams:
            continue
        rows.append(
            {
                "team": team,
                "elo": 1700.0,
                "attack": 0.55,
                "defense": 0.55,
                "recent_form": 0.55,
                "fifa_rank_proxy": 70,
                "training_temp_c": 20.0,
                "training_humidity_pct": 60.0,
                "data_quality": "neutral_fixture_base",
                "last_updated": today_iso(),
                "notes": "Neutral fixture base because this team is missing from team_ratings.csv; historical behavior may blend conservatively.",
            }
        )
    if not rows:
        return ratings
    out = pd.concat([ratings, pd.DataFrame(rows)], ignore_index=True)
    return out.sort_values("team").reset_index(drop=True)


def _fixture_team_names() -> set[str]:
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    if fixtures.empty:
        return set()
    teams: set[str] = set()
    for col in ["home", "away"]:
        if col in fixtures.columns:
            teams.update(team for team in fixtures[col].dropna().astype(str).str.strip() if team)
    return teams


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


def _matching_base_row(df: pd.DataFrame, team: str) -> pd.Series:
    if df is None or df.empty or "team" not in df.columns:
        return pd.Series(dtype="object")
    row = df.loc[df["team"].astype(str).str.lower() == str(team).lower()]
    if row.empty:
        return pd.Series(dtype="object")
    return row.iloc[0]


def _rating_source_label(row: pd.Series) -> str:
    quality = str(row.get("data_quality", "") or "")
    if "recent_match_history" in quality:
        return "recent_match_history"
    if "api" in quality:
        return "ratings_api"
    if "cache" in quality:
        return "ratings_cache"
    if "manual" in quality:
        return "manual_csv"
    return quality or "rating_input"


def _behavior_blend_weights(row: pd.Series) -> dict[str, float] | None:
    quality = str(row.get("overall_data_quality", "")).lower()
    if quality not in {"high", "moderate", "low"}:
        return None
    n_matches = coerce_float(row.get("n_matches"), float("nan"))
    if pd.isna(n_matches):
        n_matches = coerce_float(row.get("matches_used_recent"), 0.0)
    if n_matches < 8:
        return None
    for metric in ["attack_index", "defense_index", "recent_form_index"]:
        value = coerce_float(row.get(metric), float("nan"))
        if pd.isna(value):
            return None

    if quality == "low":
        weights = {
            "attack": min(0.15, DEFAULT_BEHAVIOR_CONFIG.max_behavior_blend / 2.0),
            "defense": min(0.15, DEFAULT_BEHAVIOR_CONFIG.max_behavior_blend / 2.0),
            "recent_form": min(0.25, DEFAULT_BEHAVIOR_CONFIG.max_form_blend / 2.0),
        }
    else:
        weights = {
            "attack": DEFAULT_BEHAVIOR_CONFIG.max_behavior_blend,
            "defense": DEFAULT_BEHAVIOR_CONFIG.max_behavior_blend,
            "recent_form": DEFAULT_BEHAVIOR_CONFIG.max_form_blend,
        }

    coverage = coerce_float(row.get("opponent_elo_coverage_recent"), 1.0)
    residual_coverage = coerce_float(row.get("residual_coverage_recent"), 1.0)
    schedule_label = str(row.get("schedule_strength_label", "") or "").lower()
    if (
        schedule_label == "unknown"
        or coverage < DEFAULT_BEHAVIOR_CONFIG.min_opponent_elo_coverage
        or residual_coverage < DEFAULT_BEHAVIOR_CONFIG.min_residual_coverage
    ):
        return None
    attack_raw = coerce_float(row.get("attack_index_raw"), float("nan"))
    attack_adjusted = coerce_float(row.get("attack_index_adjusted_old", row.get("attack_index_adjusted")), float("nan"))
    if schedule_label == "weak" and not pd.isna(attack_raw) and not pd.isna(attack_adjusted) and attack_adjusted < attack_raw:
        weights["attack"] *= 0.50
    residual_warning = str(row.get("residual_warning", "") or "").strip()
    if residual_warning:
        weights = {key: value * 0.50 for key, value in weights.items()}
    return weights


def _capped_behavior_value(manual_value: float, behavior_value: float, blend_weight: float, delta_cap: float) -> tuple[float, float, bool]:
    manual = clamp(manual_value, 0.0, 1.0)
    behavior = clamp(behavior_value, 0.0, 1.0)
    desired_delta = float(blend_weight) * (behavior - manual)
    capped_delta = clamp(desired_delta, -float(delta_cap), float(delta_cap))
    return clamp(manual + capped_delta, 0.0, 1.0), capped_delta, abs(desired_delta) > float(delta_cap) + 1e-12


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
