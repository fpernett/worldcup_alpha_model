from __future__ import annotations

import difflib
from typing import Any

import pandas as pd

from src.historical_data import HISTORICAL_MATCH_COLUMNS
from src.team_names import load_team_name_aliases_df, normalize_team_name, team_name_key


COMMON_CANONICAL_NAMES = {
    "jordan": "Jordan",
    "algeria": "Algeria",
    "united states": "United States",
    "usa": "United States",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "cote d'ivoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "côte d’ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "cdr": "DR Congo",
    "cod": "DR Congo",
    "drc": "DR Congo",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
}


def canonicalize_team_for_data_sources(team: str, aliases_df: pd.DataFrame | None = None) -> str:
    """Canonicalize team names for fixture/history/rating joins."""
    raw = str(team or "").strip()
    if not raw:
        return ""

    alias_map = _alias_map(aliases_df)
    key = team_name_key(raw)
    if key in alias_map:
        return alias_map[key]
    if key in COMMON_CANONICAL_NAMES:
        return COMMON_CANONICAL_NAMES[key]

    normalized = normalize_team_name(raw)
    normalized_key = team_name_key(normalized)
    if normalized_key in alias_map:
        return alias_map[normalized_key]
    if normalized_key in COMMON_CANONICAL_NAMES:
        return COMMON_CANONICAL_NAMES[normalized_key]
    return normalized if normalized else raw


def suggest_close_team_names(team: str, available_teams: list[str], limit: int = 5) -> list[str]:
    keyed = {team_name_key(value): value for value in available_teams if str(value or "").strip()}
    matches = difflib.get_close_matches(team_name_key(team), list(keyed), n=limit, cutoff=0.55)
    return [keyed[key] for key in matches]


def audit_historical_file_status(historical_matches_df: pd.DataFrame | None) -> dict[str, Any]:
    df = historical_matches_df.copy() if historical_matches_df is not None else pd.DataFrame()
    missing = [col for col in HISTORICAL_MATCH_COLUMNS if col not in df.columns]
    if df.empty:
        return {
            "rows_loaded": 0,
            "unique_teams": 0,
            "earliest_date": "",
            "latest_date": "",
            "source_counts": {},
            "competition_type_counts": {},
            "missing_required_columns": missing,
            "warning": "No historical match rows loaded.",
        }

    dates = pd.to_datetime(df.get("date_utc"), utc=True, errors="coerce")
    source_counts = df.get("source", pd.Series(dtype="object")).fillna("").astype(str).value_counts().to_dict()
    competition_type_counts = (
        df.get("competition_type", pd.Series(dtype="object")).fillna("").astype(str).value_counts().to_dict()
    )
    warning = ""
    if missing:
        warning = "Historical matches missing required columns: " + ", ".join(missing)
    elif dates.notna().sum() == 0:
        warning = "Historical matches loaded, but no valid date_utc values were found."
    return {
        "rows_loaded": int(len(df)),
        "unique_teams": int(df.get("team", pd.Series(dtype="object")).dropna().astype(str).nunique()),
        "earliest_date": _date_label(dates.min()),
        "latest_date": _date_label(dates.max()),
        "source_counts": source_counts,
        "competition_type_counts": competition_type_counts,
        "missing_required_columns": missing,
        "warning": warning,
    }


def audit_historical_binding_for_match(
    home: str,
    away: str,
    historical_matches_df: pd.DataFrame,
    team_behavior_df: pd.DataFrame | None = None,
    team_ratings_df: pd.DataFrame | None = None,
    aliases_df: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    aliases = aliases_df if aliases_df is not None else load_team_name_aliases_df()
    home_canonical = canonicalize_team_for_data_sources(home, aliases)
    away_canonical = canonicalize_team_for_data_sources(away, aliases)
    matches = _canonicalized_historical(historical_matches_df, aliases)
    asof = pd.Timestamp(as_of_date, tz="UTC") if as_of_date else None
    if asof is not None and asof.tzinfo is None:
        asof = asof.tz_localize("UTC")

    home_rows = matches.loc[matches["_team_canonical"] == home_canonical]
    away_rows = matches.loc[matches["_team_canonical"] == away_canonical]
    if asof is not None:
        before_asof = matches.loc[matches["_date_utc"] < asof]
    else:
        before_asof = matches
    home_before = before_asof.loc[before_asof["_team_canonical"] == home_canonical]
    away_before = before_asof.loc[before_asof["_team_canonical"] == away_canonical]
    h2h = matches.loc[
        ((matches["_team_canonical"] == home_canonical) & (matches["_opponent_canonical"] == away_canonical))
        | ((matches["_team_canonical"] == away_canonical) & (matches["_opponent_canonical"] == home_canonical))
    ]
    h2h_before = before_asof.loc[
        ((before_asof["_team_canonical"] == home_canonical) & (before_asof["_opponent_canonical"] == away_canonical))
        | ((before_asof["_team_canonical"] == away_canonical) & (before_asof["_opponent_canonical"] == home_canonical))
    ]

    behavior = _canonical_team_set(team_behavior_df, aliases)
    ratings = _canonical_team_set(team_ratings_df, aliases)
    warnings = _binding_warnings(home_canonical, away_canonical, matches, home_rows, away_rows)
    status = "ok"
    if home_rows.empty and away_rows.empty:
        status = "missing"
    elif home_rows.empty or away_rows.empty:
        status = "partial"

    return {
        "home_input": home,
        "away_input": away,
        "home_canonical": home_canonical,
        "away_canonical": away_canonical,
        "home_historical_rows_total": int(len(home_rows)),
        "away_historical_rows_total": int(len(away_rows)),
        "home_historical_rows_before_asof": int(len(home_before)),
        "away_historical_rows_before_asof": int(len(away_before)),
        "h2h_rows_total": int(len(h2h)),
        "h2h_rows_before_asof": int(len(h2h_before)),
        "home_latest_match": _date_label(home_before["_date_utc"].max() if not home_before.empty else pd.NaT),
        "away_latest_match": _date_label(away_before["_date_utc"].max() if not away_before.empty else pd.NaT),
        "home_oldest_match": _date_label(home_before["_date_utc"].min() if not home_before.empty else pd.NaT),
        "away_oldest_match": _date_label(away_before["_date_utc"].min() if not away_before.empty else pd.NaT),
        "home_behavior_found": home_canonical in behavior,
        "away_behavior_found": away_canonical in behavior,
        "home_rating_found": home_canonical in ratings,
        "away_rating_found": away_canonical in ratings,
        "home_alias_used": team_name_key(home) != team_name_key(home_canonical),
        "away_alias_used": team_name_key(away) != team_name_key(away_canonical),
        "data_binding_status": status,
        "warning": "; ".join(warnings),
    }


def _alias_map(aliases_df: pd.DataFrame | None) -> dict[str, str]:
    aliases = load_team_name_aliases_df() if aliases_df is None else aliases_df.copy()
    mapping = {team_name_key(alias): canonical for alias, canonical in COMMON_CANONICAL_NAMES.items()}
    if aliases is not None and not aliases.empty:
        for _, row in aliases.iterrows():
            alias = str(row.get("alias", "") or "").strip()
            canonical = str(row.get("canonical", "") or "").strip()
            if alias and canonical:
                mapping[team_name_key(alias)] = canonical
                mapping.setdefault(team_name_key(canonical), canonical)
    return mapping


def _canonicalized_historical(df: pd.DataFrame | None, aliases_df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    else:
        out = df.copy()
    for col in ["date_utc", "team", "opponent"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["_date_utc"] = pd.to_datetime(out["date_utc"], utc=True, errors="coerce")
    canonical_map = _canonical_lookup_for_values(
        pd.concat([out["team"], out["opponent"]], ignore_index=True).dropna().astype(str).unique().tolist(),
        aliases_df,
    )
    out["_team_canonical"] = out["team"].astype(str).map(canonical_map).fillna(out["team"].astype(str))
    out["_opponent_canonical"] = out["opponent"].astype(str).map(canonical_map).fillna(out["opponent"].astype(str))
    return out.dropna(subset=["_date_utc"]).copy()


def _canonical_team_set(df: pd.DataFrame | None, aliases_df: pd.DataFrame) -> set[str]:
    if df is None or df.empty or "team" not in df.columns:
        return set()
    lookup = _canonical_lookup_for_values(df["team"].dropna().astype(str).unique().tolist(), aliases_df)
    return set(lookup.values())


def _canonical_lookup_for_values(values: list[str], aliases_df: pd.DataFrame) -> dict[str, str]:
    alias_map = _alias_map(aliases_df)
    lookup: dict[str, str] = {}
    for value in values:
        raw = str(value or "").strip()
        key = team_name_key(raw)
        lookup[raw] = alias_map.get(key) or COMMON_CANONICAL_NAMES.get(key) or normalize_team_name(raw) or raw
    return lookup


def _binding_warnings(
    home_canonical: str,
    away_canonical: str,
    matches: pd.DataFrame,
    home_rows: pd.DataFrame,
    away_rows: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []
    available = sorted(matches.get("team", pd.Series(dtype="object")).dropna().astype(str).unique().tolist())
    for team, rows in [(home_canonical, home_rows), (away_canonical, away_rows)]:
        if rows.empty:
            close = suggest_close_team_names(team, available)
            suffix = f" Available close matches: {', '.join(close)}." if close else ""
            warnings.append(f"No rows found for canonical team name '{team}' in data/historical_matches.csv.{suffix}")
    return warnings


def _date_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")
