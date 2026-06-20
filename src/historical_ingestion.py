from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.cache import set_source_attrs, utc_now_iso, write_dataframe_cache
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_LOCAL
from src.football_api import fetch_finished_matches
from src.historical_data import HISTORICAL_MATCH_COLUMNS, load_historical_matches, normalise_historical_matches
from src.team_behavior import rebuild_team_behavior_csv
from src.team_names import normalize_team_name
from src.utils import parse_date, read_csv_with_columns


def classify_competition_type(competition_name: str) -> str:
    name = str(competition_name or "").strip().lower()
    if not name:
        return "other"
    if "friendly" in name:
        return "friendly"
    if "nations league" in name:
        return "nations_league"
    if "world cup" in name and ("qualif" in name or "qualification" in name):
        return "world_cup_qualifier"
    if "fifa world cup" in name or name == "world cup":
        return "world_cup"
    continental_terms = [
        "european championship",
        "euro",
        "copa america",
        "africa cup",
        "afcon",
        "asian cup",
        "gold cup",
        "concacaf championship",
        "oceania nations cup",
    ]
    if any(term in name for term in continental_terms):
        return "continental_tournament"
    return "other"


def fetch_historical_matches_for_team(team: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch and normalize recent matches involving one team.

    The returned dataframe uses the long team-match historical schema. Missing
    API keys or empty API/cache responses return a safe empty dataframe.
    """
    raw_matches, source_label, warning = fetch_finished_matches(start_date, end_date)
    out = football_matches_to_historical_rows(raw_matches, teams=[team], source=_source_detail(source_label))
    return set_source_attrs(out, source_label, _source_detail(source_label), warning=warning or None)


def fetch_historical_matches_for_teams(
    teams: list[str],
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    raw_matches, source_label, warning = fetch_finished_matches(start_date, end_date, force_refresh=force_refresh)
    out = football_matches_to_historical_rows(raw_matches, teams=teams, source=_source_detail(source_label))
    return set_source_attrs(out, source_label, _source_detail(source_label), warning=warning or None)


def football_matches_to_historical_rows(
    matches: list[dict[str, Any]] | pd.DataFrame | None,
    teams: list[str] | None = None,
    source: str = "football-data.org",
) -> pd.DataFrame:
    """Convert football-data raw match payloads into two long rows per match."""
    raw_matches = _records(matches)
    target_teams = {normalize_team_name(team).lower() for team in (teams or []) if str(team).strip()}
    rows: list[dict[str, Any]] = []
    now = utc_now_iso()

    for match in raw_matches:
        row = _match_payload_to_result_row(match)
        if not row:
            continue
        home = str(row["home"])
        away = str(row["away"])
        if target_teams and home.lower() not in target_teams and away.lower() not in target_teams:
            continue
        rows.extend(_result_row_to_long_rows(row, source=source, last_updated=now))

    return normalise_historical_matches(pd.DataFrame(rows))


def merge_historical_matches(existing_df: pd.DataFrame | None, new_df: pd.DataFrame | None) -> pd.DataFrame:
    """Merge historical rows while preserving manual rows and deduping safely."""
    existing = normalise_historical_matches(existing_df)
    incoming = normalise_historical_matches(new_df)
    if existing.empty and incoming.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    if incoming.empty:
        return existing
    if existing.empty:
        return incoming

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined["__merge_key"] = combined.apply(_dedupe_key, axis=1)
    combined["__manual_priority"] = combined["source"].astype(str).str.lower().eq("manual").astype(int)
    combined["__last_updated_ts"] = pd.to_datetime(combined["last_updated"], errors="coerce", utc=True)
    combined["__last_updated_ts"] = combined["__last_updated_ts"].fillna(pd.Timestamp("1970-01-01", tz="UTC"))
    combined["__row_order"] = range(len(combined))
    combined = combined.sort_values(
        ["__merge_key", "__manual_priority", "__last_updated_ts", "__row_order"],
        ascending=[True, True, True, True],
    )
    deduped = combined.drop_duplicates(subset=["__merge_key"], keep="last")
    deduped = deduped.drop(columns=["__merge_key", "__manual_priority", "__last_updated_ts", "__row_order"])
    return normalise_historical_matches(deduped)


def update_historical_matches(
    start_date: str | date,
    end_date: str | date,
    teams: list[str] | None = None,
    rebuild_behavior: bool = True,
    existing_path: Path | None = None,
    output_path: Path | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch, merge, save historical matches, and optionally rebuild behavior."""
    start = parse_date(start_date).isoformat()
    end = parse_date(end_date).isoformat()
    team_list = teams or load_team_list_from_ratings()
    output = output_path or existing_path or (DATA_DIR / "historical_matches.csv")
    existing = load_historical_matches(path=existing_path or output, use_cache=False)
    fetched = fetch_historical_matches_for_teams(team_list, start, end, force_refresh=force_refresh)
    merged = merge_historical_matches(existing, fetched)
    merged.to_csv(output, index=False)
    write_dataframe_cache(merged, "historical_matches_latest.csv", "historical ingestion")

    behavior = pd.DataFrame()
    if rebuild_behavior:
        teams_df = pd.DataFrame({"team": team_list})
        behavior = rebuild_team_behavior_csv(merged, teams_df=teams_df, reference_date=end)

    return {
        "status": "success",
        "start_date": start,
        "end_date": end,
        "teams": team_list,
        "teams_count": len(team_list),
        "fetched_rows": len(fetched),
        "historical_rows": len(merged),
        "team_behavior_rows": len(behavior),
        "source": fetched.attrs.get("source_label", ""),
        "warning": fetched.attrs.get("warning", ""),
        "output": str(output),
    }


def load_team_list_from_ratings(path: Path | None = None) -> list[str]:
    ratings_path = path or (DATA_DIR / "team_ratings.csv")
    ratings = read_csv_with_columns(ratings_path, ["team"])
    if ratings.empty or "team" not in ratings.columns:
        return []
    return sorted({normalize_team_name(team) for team in ratings["team"].dropna().astype(str) if team.strip()})


def enrich_historical_matches_with_environment(matches_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """Placeholder for a later weather enrichment pass.

    Historical Data Ingestion v1 is match-result only. This function is kept
    inactive by default so missing weather cannot block ingestion.
    """
    return normalise_historical_matches(matches_df)


def _records(matches: list[dict[str, Any]] | pd.DataFrame | None) -> list[dict[str, Any]]:
    if matches is None:
        return []
    if isinstance(matches, pd.DataFrame):
        return matches.to_dict("records")
    if isinstance(matches, list):
        return [match for match in matches if isinstance(match, dict)]
    return []


def _match_payload_to_result_row(match: dict[str, Any]) -> dict[str, Any] | None:
    kickoff = pd.to_datetime(match.get("utcDate") or match.get("date_utc"), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return None
    score = match.get("score") or {}
    full_time = score.get("fullTime") if isinstance(score, dict) else {}
    if not isinstance(full_time, dict):
        full_time = {}

    home_goals = match.get("home_goals", full_time.get("home"))
    away_goals = match.get("away_goals", full_time.get("away"))
    if home_goals is None or away_goals is None:
        return None

    home_payload = match.get("homeTeam") or {}
    away_payload = match.get("awayTeam") or {}
    home = normalize_team_name(_team_name(home_payload) or match.get("home", ""))
    away = normalize_team_name(_team_name(away_payload) or match.get("away", ""))
    if not home or not away:
        return None

    competition = match.get("competition") or {}
    competition_name = competition.get("name") if isinstance(competition, dict) else str(competition or "")
    return {
        "match_id": str(match.get("id") or match.get("match_id") or ""),
        "date_utc": kickoff.date().isoformat(),
        "competition": competition_name or "",
        "competition_type": classify_competition_type(competition_name or ""),
        "home": home,
        "away": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "country": _area_name(match),
    }


def _result_row_to_long_rows(row: dict[str, Any], source: str, last_updated: str) -> list[dict[str, Any]]:
    common = {
        "match_id": row["match_id"],
        "date_utc": row["date_utc"],
        "competition": row["competition"],
        "competition_type": row["competition_type"],
        "is_neutral": pd.NA,
        "venue": "",
        "city": "",
        "country": row.get("country", ""),
        "team_xg": pd.NA,
        "opponent_xg": pd.NA,
        "shots_for": pd.NA,
        "shots_against": pd.NA,
        "shots_on_target_for": pd.NA,
        "shots_on_target_against": pd.NA,
        "possession_pct": pd.NA,
        "opponent_elo": pd.NA,
        "team_elo_pre": pd.NA,
        "temperature_c": pd.NA,
        "humidity_pct": pd.NA,
        "altitude_m": pd.NA,
        "wind_kmh": pd.NA,
        "precipitation_mm": pd.NA,
        "roof_closed": pd.NA,
        "source": source,
        "last_updated": last_updated,
    }
    return [
        {
            **common,
            "team": row["home"],
            "opponent": row["away"],
            "is_home": 1,
            "team_goals": row["home_goals"],
            "opponent_goals": row["away_goals"],
        },
        {
            **common,
            "team": row["away"],
            "opponent": row["home"],
            "is_home": 0,
            "team_goals": row["away_goals"],
            "opponent_goals": row["home_goals"],
        },
    ]


def _team_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("shortName") or payload.get("name") or payload.get("tla") or "").strip()


def _area_name(match: dict[str, Any]) -> str:
    area = match.get("area") or {}
    if isinstance(area, dict):
        return str(area.get("name") or "").strip()
    return ""


def _dedupe_key(row: pd.Series) -> str:
    match_id = str(row.get("match_id", "") or "").strip()
    team = str(row.get("team", "") or "").strip().lower()
    opponent = str(row.get("opponent", "") or "").strip().lower()
    if match_id:
        return f"match_id|{match_id}|{team}|{opponent}"
    return "|".join(
        [
            "fallback",
            str(row.get("date_utc", "") or ""),
            team,
            opponent,
            str(row.get("team_goals", "") or ""),
            str(row.get("opponent_goals", "") or ""),
        ]
    )


def _source_detail(source_label: str) -> str:
    if source_label == SOURCE_API:
        return "football-data.org API"
    if source_label == SOURCE_CACHE:
        return "data/cache football-data.org responses"
    if source_label == SOURCE_LOCAL:
        return "data/historical_matches.csv"
    return "football-data.org unavailable"
