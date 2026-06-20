from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.cache import utc_now_iso, write_dataframe_cache
from src.config import DATA_DIR
from src.historical_data import load_historical_matches, normalise_historical_matches
from src.historical_ingestion import classify_competition_type, load_team_list_from_ratings, merge_historical_matches
from src.team_behavior import rebuild_team_behavior_csv
from src.team_names import load_team_name_aliases, normalize_team_name
from src.utils import coerce_bool


class HistoricalCsvImportError(ValueError):
    pass


FIELD_ALIASES = {
    "date": ["date", "date_utc", "match_date", "match date", "utc_date"],
    "home_team": ["home_team", "home team", "home", "team1", "team_1"],
    "away_team": ["away_team", "away team", "away", "team2", "team_2"],
    "home_score": ["home_score", "home score", "home_goals", "home goals", "score_home", "home_score_ft"],
    "away_score": ["away_score", "away score", "away_goals", "away goals", "score_away", "away_score_ft"],
    "tournament": ["tournament", "competition", "competition_name", "event", "cup"],
    "competition": ["competition", "tournament", "competition_name", "event", "cup"],
    "city": ["city", "venue_city", "match_city"],
    "country": ["country", "venue_country", "match_country"],
    "neutral": ["neutral", "is_neutral", "neutral_site"],
    "venue": ["venue", "stadium", "ground"],
}

REQUIRED_FIELDS = ["date", "home_team", "away_team", "home_score", "away_score"]


def import_historical_csv(
    input_path: str | Path,
    source: str,
    teams: list[str] | None = None,
    rebuild_behavior: bool = False,
    existing_path: str | Path | None = None,
    output_path: str | Path | None = None,
    behavior_path: str | Path | None = None,
    write_cache: bool = True,
) -> dict[str, Any]:
    input_file = Path(input_path)
    if not input_file.exists():
        raise HistoricalCsvImportError(f"Input CSV not found: {input_file}")

    source_df = pd.read_csv(input_file)
    existing_file = Path(existing_path) if existing_path is not None else DATA_DIR / "historical_matches.csv"
    output_file = Path(output_path) if output_path is not None else existing_file

    converted, conversion_diagnostics = convert_results_csv_to_historical(
        source_df,
        source=source,
        teams=teams,
    )
    existing = load_historical_matches(path=existing_file, use_cache=False)
    merged = merge_historical_matches(existing, converted)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    if write_cache:
        write_dataframe_cache(merged, "historical_matches_latest.csv", "historical CSV import")

    behavior = pd.DataFrame()
    team_list = _target_teams(teams)
    if rebuild_behavior:
        teams_df = pd.DataFrame({"team": team_list}) if team_list else None
        behavior = rebuild_team_behavior_csv(
            merged,
            teams_df=teams_df,
            reference_date=pd.Timestamp.now(tz="UTC").date().isoformat(),
            path=Path(behavior_path) if behavior_path is not None else None,
        )

    before_rows = len(existing)
    combined_rows = before_rows + len(converted)
    final_rows = len(merged)
    diagnostics = {
        "status": "success",
        "input": str(input_file),
        "source": source,
        "source_rows_read": int(len(source_df)),
        "rows_converted": int(len(converted)),
        "rows_added": int(max(final_rows - before_rows, 0)),
        "rows_deduplicated": int(max(combined_rows - final_rows, 0)),
        "final_historical_rows": int(final_rows),
        "final_team_behavior_rows": int(len(behavior)),
        "teams_with_zero_historical_rows": _teams_with_zero_rows(merged, team_list),
        "unknown_team_names": conversion_diagnostics["unknown_team_names"],
        "warnings": conversion_diagnostics["warnings"],
        "output": str(output_file),
    }
    return diagnostics


def convert_results_csv_to_historical(
    df: pd.DataFrame,
    source: str,
    teams: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df is None:
        raise HistoricalCsvImportError("Input dataframe is missing.")
    column_map = _detect_columns(df)
    missing = [field for field in REQUIRED_FIELDS if field not in column_map]
    if missing:
        expected = ", ".join(REQUIRED_FIELDS)
        found = ", ".join(df.columns.astype(str))
        raise HistoricalCsvImportError(
            f"Missing required source column(s): {', '.join(missing)}. "
            f"Expected at least: {expected}. Found: {found}"
        )

    target_teams = {normalize_team_name(team).lower() for team in (teams or []) if str(team).strip()}
    known_names = _known_team_names()
    unknown_names: set[str] = set()
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    last_updated = utc_now_iso()

    for idx, row in df.iterrows():
        home_raw = _value(row, column_map, "home_team")
        away_raw = _value(row, column_map, "away_team")
        home = normalize_team_name(home_raw)
        away = normalize_team_name(away_raw)
        if not home or not away:
            warnings.append(f"Row {idx}: missing home or away team; skipped.")
            continue

        if known_names and home.lower() not in known_names:
            unknown_names.add(str(home_raw))
        if known_names and away.lower() not in known_names:
            unknown_names.add(str(away_raw))

        if target_teams and home.lower() not in target_teams and away.lower() not in target_teams:
            continue

        date_value = pd.to_datetime(_value(row, column_map, "date"), errors="coerce")
        if pd.isna(date_value):
            warnings.append(f"Row {idx}: invalid date; skipped.")
            continue
        home_goals = pd.to_numeric(pd.Series([_value(row, column_map, "home_score")]), errors="coerce").iloc[0]
        away_goals = pd.to_numeric(pd.Series([_value(row, column_map, "away_score")]), errors="coerce").iloc[0]
        if pd.isna(home_goals) or pd.isna(away_goals):
            warnings.append(f"Row {idx}: invalid score; skipped.")
            continue

        competition = str(_value(row, column_map, "competition") or _value(row, column_map, "tournament") or "").strip()
        if not competition:
            competition = "Unknown"
        source_row = {
            "match_id": _source_match_id(date_value, home, away, home_goals, away_goals, competition),
            "date_utc": date_value.date().isoformat(),
            "competition": competition,
            "competition_type": classify_competition_type(competition),
            "home": home,
            "away": away,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "city": str(_value(row, column_map, "city") or "").strip(),
            "country": str(_value(row, column_map, "country") or "").strip(),
            "venue": str(_value(row, column_map, "venue") or "").strip(),
            "is_neutral": _neutral_value(_value(row, column_map, "neutral")),
        }
        rows.extend(_long_rows(source_row, source=source, last_updated=last_updated))

    diagnostics = {
        "unknown_team_names": sorted(unknown_names),
        "warnings": _compact_warnings(warnings, sorted(unknown_names)),
    }
    return normalise_historical_matches(pd.DataFrame(rows)), diagnostics


def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized = {_column_key(col): col for col in df.columns}
    detected: dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            key = _column_key(alias)
            if key in normalized:
                detected[field] = normalized[key]
                break
    return detected


def _value(row: pd.Series, column_map: dict[str, str], field: str) -> Any:
    column = column_map.get(field)
    if not column:
        return None
    value = row.get(column)
    if pd.isna(value):
        return None
    return value


def _long_rows(row: dict[str, Any], source: str, last_updated: str) -> list[dict[str, Any]]:
    common = {
        "match_id": row["match_id"],
        "date_utc": row["date_utc"],
        "competition": row["competition"],
        "competition_type": row["competition_type"],
        "is_neutral": row["is_neutral"],
        "venue": row["venue"],
        "city": row["city"],
        "country": row["country"],
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


def _source_match_id(date_value: pd.Timestamp, home: str, away: str, home_goals: Any, away_goals: Any, competition: str) -> str:
    parts = [
        "csv",
        date_value.date().isoformat(),
        home,
        away,
        str(int(home_goals)),
        str(int(away_goals)),
        competition,
    ]
    return "_".join(_slug(part) for part in parts if str(part).strip())


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "blank"


def _column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _neutral_value(value: Any) -> float:
    if value is None:
        return pd.NA
    return 1.0 if coerce_bool(value) else 0.0


def _target_teams(teams: list[str] | None) -> list[str]:
    if teams:
        return sorted({normalize_team_name(team) for team in teams if str(team).strip()})
    return load_team_list_from_ratings()


def _teams_with_zero_rows(matches: pd.DataFrame, teams: list[str]) -> list[str]:
    if not teams:
        return []
    if matches.empty or "team" not in matches.columns:
        return teams
    present = {str(team).lower() for team in matches["team"].dropna().astype(str)}
    return [team for team in teams if team.lower() not in present]


def _known_team_names() -> set[str]:
    names = set()
    for team in load_team_list_from_ratings():
        names.add(team.lower())
    aliases = load_team_name_aliases()
    for canonical in aliases.values():
        names.add(str(canonical).lower())
    return names


def _compact_warnings(warnings: list[str], unknown_names: list[str]) -> list[str]:
    out = list(warnings[:25])
    if len(warnings) > 25:
        out.append(f"{len(warnings) - 25} additional row warning(s) suppressed.")
    if unknown_names:
        preview = ", ".join(unknown_names[:25])
        suffix = f" (+{len(unknown_names) - 25} more)" if len(unknown_names) > 25 else ""
        out.append(f"Unknown team names not in team_ratings.csv or team_name_aliases.csv: {preview}{suffix}")
    return out


def read_historical_csv_source(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
