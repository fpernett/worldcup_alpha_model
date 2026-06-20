from __future__ import annotations

from datetime import date, timezone
from typing import Any

import pandas as pd

from src.cache import cache_last_updated, read_dataframe_cache, set_source_attrs, utc_now_iso, write_dataframe_cache
from src.config import DATA_DIR, SOURCE_CACHE, SOURCE_LOCAL
from src.schema_registry import validate_historical_matches_schema
from src.utils import read_csv_with_columns


HISTORICAL_MATCH_COLUMNS = [
    "match_id",
    "date_utc",
    "competition",
    "competition_type",
    "team",
    "opponent",
    "is_home",
    "is_neutral",
    "venue",
    "city",
    "country",
    "team_goals",
    "opponent_goals",
    "team_xg",
    "opponent_xg",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "possession_pct",
    "opponent_elo",
    "team_elo_pre",
    "temperature_c",
    "humidity_pct",
    "altitude_m",
    "wind_kmh",
    "precipitation_mm",
    "roof_closed",
    "source",
    "last_updated",
]


def load_historical_matches(path=None, use_cache: bool = True) -> pd.DataFrame:
    """Load team-match long historical data with cache/local fallback.

    Missing files, empty files, and malformed schemas return an empty dataframe
    with the required columns rather than raising into Streamlit.
    """
    warnings: list[str] = []
    if path is None and use_cache:
        cached = read_dataframe_cache("historical_matches_latest.csv", max_age_hours=24)
        if cached is not None and not cached.empty:
            out = _normalise_historical_matches(cached)
            warnings = validate_historical_matches_schema(out)
            return set_source_attrs(
                out,
                SOURCE_CACHE,
                "data/cache/historical_matches_latest.csv",
                cache_last_updated("historical_matches_latest.csv"),
                warning="; ".join(warnings) if warnings else None,
            )

    local_path = DATA_DIR / "historical_matches.csv" if path is None else path
    try:
        local = read_csv_with_columns(local_path, HISTORICAL_MATCH_COLUMNS)
        out = _normalise_historical_matches(local)
    except Exception as exc:
        out = pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
        warnings.append(f"historical_matches.csv read failed: {exc}")

    warnings.extend(validate_historical_matches_schema(out))
    detail = _relative_data_path(local_path)
    return set_source_attrs(
        out,
        SOURCE_LOCAL,
        detail,
        _csv_last_modified(local_path),
        warning="; ".join(dict.fromkeys(warnings)) if warnings else None,
    )


def build_historical_matches_from_results(results_df: pd.DataFrame | None, source: str = "football_results") -> pd.DataFrame:
    """Convert home/away result rows into required team-match long format."""
    if results_df is None or results_df.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)

    df = results_df.copy()
    for col in ["date_utc", "home", "away", "home_goals", "away_goals", "home_elo", "away_elo"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    for col in ["home_goals", "away_goals", "home_elo", "away_elo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"]).copy()

    rows: list[dict[str, Any]] = []
    now = utc_now_iso()
    for _, row in df.iterrows():
        match_id = str(row.get("match_id") or _generated_match_id(row))
        common = {
            "match_id": match_id,
            "date_utc": row["date_utc"].date().isoformat(),
            "competition": row.get("competition", "") or "",
            "competition_type": row.get("competition_type", "other") or "other",
            "is_neutral": row.get("is_neutral", pd.NA),
            "venue": row.get("venue", "") or "",
            "city": row.get("city", "") or "",
            "country": row.get("country", "") or "",
            "temperature_c": row.get("temperature_c", pd.NA),
            "humidity_pct": row.get("humidity_pct", pd.NA),
            "altitude_m": row.get("altitude_m", pd.NA),
            "wind_kmh": row.get("wind_kmh", pd.NA),
            "precipitation_mm": row.get("precipitation_mm", pd.NA),
            "roof_closed": row.get("roof_closed", pd.NA),
            "source": row.get("source", source) or source,
            "last_updated": row.get("last_updated", now) or now,
        }
        home = str(row["home"]).strip()
        away = str(row["away"]).strip()
        rows.append(
            {
                **common,
                "team": home,
                "opponent": away,
                "is_home": 1,
                "team_goals": row["home_goals"],
                "opponent_goals": row["away_goals"],
                "team_xg": row.get("home_xg", pd.NA),
                "opponent_xg": row.get("away_xg", pd.NA),
                "shots_for": row.get("home_shots", pd.NA),
                "shots_against": row.get("away_shots", pd.NA),
                "shots_on_target_for": row.get("home_shots_on_target", pd.NA),
                "shots_on_target_against": row.get("away_shots_on_target", pd.NA),
                "possession_pct": row.get("home_possession_pct", pd.NA),
                "opponent_elo": row.get("away_elo", pd.NA),
                "team_elo_pre": row.get("home_elo", pd.NA),
            }
        )
        rows.append(
            {
                **common,
                "team": away,
                "opponent": home,
                "is_home": 0,
                "team_goals": row["away_goals"],
                "opponent_goals": row["home_goals"],
                "team_xg": row.get("away_xg", pd.NA),
                "opponent_xg": row.get("home_xg", pd.NA),
                "shots_for": row.get("away_shots", pd.NA),
                "shots_against": row.get("home_shots", pd.NA),
                "shots_on_target_for": row.get("away_shots_on_target", pd.NA),
                "shots_on_target_against": row.get("home_shots_on_target", pd.NA),
                "possession_pct": row.get("away_possession_pct", pd.NA),
                "opponent_elo": row.get("home_elo", pd.NA),
                "team_elo_pre": row.get("away_elo", pd.NA),
            }
        )

    return _normalise_historical_matches(pd.DataFrame(rows))


def merge_and_save_historical_matches(new_matches: pd.DataFrame, path=None) -> pd.DataFrame:
    """Merge new long-format rows into local historical CSV without dropping manual rows."""
    local_path = DATA_DIR / "historical_matches.csv" if path is None else path
    existing = _normalise_historical_matches(read_csv_with_columns(local_path, HISTORICAL_MATCH_COLUMNS))
    incoming = _normalise_historical_matches(new_matches)
    if incoming.empty:
        return existing

    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=["match_id", "team"], keep="last")
    combined = combined.sort_values(["date_utc", "match_id", "team"], ascending=[False, True, True])
    combined.to_csv(local_path, index=False)
    write_dataframe_cache(combined, "historical_matches_latest.csv", "local historical match merge")
    return set_source_attrs(
        combined.reset_index(drop=True),
        SOURCE_LOCAL,
        _relative_data_path(local_path),
        _csv_last_modified(local_path),
    )


def normalise_historical_matches(df: pd.DataFrame | None) -> pd.DataFrame:
    return _normalise_historical_matches(df)


def _normalise_historical_matches(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    out = df.copy()
    aliases = {
        "date": "date_utc",
        "team_name": "team",
        "opponent_name": "opponent",
        "goals_for": "team_goals",
        "goals_against": "opponent_goals",
        "xg_for": "team_xg",
        "xg_against": "opponent_xg",
        "temp_c": "temperature_c",
        "roof_expected_closed": "roof_closed",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[HISTORICAL_MATCH_COLUMNS].copy()
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce")
    out = out.dropna(subset=["date_utc", "team", "opponent"]).copy()
    for col in [
        "team_goals",
        "opponent_goals",
        "team_xg",
        "opponent_xg",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "possession_pct",
        "opponent_elo",
        "team_elo_pre",
        "temperature_c",
        "humidity_pct",
        "altitude_m",
        "wind_kmh",
        "precipitation_mm",
        "is_home",
        "is_neutral",
        "roof_closed",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["match_id", "competition", "competition_type", "team", "opponent", "venue", "city", "country", "source"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["competition_type"] = out["competition_type"].replace("", "other")
    out["match_id"] = out.apply(lambda row: row["match_id"] or _generated_match_id(row), axis=1)
    out["date_utc"] = out["date_utc"].dt.date.astype(str)
    out["last_updated"] = out["last_updated"].fillna("").astype(str)
    return out.sort_values(["date_utc", "match_id", "team"], ascending=[False, True, True]).reset_index(drop=True)


def _generated_match_id(row: pd.Series) -> str:
    date_value = pd.to_datetime(row.get("date_utc"), errors="coerce")
    date_part = date_value.date().isoformat().replace("-", "") if not pd.isna(date_value) else "unknown_date"
    home = str(row.get("home", row.get("team", "team"))).strip().lower().replace(" ", "_")
    away = str(row.get("away", row.get("opponent", "opponent"))).strip().lower().replace(" ", "_")
    goals_for = str(row.get("team_goals", "") or "").replace(".", "_")
    goals_against = str(row.get("opponent_goals", "") or "").replace(".", "_")
    score = f"_{goals_for}_{goals_against}" if goals_for or goals_against else ""
    return f"{date_part}_{home}_{away}{score}"


def _csv_last_modified(path) -> str:
    if path is None or not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz=timezone.utc).isoformat()


def _relative_data_path(path) -> str:
    try:
        return str(path.relative_to(DATA_DIR.parent))
    except Exception:
        return str(path)
