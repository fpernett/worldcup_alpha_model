from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.team_names import normalize_team_name, team_name_key
from src.utils import read_csv_with_columns


POLYMARKET_EVENT_REGISTRY_COLUMNS = [
    "competition",
    "event_date",
    "event_slug",
    "event_url",
    "home",
    "away",
    "home_code",
    "away_code",
    "event_title",
    "source",
    "last_checked_utc",
    "confidence",
    "notes",
]

DEFAULT_POLYMARKET_EVENT_REGISTRY_PATH = DATA_DIR / "polymarket_sports_event_registry.csv"


def load_polymarket_event_registry(path: str = "data/polymarket_sports_event_registry.csv") -> pd.DataFrame:
    registry_path = _resolve_registry_path(path)
    registry = read_csv_with_columns(registry_path, POLYMARKET_EVENT_REGISTRY_COLUMNS)
    for col in POLYMARKET_EVENT_REGISTRY_COLUMNS:
        if col not in registry.columns:
            registry[col] = pd.NA
    if registry.empty:
        return registry[POLYMARKET_EVENT_REGISTRY_COLUMNS].copy()
    registry["event_date"] = pd.to_datetime(registry["event_date"], errors="coerce").dt.date.astype(str)
    return registry[POLYMARKET_EVENT_REGISTRY_COLUMNS].copy()


def match_fixture_to_polymarket_registry(
    home: str,
    away: str,
    fixture_date: str,
    competition: str,
    registry_df: pd.DataFrame,
) -> dict[str, Any]:
    registry = registry_df.copy() if registry_df is not None else pd.DataFrame(columns=POLYMARKET_EVENT_REGISTRY_COLUMNS)
    for col in POLYMARKET_EVENT_REGISTRY_COLUMNS:
        if col not in registry.columns:
            registry[col] = pd.NA
    fixture_date_slug = _date_slug(fixture_date)
    home_key = team_name_key(normalize_team_name(home))
    away_key = team_name_key(normalize_team_name(away))
    competition_key = _competition_key(competition)

    base = {
        "resolved_slug": "",
        "resolved_url": "",
        "resolution_status": "unresolved",
        "confidence": "",
        "matched_home": False,
        "matched_away": False,
        "matched_date": False,
        "registry_home": "",
        "registry_away": "",
        "registry_home_code": "",
        "registry_away_code": "",
        "team_order": "",
        "source": "polymarket_event_registry",
        "warning": "No registry match found for fixture.",
    }
    if registry.empty or not fixture_date_slug:
        return base

    registry["_event_date_slug"] = pd.to_datetime(registry["event_date"], errors="coerce").dt.date.astype(str)
    date_matches = registry.loc[registry["_event_date_slug"] == fixture_date_slug].copy()
    if date_matches.empty:
        out = base.copy()
        out["warning"] = "No registry row matched fixture date."
        return out

    if competition_key:
        date_matches["_competition_key"] = date_matches["competition"].map(_competition_key)
        competition_matches = date_matches.loc[date_matches["_competition_key"] == competition_key].copy()
        if not competition_matches.empty:
            date_matches = competition_matches

    for _, row in date_matches.iterrows():
        registry_home_key = team_name_key(normalize_team_name(str(row.get("home", "") or "")))
        registry_away_key = team_name_key(normalize_team_name(str(row.get("away", "") or "")))
        home_away = home_key == registry_home_key and away_key == registry_away_key
        away_home = home_key == registry_away_key and away_key == registry_home_key
        if not (home_away or away_home):
            continue
        return {
            "resolved_slug": str(row.get("event_slug", "") or "").strip(),
            "resolved_url": str(row.get("event_url", "") or "").strip(),
            "resolution_status": "resolved",
            "confidence": str(row.get("confidence", "") or "high").strip() or "high",
            "matched_home": True,
            "matched_away": True,
            "matched_date": True,
            "registry_home": str(row.get("home", "") or ""),
            "registry_away": str(row.get("away", "") or ""),
            "registry_home_code": str(row.get("home_code", "") or ""),
            "registry_away_code": str(row.get("away_code", "") or ""),
            "team_order": "home-away" if home_away else "away-home",
            "source": "polymarket_event_registry",
            "warning": "",
        }

    out = base.copy()
    out["matched_date"] = True
    out["warning"] = "Registry date matched, but teams did not match fixture."
    return out


def _resolve_registry_path(path: str) -> Path:
    value = Path(str(path or ""))
    if value.is_absolute():
        return value
    if value.parts and value.parts[0] == "data":
        return DATA_DIR.parent / value
    return DATA_DIR / value


def _date_slug(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.date().isoformat()


def _competition_key(value: Any) -> str:
    text = team_name_key(str(value or ""))
    if "world cup" in text or "fifa world cup" in text:
        return "world cup"
    return text
