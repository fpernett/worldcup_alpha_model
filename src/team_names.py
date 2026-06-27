from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from src.config import DATA_DIR
from src.utils import read_csv_with_columns


TEAM_NAME_ALIAS_COLUMNS = ["alias", "canonical"]

DEFAULT_TEAM_ALIASES = {
    "col": "Colombia",
    "usa": "United States",
    "por": "Portugal",
    "prt": "Portugal",
    "u.s.a.": "United States",
    "usmnt": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    "south korea": "South Korea",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina",
    "dr congo": "DR Congo",
    "cdr": "DR Congo",
    "cod": "DR Congo",
    "drc": "DR Congo",
    "congo dr": "DR Congo",
    "congo, the democratic republic of the": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "bosnia-h.": "Bosnia and Herzegovina",
    "bosnia h": "Bosnia and Herzegovina",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "cuw": "Curaçao",
    "cur": "Curaçao",
    "ir iran": "Iran",
    "iri": "Iran",
    "iran": "Iran",
    "ksa": "Saudi Arabia",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "tur": "Turkey",
    "turkey": "Turkey",
    "kor": "South Korea",
    "cze": "Czechia",
    "civ": "Ivory Coast",
}

POLYMARKET_TEAM_CODE_ALIASES = {
    "URY": "Uruguay",
    "URU": "Uruguay",
    "ESP": "Spain",
    "PRT": "Portugal",
    "POR": "Portugal",
    "CVI": "Cape Verde",
    "CPV": "Cape Verde",
    "CDR": "DR Congo",
    "COD": "DR Congo",
    "DRC": "DR Congo",
    "COL": "Colombia",
    "USA": "United States",
    "KOR": "South Korea",
    "CZE": "Czechia",
    "CUW": "Curaçao",
    "CUR": "Curaçao",
    "CIV": "Ivory Coast",
    "IRI": "Iran",
    "IRN": "Iran",
    "KSA": "Saudi Arabia",
    "TUR": "Turkey",
    "DZA": "Algeria",
    "ALG": "Algeria",
    "JOR": "Jordan",
}

POLYMARKET_CANONICAL_TEAM_CODES = {
    "Uruguay": ["URY", "URU"],
    "Spain": ["ESP"],
    "Portugal": ["PRT", "POR"],
    "Cape Verde": ["CVI", "CPV"],
    "Colombia": ["COL"],
    "DR Congo": ["CDR", "COD", "DRC"],
    "United States": ["USA", "US"],
    "South Korea": ["KOR", "KR"],
    "Czechia": ["CZE"],
    "Curaçao": ["CUW", "CUR"],
    "Ivory Coast": ["CIV"],
    "Iran": ["IRI", "IRN"],
    "Saudi Arabia": ["KSA"],
    "Turkey": ["TUR"],
    "Algeria": ["DZA", "ALG"],
    "Jordan": ["JOR"],
}


def normalize_team_name(name: str) -> str:
    """Return the canonical team name used by local CSVs where known."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    aliases = load_team_name_aliases()
    return aliases.get(_alias_key(raw), raw)


def team_name_key(name: str) -> str:
    """Return the accent-insensitive comparison key used for aliases."""
    return _alias_key(name)


def normalize_polymarket_team_code(code: str) -> str:
    """Return canonical team name for a Polymarket sports abbreviation."""
    value = str(code or "").strip().upper()
    if not value:
        return ""
    return POLYMARKET_TEAM_CODE_ALIASES.get(value, normalize_team_name(value))


def polymarket_team_codes(team: str) -> list[str]:
    """Return Polymarket-specific code variants for a canonical team."""
    canonical = normalize_team_name(team)
    configured = POLYMARKET_CANONICAL_TEAM_CODES.get(canonical, [])
    generic = []
    raw = str(canonical or "").strip()
    if raw:
        letters = re.findall(r"[A-Za-z0-9]+", raw)
        fallback = "".join(letters)[:3].upper()
        if fallback:
            generic.append(fallback)
    return list(dict.fromkeys([code.upper() for code in [*configured, *generic] if str(code).strip()]))


def load_team_name_aliases_df(include_defaults: bool = True) -> pd.DataFrame:
    """Return aliases as a dataframe so diagnostics can report name issues."""
    rows: list[dict[str, str]] = []
    if include_defaults:
        rows.extend({"alias": alias, "canonical": canonical} for alias, canonical in DEFAULT_TEAM_ALIASES.items())
    path = DATA_DIR / "team_name_aliases.csv"
    df = read_csv_with_columns(path, TEAM_NAME_ALIAS_COLUMNS)
    if not df.empty:
        for _, row in df.iterrows():
            alias = str(row.get("alias", "") or "").strip()
            canonical = str(row.get("canonical", "") or "").strip()
            if alias and canonical:
                rows.append({"alias": alias, "canonical": canonical})
    out = pd.DataFrame(rows, columns=TEAM_NAME_ALIAS_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=TEAM_NAME_ALIAS_COLUMNS)
    out["_alias_key"] = out["alias"].map(_alias_key)
    out = out.drop_duplicates(subset=["_alias_key"], keep="last").drop(columns=["_alias_key"])
    return out.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_team_name_aliases() -> dict[str, str]:
    aliases = {_alias_key(alias): canonical for alias, canonical in DEFAULT_TEAM_ALIASES.items()}
    path = DATA_DIR / "team_name_aliases.csv"
    df = read_csv_with_columns(path, TEAM_NAME_ALIAS_COLUMNS)
    if df.empty:
        return aliases
    for _, row in df.iterrows():
        alias = str(row.get("alias", "") or "").strip()
        canonical = str(row.get("canonical", "") or "").strip()
        if alias and canonical:
            aliases[_alias_key(alias)] = canonical
    return aliases


def _alias_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_value = ascii_value.lower().replace("&", " and ")
    ascii_value = re.sub(r"[^a-z0-9']+", " ", ascii_value)
    return " ".join(ascii_value.split())
