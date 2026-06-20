from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from src.config import DATA_DIR
from src.utils import read_csv_with_columns


TEAM_NAME_ALIAS_COLUMNS = ["alias", "canonical"]

DEFAULT_TEAM_ALIASES = {
    "usa": "United States",
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
    "congo dr": "DR Congo",
    "congo, the democratic republic of the": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "cote d ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
}


def normalize_team_name(name: str) -> str:
    """Return the canonical team name used by local CSVs where known."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    aliases = load_team_name_aliases()
    return aliases.get(_alias_key(raw), raw)


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
