from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.polymarket import POLYMARKET_COLUMNS
from src.storage import load_csv


MAPPING_COLUMNS = [
    "match_id",
    "home",
    "away",
    "market_id",
    "question",
    "market_type",
    "model_side",
    "polymarket_side",
    "mapping_confidence",
    "mapping_reason",
    "manual_confirmed",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
    "closed",
]

MANUAL_MAPPING_COLUMNS = [
    "match_id",
    "market_id",
    "market_type",
    "model_side",
    "polymarket_side",
    "manual_confirmed",
    "notes",
]

TEAM_ALIASES = {
    "Bosnia and Herzegovina": ["bosnia", "bih"],
    "Canada": ["can"],
    "Colombia": ["col"],
    "Croatia": ["cro"],
    "Czechia": ["czech republic", "cze"],
    "DR Congo": ["congo", "drc", "cod"],
    "England": ["eng"],
    "Ghana": ["gha"],
    "Mexico": ["mex"],
    "Panama": ["pan"],
    "Portugal": ["por"],
    "Qatar": ["qat"],
    "South Africa": ["rsa", "south-africa"],
    "South Korea": ["korea republic", "kor"],
    "Switzerland": ["sui", "swiss"],
    "Uzbekistan": ["uzb"],
}


def map_match_to_polymarket_markets(match_row: pd.Series, polymarket_df: pd.DataFrame) -> pd.DataFrame:
    home = str(match_row.get("home", ""))
    away = str(match_row.get("away", ""))
    match_id = str(match_row.get("match_id", ""))
    manual = _manual_mappings(match_id, polymarket_df, home, away)
    automatic = _automatic_mappings(match_row, polymarket_df)

    if not manual.empty:
        manual_ids = set(manual["market_id"].astype(str))
        automatic = automatic.loc[~automatic["market_id"].astype(str).isin(manual_ids)]
    out = pd.concat([manual, automatic], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    return out[MAPPING_COLUMNS].drop_duplicates(subset=["match_id", "market_id", "market_type", "polymarket_side"])


def load_manual_mappings() -> pd.DataFrame:
    return load_csv("market_mappings.csv", MANUAL_MAPPING_COLUMNS)


def mapping_status(mapped_markets: pd.DataFrame) -> str:
    if mapped_markets.empty:
        return "unmapped"
    if mapped_markets["manual_confirmed"].astype(str).str.lower().isin({"1", "true", "yes"}).any():
        return "manual"
    if (mapped_markets["mapping_confidence"] == "high").any():
        return "automatic high confidence"
    return "automatic low confidence"


def _manual_mappings(match_id: str, markets: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    manual = load_manual_mappings()
    manual = manual.loc[manual["match_id"].astype(str) == str(match_id)].copy()
    if manual.empty or markets.empty:
        return pd.DataFrame(columns=MAPPING_COLUMNS)

    market_cols = [col for col in POLYMARKET_COLUMNS if col in markets.columns]
    merged = manual.merge(markets[market_cols], on="market_id", how="left")
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "match_id": match_id,
                "home": home,
                "away": away,
                "market_id": row.get("market_id", ""),
                "question": row.get("question", ""),
                "market_type": row.get("market_type", "other"),
                "model_side": row.get("model_side", ""),
                "polymarket_side": _normalise_side(row.get("polymarket_side", "YES")),
                "mapping_confidence": "manual",
                "mapping_reason": f"Manual mapping file. {row.get('notes', '')}".strip(),
                "manual_confirmed": True,
                "yes_price": row.get("yes_price", pd.NA),
                "no_price": row.get("no_price", pd.NA),
                "liquidity": row.get("liquidity", pd.NA),
                "volume": row.get("volume", pd.NA),
                "closed": row.get("closed", pd.NA),
            }
        )
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def _automatic_mappings(match_row: pd.Series, markets: pd.DataFrame) -> pd.DataFrame:
    if markets is None or markets.empty:
        return pd.DataFrame(columns=MAPPING_COLUMNS)

    home = str(match_row.get("home", ""))
    away = str(match_row.get("away", ""))
    match_id = str(match_row.get("match_id", ""))
    kickoff = pd.to_datetime(match_row.get("date_utc"), errors="coerce")
    rows = []

    for _, market in markets.iterrows():
        text = _market_text(market)
        score, reasons = _team_match_score(text, home, away)
        if score <= 0:
            continue

        proximity_score, proximity_reason = _date_proximity_score(kickoff, market)
        score += proximity_score
        if proximity_reason:
            reasons.append(proximity_reason)

        market_type, model_side, polymarket_side, type_reason, type_score = _infer_market_type(text, home, away)
        score += type_score
        reasons.append(type_reason)

        if market_type == "other" and score < 0.60:
            continue

        confidence = "high" if score >= 0.70 else "low"
        rows.append(
            {
                "match_id": match_id,
                "home": home,
                "away": away,
                "market_id": market.get("market_id", ""),
                "question": market.get("question", ""),
                "market_type": market_type,
                "model_side": model_side,
                "polymarket_side": polymarket_side,
                "mapping_confidence": confidence,
                "mapping_reason": "; ".join(reasons),
                "manual_confirmed": False,
                "yes_price": market.get("yes_price", pd.NA),
                "no_price": market.get("no_price", pd.NA),
                "liquidity": market.get("liquidity", pd.NA),
                "volume": market.get("volume", pd.NA),
                "closed": market.get("closed", pd.NA),
            }
        )

    if not rows:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    out = pd.DataFrame(rows, columns=MAPPING_COLUMNS)
    return out.sort_values(["mapping_confidence", "market_type"], ascending=[True, True]).reset_index(drop=True)


def _team_match_score(text: str, home: str, away: str) -> tuple[float, list[str]]:
    home_terms = _team_terms(home)
    away_terms = _team_terms(away)
    has_home = any(_contains_term(text, term) for term in home_terms)
    has_away = any(_contains_term(text, term) for term in away_terms)
    reasons = []
    score = 0.0
    if has_home and has_away:
        score += 0.50
        reasons.append("home and away team found")
    elif has_home or has_away:
        score += 0.20
        reasons.append("one team found")
    return score, reasons


def _infer_market_type(text: str, home: str, away: str) -> tuple[str, str, str, str, float]:
    if "under 2.5" in text or "under-2.5" in text:
        return "under_2_5", "under_2_5", "YES", "under 2.5 market", 0.25
    if "over 2.5" in text or "over-2.5" in text:
        return "over_2_5", "over_2_5", "YES", "over 2.5 market", 0.25
    if "both teams to score" in text or "btts" in text:
        if " no" in f" {text}" or "not both" in text:
            return "btts_no", "btts_no", "YES", "BTTS No market", 0.25
        return "btts_yes", "btts_yes", "YES", "BTTS Yes market", 0.25
    if "draw" in text or "tie" in text:
        return "draw", "draw", "YES", "draw market", 0.25
    if "correct score" in text or re.search(r"\b\d+\s*-\s*\d+\b", text):
        return "correct_score", "correct_score", "YES", "correct score market", 0.15
    if "group" in text and ("winner" in text or "win group" in text):
        return "group_winner", "group_winner", "YES", "group winner market", 0.10
    if "qualif" in text or "advance" in text:
        return "qualification", "qualification", "YES", "qualification market", 0.10

    home_pos = _first_team_position(text, home)
    away_pos = _first_team_position(text, away)
    winner_words = any(word in text for word in ["beat", "defeat", "win", "winner"])
    if winner_words and home_pos is not None and away_pos is not None:
        if home_pos < away_pos:
            return "match_winner_home", "home_win", "YES", f"{home} appears as winner side", 0.25
        return "match_winner_away", "away_win", "YES", f"{away} appears as winner side", 0.25
    if winner_words and home_pos is not None:
        return "match_winner_home", "home_win", "YES", f"{home} winner wording", 0.20
    if winner_words and away_pos is not None:
        return "match_winner_away", "away_win", "YES", f"{away} winner wording", 0.20
    return "other", "other", "YES", "generic related market", 0.00


def _date_proximity_score(kickoff: pd.Timestamp, market: pd.Series) -> tuple[float, str]:
    if pd.isna(kickoff):
        return 0.0, ""
    dates = [
        pd.to_datetime(market.get("start_date"), errors="coerce"),
        pd.to_datetime(market.get("end_date"), errors="coerce"),
    ]
    for value in dates:
        if pd.isna(value):
            continue
        if abs(value.date() - kickoff.date()) <= timedelta(days=2):
            return 0.15, "market date near kickoff"
    return 0.0, ""


def _market_text(market: pd.Series) -> str:
    return " ".join(
        str(market.get(col, "") or "")
        for col in ["question", "slug", "event_title", "category"]
    ).lower()


def _team_terms(team: str) -> list[str]:
    terms = [team.lower()]
    terms.extend(alias.lower() for alias in TEAM_ALIASES.get(team, []))
    return [term for term in terms if term]


def _contains_term(text: str, term: str) -> bool:
    term = term.lower().strip()
    if len(term) <= 3:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _first_team_position(text: str, team: str) -> int | None:
    positions = [text.find(term) for term in _team_terms(team) if term in text]
    positions = [pos for pos in positions if pos >= 0]
    return min(positions) if positions else None


def _normalise_side(value: Any) -> str:
    text = str(value or "YES").strip().upper()
    return "NO" if text == "NO" else "YES"
