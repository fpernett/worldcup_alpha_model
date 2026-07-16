from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import pandas as pd

from src.config import DATA_DIR
from src.model import is_decisive_world_cup_fixture
from src.polymarket import POLYMARKET_COLUMNS
from src.polymarket_match_search import score_polymarket_market_for_match
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
    "mapping_score",
    "mapping_reason",
    "reject_reason",
    "manual_confirmed",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
    "closed",
]

UNMAPPED_DIAGNOSTIC_COLUMNS = [
    "market_id",
    "question",
    "event_title",
    "has_home",
    "has_away",
    "market_type_guess",
    "rejection_reason",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
]

MANUAL_MAPPING_COLUMNS = [
    "match_id",
    "market_id",
    "market_type",
    "model_side",
    "polymarket_side",
    "manual_confirmed",
    "mapping_confidence",
    "mapping_score",
    "mapping_reason",
    "reject_reason",
    "notes",
]

TEAM_ALIASES = {
    "Australia": ["aus"],
    "Belgium": ["bel"],
    "Bosnia and Herzegovina": ["bosnia", "bih"],
    "Brazil": ["bra"],
    "Cape Verde": ["cabo verde", "cpv"],
    "Canada": ["can"],
    "Colombia": ["col"],
    "Croatia": ["cro"],
    "Curacao": ["curaçao", "cuw"],
    "Czechia": ["czech republic", "cze"],
    "DR Congo": ["congo", "congo dr", "cdr", "drc", "cod"],
    "Ecuador": ["ecu"],
    "England": ["eng"],
    "Germany": ["ger"],
    "Ghana": ["gha"],
    "Haiti": ["hai"],
    "Iran": ["irn", "iri"],
    "Ivory Coast": ["cote d'ivoire", "côte d'ivoire", "civ"],
    "Japan": ["jpn"],
    "Mexico": ["mex"],
    "Morocco": ["mar"],
    "Netherlands": ["ned", "holland"],
    "New Zealand": ["nzl"],
    "Panama": ["pan"],
    "Paraguay": ["par"],
    "Portugal": ["por"],
    "Qatar": ["qat"],
    "Saudi Arabia": ["ksa"],
    "Scotland": ["sco"],
    "South Africa": ["rsa", "south-africa"],
    "South Korea": ["korea republic", "kor"],
    "Spain": ["esp"],
    "Switzerland": ["sui", "swiss"],
    "Sweden": ["swe"],
    "Tunisia": ["tun"],
    "Turkey": ["turkiye", "türkiye", "tur"],
    "Turkiye": ["turkey", "türkiye", "tur"],
    "United States": ["usa", "usmnt", "united states of america"],
    "Uruguay": ["uru"],
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


def explain_unmapped_polymarket_markets(match_row: pd.Series, polymarket_df: pd.DataFrame) -> pd.DataFrame:
    if polymarket_df is None or polymarket_df.empty:
        return pd.DataFrame(columns=UNMAPPED_DIAGNOSTIC_COLUMNS)

    home = str(match_row.get("home", ""))
    away = str(match_row.get("away", ""))
    decisive_winner = is_decisive_world_cup_fixture(match_row)
    rows = []
    for _, market in polymarket_df.iterrows():
        text = _market_text(market)
        precision = score_polymarket_market_for_match(market, home, away, competition="World Cup")
        has_home = bool(precision.get("matched_home", False))
        has_away = bool(precision.get("matched_away", False))
        market_type, _, _, _, _ = _infer_market_type(text, home, away, decisive_winner=decisive_winner)
        rows.append(
            {
                "market_id": market.get("market_id", ""),
                "question": market.get("question", ""),
                "event_title": market.get("event_title", ""),
                "has_home": has_home,
                "has_away": has_away,
                "market_type_guess": market_type,
                "rejection_reason": (
                    _unmapped_reason(text, home, away, has_home, has_away, market_type)
                    if _is_tournament_outright(text)
                    else precision.get("reject_reason") or _unmapped_reason(text, home, away, has_home, has_away, market_type)
                ),
                "yes_price": market.get("yes_price", pd.NA),
                "no_price": market.get("no_price", pd.NA),
                "liquidity": market.get("liquidity", pd.NA),
                "volume": market.get("volume", pd.NA),
            }
        )

    out = pd.DataFrame(rows, columns=UNMAPPED_DIAGNOSTIC_COLUMNS)
    out["_team_hits"] = out[["has_home", "has_away"]].sum(axis=1)
    out["_liquidity_sort"] = pd.to_numeric(out["liquidity"], errors="coerce")
    out = out.sort_values(["_team_hits", "_liquidity_sort"], ascending=[False, False], na_position="last")
    return out.drop(columns=["_team_hits", "_liquidity_sort"]).reset_index(drop=True)


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
                "mapping_score": row.get("mapping_score", pd.NA),
                "mapping_reason": f"Manual mapping file. {row.get('notes', '')}".strip(),
                "reject_reason": "",
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
    decisive_winner = is_decisive_world_cup_fixture(match_row)
    kickoff = pd.to_datetime(match_row.get("date_utc"), errors="coerce")
    rows = []

    for _, market in markets.iterrows():
        text = _market_text(market)
        precision = score_polymarket_market_for_match(
            market,
            home,
            away,
            competition=str(match_row.get("competition", "World Cup") or "World Cup"),
            fixture_date=str(match_row.get("date_utc", "") or ""),
        )
        if precision.get("rejected") or precision.get("confidence") != "high":
            continue
        if _is_tournament_outright(text):
            continue

        market_type, model_side, polymarket_side, type_reason, type_score = _infer_market_type(
            text,
            home,
            away,
            decisive_winner=decisive_winner,
        )
        has_home = bool(precision.get("matched_home", False))
        has_away = bool(precision.get("matched_away", False))

        if _is_match_level_market(market_type) and not (has_home and has_away):
            continue

        if market_type in {"other", "group_winner", "qualification"}:
            continue

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
                "mapping_confidence": "high",
                "mapping_score": precision.get("score", pd.NA),
                "mapping_reason": "; ".join(
                    part for part in [precision.get("score_breakdown", ""), type_reason] if str(part).strip()
                ),
                "reject_reason": "",
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
    has_home = _has_team(text, home)
    has_away = _has_team(text, away)
    reasons = []
    score = 0.0
    if has_home and has_away:
        score += 0.50
        reasons.append("home and away team found")
    elif has_home or has_away:
        score += 0.20
        reasons.append("one team found")
    return score, reasons


def _unmapped_reason(text: str, home: str, away: str, has_home: bool, has_away: bool, market_type: str) -> str:
    if _is_tournament_outright(text):
        return "Tournament outright, not a selected-match market"
    if not has_home and not has_away:
        return "Neither selected team found"
    if has_home and not has_away:
        return f"Only {home} found; match-level markets require both teams"
    if has_away and not has_home:
        return f"Only {away} found; match-level markets require both teams"
    if market_type in {"group_winner", "qualification", "other"}:
        return "Related market type is not supported for selected-match alpha"
    return "Below automatic mapping confidence threshold"


def _has_team(text: str, team: str) -> bool:
    return any(_contains_term(text, term) for term in _team_terms(team))


def _infer_market_type(
    text: str,
    home: str,
    away: str,
    decisive_winner: bool = False,
) -> tuple[str, str, str, str, float]:
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
    if "correct score" in text or re.search(r"(?<![\d-])\d{1,2}\s*-\s*\d{1,2}(?![\d-])", text):
        return "correct_score", "correct_score", "YES", "correct score market", 0.15
    if "group" in text and ("winner" in text or "win group" in text):
        return "group_winner", "group_winner", "YES", "group winner market", 0.10
    home_pos = _first_team_position(text, home)
    away_pos = _first_team_position(text, away)
    if "qualif" in text or "advance" in text:
        if decisive_winner and home_pos is not None and (away_pos is None or home_pos < away_pos):
            return "decisive_winner_home", "home_advance", "YES", f"{home} advancement wording", 0.25
        if decisive_winner and away_pos is not None:
            return "decisive_winner_away", "away_advance", "YES", f"{away} advancement wording", 0.25
        return "qualification", "qualification", "YES", "qualification market", 0.10

    winner_words = any(word in text for word in ["beat", "defeat", "win", "winner"])
    if winner_words and home_pos is not None and away_pos is not None:
        if home_pos < away_pos:
            if decisive_winner:
                return "decisive_winner_home", "home_advance", "YES", f"{home} eventual-winner side", 0.25
            return "match_winner_home", "home_win", "YES", f"{home} appears as winner side", 0.25
        if decisive_winner:
            return "decisive_winner_away", "away_advance", "YES", f"{away} eventual-winner side", 0.25
        return "match_winner_away", "away_win", "YES", f"{away} appears as winner side", 0.25
    if winner_words and home_pos is not None:
        if decisive_winner:
            return "decisive_winner_home", "home_advance", "YES", f"{home} eventual-winner wording", 0.20
        return "match_winner_home", "home_win", "YES", f"{home} winner wording", 0.20
    if winner_words and away_pos is not None:
        if decisive_winner:
            return "decisive_winner_away", "away_advance", "YES", f"{away} eventual-winner wording", 0.20
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


def _is_tournament_outright(text: str) -> bool:
    outright_patterns = [
        "win the 2026 fifa world cup",
        "win 2026 fifa world cup",
        "world cup winner",
        "winner of the 2026 fifa world cup",
    ]
    return any(pattern in text for pattern in outright_patterns)


def _is_match_level_market(market_type: str) -> bool:
    return market_type in {
        "match_winner_home",
        "match_winner_away",
        "decisive_winner_home",
        "decisive_winner_away",
        "draw",
        "home_not_win",
        "away_not_win",
        "over_2_5",
        "under_2_5",
        "btts_yes",
        "btts_no",
        "correct_score",
    }


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
