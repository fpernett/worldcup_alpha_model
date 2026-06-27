from __future__ import annotations

import json
import re
from itertools import product
from typing import Any

import pandas as pd

from src.config import SOURCE_API, SOURCE_CACHE
from src.model import fair_odds
from src.polymarket_sports_discovery import (
    GAMMA_EVENT_MARKET_COLUMNS,
    MARKETS_CACHE_PATH,
    flatten_gamma_events_to_markets,
)
from src.polymarket_url_resolver import (
    DEFAULT_POLYMARKET_WEB_URL,
    classify_polymarket_market_type,
    fetch_polymarket_event_by_slug_with_diagnostics,
    parse_polymarket_url_or_slug,
    score_polymarket_event_slug_for_fixture,
)
from src.team_names import normalize_team_name, polymarket_team_codes, team_name_key
from src.utils import coerce_float, read_csv_with_columns, utc_now_iso


POLYMARKET_EVENT_MARKET_COLUMNS = [
    "event_slug",
    "event_title",
    "market_id",
    "market_slug",
    "question",
    "market_title",
    "market_type",
    "outcomes",
    "outcome_name",
    "price_cents",
    "odds_decimal",
    "liquidity",
    "volume",
    "source",
    "last_updated",
]

JOINED_MARKET_COLUMNS = [
    "market",
    "selection",
    "model_probability",
    "fair_odds",
    "fair_price_cents",
    "market_price_cents",
    "market_odds_decimal",
    "alpha_gap_cents",
    "ev",
    "context_probability",
    "context_fair_price_cents",
    "context_alpha_gap_cents",
    "context_signal",
    "context_reason",
    "score",
    "signal",
    "polymarket_market_id",
    "polymarket_market_slug",
    "polymarket_question",
    "mapping_confidence",
    "mapping_reason",
    "event_slug",
]


def get_polymarket_team_code_variants(team: str) -> list[str]:
    """Return Polymarket sports slug code variants, prioritized for slug building."""
    canonical = normalize_team_name(str(team or "").strip())
    codes = polymarket_team_codes(canonical or team)
    return list(dict.fromkeys(code.upper() for code in codes if str(code).strip()))


def build_polymarket_sports_slug_candidates(
    home: str,
    away: str,
    fixture_date: str,
    competition: str = "World Cup",
) -> list[str]:
    prefix = _competition_prefix(competition)
    date_slug = _fixture_date_slug(fixture_date)
    home_codes = get_polymarket_team_code_variants(home)
    away_codes = get_polymarket_team_code_variants(away)
    if not prefix or not date_slug or not home_codes or not away_codes:
        return []

    candidates: list[str] = []
    for home_code, away_code in product(home_codes, away_codes):
        candidates.append(f"{prefix}-{home_code.lower()}-{away_code.lower()}-{date_slug}")
        candidates.append(f"{prefix}-{away_code.lower()}-{home_code.lower()}-{date_slug}")
    return list(dict.fromkeys(candidates))


def resolve_polymarket_slug_for_fixture(
    home: str,
    away: str,
    fixture_date: str,
    competition: str = "World Cup",
    user_supplied_slug_or_url: str | None = None,
    cache_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    candidates = build_polymarket_sports_slug_candidates(home, away, fixture_date, competition)
    tried: list[str] = []

    def base_response(slug: str = "", source: str = "", warning: str = "") -> dict[str, Any]:
        score = score_polymarket_event_slug_for_fixture(slug, home, away, fixture_date) if slug else {}
        return {
            "resolved_slug": slug,
            "resolved_url": _event_url(slug, competition) if slug else "",
            "resolution_status": "resolved" if slug and score.get("confidence") in {"high", "medium"} else "unresolved",
            "confidence": score.get("confidence", ""),
            "matched_home": bool(score.get("matched_home", False)),
            "matched_away": bool(score.get("matched_away", False)),
            "matched_date": bool(score.get("matched_date", False)),
            "team_order": _team_order(score, home, away),
            "candidates_tried": tried.copy(),
            "source": source,
            "warning": warning or score.get("warning", ""),
        }

    if user_supplied_slug_or_url:
        parsed = parse_polymarket_url_or_slug(user_supplied_slug_or_url)
        slug = str(parsed.get("slug", "") or "").strip()
        if slug:
            tried.append(slug)
            result = base_response(slug, "user_supplied_slug_or_url", str(parsed.get("warning", "") or ""))
            if result["resolution_status"] == "resolved":
                return result
            result["warning"] = "; ".join(part for part in [result["warning"], "User-supplied slug did not validate against fixture."] if part)
            return result

    cache_match = _find_slug_in_cache(candidates, cache_df)
    tried.extend(slug for slug in candidates if slug not in tried)
    if cache_match:
        return base_response(cache_match, "local_event_or_market_cache")

    for slug in candidates:
        event, diagnostics = fetch_polymarket_event_by_slug_with_diagnostics(slug)
        if event:
            if slug not in tried:
                tried.append(slug)
            return base_response(slug, diagnostics.get("method", "gamma_event_api"))

    text_match = _find_text_cache_match(home, away, fixture_date, cache_df)
    if text_match:
        if text_match not in tried:
            tried.append(text_match)
        return base_response(text_match, "local_cache_text_search", "Matched by cache text search after deterministic candidates.")

    out = base_response("", "", "No Polymarket event resolved for this fixture.")
    out["candidates_tried"] = tried
    return out


def load_polymarket_event_markets_by_slug(slug: str) -> pd.DataFrame:
    slug = str(slug or "").strip()
    if not slug:
        return pd.DataFrame(columns=POLYMARKET_EVENT_MARKET_COLUMNS)

    event, diagnostics = fetch_polymarket_event_by_slug_with_diagnostics(slug)
    if event:
        flat = flatten_gamma_events_to_markets([event])
        out = _outcome_level_market_rows(flat, fallback_slug=slug, source=SOURCE_API)
        out.attrs["source_label"] = f"Polymarket event {diagnostics.get('method', 'api')}"
        out.attrs["last_updated"] = utc_now_iso()
        return out

    cached = read_csv_with_columns(MARKETS_CACHE_PATH, GAMMA_EVENT_MARKET_COLUMNS)
    if not cached.empty and "event_slug" in cached.columns:
        cached = cached.loc[cached["event_slug"].astype(str) == slug].copy()
        if not cached.empty:
            out = _outcome_level_market_rows(cached, fallback_slug=slug, source=SOURCE_CACHE)
            out.attrs["source_label"] = "data/polymarket_markets_cache.csv"
            out.attrs["last_updated"] = utc_now_iso()
            return out

    out = pd.DataFrame(columns=POLYMARKET_EVENT_MARKET_COLUMNS)
    out.attrs["warning"] = "; ".join(diagnostics.get("warnings", [])) if isinstance(diagnostics, dict) else "No event found for slug."
    return out


def join_polymarket_prices_to_model_markets(
    model_market_families_df: pd.DataFrame,
    polymarket_event_markets_df: pd.DataFrame,
    home: str,
    away: str,
) -> pd.DataFrame:
    model = _normalise_model_market_rows(model_market_families_df)
    pm = polymarket_event_markets_df.copy() if polymarket_event_markets_df is not None else pd.DataFrame(columns=POLYMARKET_EVENT_MARKET_COLUMNS)
    for col in POLYMARKET_EVENT_MARKET_COLUMNS:
        if col not in pm.columns:
            pm[col] = pd.NA

    rows: list[dict[str, Any]] = []
    for _, model_row in model.iterrows():
        match = _best_market_match(model_row, pm, home, away)
        price_cents = match.get("price_cents", pd.NA)
        model_prob = coerce_float(model_row.get("model_probability"), float("nan"))
        fair_price = model_prob * 100.0 if pd.notna(model_prob) else pd.NA
        market_odds = 100.0 / float(price_cents) if pd.notna(price_cents) and float(price_cents) > 0 else pd.NA
        alpha_gap = float(fair_price) - float(price_cents) if pd.notna(fair_price) and pd.notna(price_cents) else pd.NA
        ev = float(model_prob) * float(market_odds) - 1.0 if pd.notna(model_prob) and pd.notna(market_odds) else pd.NA
        context_prob = model_row.get("context_probability", pd.NA)
        context_fair_price = float(context_prob) * 100.0 if pd.notna(context_prob) else pd.NA
        context_gap = float(context_fair_price) - float(price_cents) if pd.notna(context_fair_price) and pd.notna(price_cents) else pd.NA
        confidence = str(match.get("mapping_confidence", "low") or "low")
        score = _score(alpha_gap, confidence)
        signal = _signal(alpha_gap, confidence, pd.notna(price_cents))
        reason = str(match.get("mapping_reason", "") or "")
        if pd.isna(price_cents):
            reason = reason or "No matching Polymarket price found for this local market."

        rows.append(
            {
                "market": model_row.get("market", ""),
                "selection": model_row.get("selection", ""),
                "model_probability": model_prob,
                "fair_odds": model_row.get("fair_odds", fair_odds(model_prob) if pd.notna(model_prob) else pd.NA),
                "fair_price_cents": fair_price,
                "market_price_cents": price_cents,
                "market_odds_decimal": market_odds,
                "alpha_gap_cents": alpha_gap,
                "ev": ev,
                "context_probability": context_prob,
                "context_fair_price_cents": context_fair_price,
                "context_alpha_gap_cents": context_gap,
                "context_signal": _context_signal(context_gap, confidence, pd.notna(price_cents)),
                "context_reason": model_row.get("context_reason", ""),
                "score": score,
                "signal": signal,
                "polymarket_market_id": match.get("market_id", ""),
                "polymarket_market_slug": match.get("market_slug", ""),
                "polymarket_question": match.get("question", ""),
                "mapping_confidence": confidence,
                "mapping_reason": reason,
                "event_slug": match.get("event_slug", ""),
            }
        )

    return pd.DataFrame(rows, columns=JOINED_MARKET_COLUMNS)


def joined_market_groups(joined_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {
        "High Scoring": [],
        "Low Scoring": [],
        "Result Home": [],
        "Result Away": [],
        "Other Markets": [],
    }
    joined = joined_df.copy() if joined_df is not None else pd.DataFrame(columns=JOINED_MARKET_COLUMNS)
    for _, row in joined.iterrows():
        groups[_group_for_joined_row(row)].append(_display_joined_row(row))
    return {
        name: pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "Market",
                "Model probability",
                "Fair odds / fair price",
                "Odds / Price",
                "EV / Alpha Gap",
                "Context probability",
                "Context fair price",
                "Context alpha gap",
                "Context signal",
                "Context reason",
                "Score",
                "Signal",
                "Mapping confidence",
            ]
        )
        for name, rows in groups.items()
    }


def polymarket_alpha_rows(joined_df: pd.DataFrame) -> pd.DataFrame:
    joined = joined_df.copy() if joined_df is not None else pd.DataFrame(columns=JOINED_MARKET_COLUMNS)
    if joined.empty:
        return pd.DataFrame(
            columns=[
                "market",
                "model_probability",
                "fair_price_cents",
                "market_price_cents",
                "alpha_gap_cents",
                "context_probability",
                "context_fair_price_cents",
                "context_alpha_gap_cents",
                "context_signal",
                "score",
                "signal",
                "polymarket_question",
                "mapping_confidence",
                "event_slug",
            ]
        )
    available = joined.loc[joined["market_price_cents"].notna()].copy()
    if available.empty:
        return pd.DataFrame(columns=[
            "market",
            "model_probability",
            "fair_price_cents",
            "market_price_cents",
            "alpha_gap_cents",
            "context_probability",
            "context_fair_price_cents",
            "context_alpha_gap_cents",
            "context_signal",
            "score",
            "signal",
            "polymarket_question",
            "mapping_confidence",
            "event_slug",
        ])
    available["market"] = available["market"].astype(str) + ": " + available["selection"].astype(str)
    return available[
        [
            "market",
            "model_probability",
            "fair_price_cents",
            "market_price_cents",
            "alpha_gap_cents",
            "context_probability",
            "context_fair_price_cents",
            "context_alpha_gap_cents",
            "context_signal",
            "score",
            "signal",
            "polymarket_question",
            "mapping_confidence",
            "event_slug",
        ]
    ].sort_values("score", ascending=False).reset_index(drop=True)


def _competition_prefix(competition: str) -> str:
    return "fifwc" if "world" in str(competition or "").lower() and "cup" in str(competition or "").lower() else "fifwc"


def _fixture_date_slug(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.date().isoformat()


def _event_url(slug: str, competition: str) -> str:
    category = "world-cup" if "world" in str(competition or "").lower() else "world-cup"
    return f"{DEFAULT_POLYMARKET_WEB_URL}/sports/{category}/{slug}"


def _team_order(score: dict[str, Any], home: str, away: str) -> str:
    teams = [score.get("slug_team_1", ""), score.get("slug_team_2", "")]
    if not all(teams):
        return ""
    home_name = normalize_team_name(home)
    away_name = normalize_team_name(away)
    if teams == [home_name, away_name]:
        return "home-away"
    if teams == [away_name, home_name]:
        return "away-home"
    return "mixed"


def _find_slug_in_cache(candidates: list[str], cache_df: pd.DataFrame | None) -> str:
    cache = cache_df.copy() if cache_df is not None else read_csv_with_columns(MARKETS_CACHE_PATH, GAMMA_EVENT_MARKET_COLUMNS)
    if cache.empty:
        return ""
    candidate_set = set(candidates)
    for col in ["event_slug", "slug", "market_slug"]:
        if col in cache.columns:
            matches = cache.loc[cache[col].astype(str).isin(candidate_set)]
            if not matches.empty:
                if col == "market_slug" and "event_slug" in matches.columns:
                    return str(matches.iloc[0].get("event_slug", "") or "")
                return str(matches.iloc[0].get(col, "") or "")
    return ""


def _find_text_cache_match(home: str, away: str, fixture_date: str, cache_df: pd.DataFrame | None) -> str:
    cache = cache_df.copy() if cache_df is not None else read_csv_with_columns(MARKETS_CACHE_PATH, GAMMA_EVENT_MARKET_COLUMNS)
    if cache.empty or "event_slug" not in cache.columns:
        return ""
    home_key = team_name_key(home)
    away_key = team_name_key(away)
    date_slug = _fixture_date_slug(fixture_date)
    text_cols = [col for col in ["event_title", "question", "market_title", "market_slug", "event_slug"] if col in cache.columns]
    for _, row in cache.iterrows():
        text = team_name_key(" ".join(str(row.get(col, "") or "") for col in text_cols))
        if home_key in text and away_key in text and date_slug in str(row.get("event_slug", "")):
            return str(row.get("event_slug", "") or "")
    return ""


def _outcome_level_market_rows(flat: pd.DataFrame, fallback_slug: str, source: str) -> pd.DataFrame:
    if flat is None or flat.empty:
        return pd.DataFrame(columns=POLYMARKET_EVENT_MARKET_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in flat.iterrows():
        outcomes = _parse_jsonish(row.get("outcomes"))
        raw_market = _parse_jsonish(row.get("raw_market_json"))
        prices = _parse_jsonish(raw_market.get("outcomePrices", raw_market.get("outcome_prices"))) if isinstance(raw_market, dict) else []
        if not prices:
            prices = [row.get("yes_price", pd.NA), row.get("no_price", pd.NA)]
        market_type = classify_polymarket_market_type(row)
        base = {
            "event_slug": row.get("event_slug", fallback_slug),
            "event_title": row.get("event_title", ""),
            "market_id": row.get("market_id", ""),
            "market_slug": row.get("market_slug", ""),
            "question": row.get("question", ""),
            "market_title": row.get("market_title", ""),
            "market_type": market_type,
            "outcomes": json.dumps(outcomes) if isinstance(outcomes, list) else str(row.get("outcomes", "") or ""),
            "liquidity": row.get("liquidity", pd.NA),
            "volume": row.get("volume", pd.NA),
            "source": row.get("source", source),
            "last_updated": row.get("last_updated", utc_now_iso()),
        }

        if _is_binary_yes_no(outcomes):
            outcome_name = _infer_binary_outcome_name(row, market_type)
            price = _normalise_price_cents(row.get("yes_price", prices[0] if prices else pd.NA))
            rows.append({**base, "outcome_name": outcome_name, "price_cents": price, "odds_decimal": _price_to_odds(price)})
        elif isinstance(outcomes, list):
            for idx, outcome in enumerate(outcomes):
                price = _normalise_price_cents(prices[idx] if idx < len(prices) else pd.NA)
                rows.append({**base, "outcome_name": str(outcome), "price_cents": price, "odds_decimal": _price_to_odds(price)})
        else:
            price = _normalise_price_cents(row.get("yes_price", pd.NA))
            rows.append({**base, "outcome_name": _infer_binary_outcome_name(row, market_type), "price_cents": price, "odds_decimal": _price_to_odds(price)})
    out = pd.DataFrame(rows, columns=POLYMARKET_EVENT_MARKET_COLUMNS)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["market_id", "outcome_name"]).reset_index(drop=True)


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def _is_binary_yes_no(outcomes: Any) -> bool:
    return isinstance(outcomes, list) and len(outcomes) == 2 and {str(x).strip().lower() for x in outcomes} == {"yes", "no"}


def _infer_binary_outcome_name(row: pd.Series, market_type: str) -> str:
    text = " ".join(str(row.get(col, "") or "") for col in ["question", "market_title", "market_slug"])
    clean = _clean_text(text)
    if market_type == "total":
        match = re.search(r"\b(over|under)\s+(\d+(?:\.\d+)?)", clean)
        if match:
            return f"{match.group(1).title()} {match.group(2)}"
    if market_type == "btts":
        if re.search(r"\b(no|not)\b", clean):
            return "No"
        return "Yes"
    if market_type == "spread":
        team = _team_before_spread(text) or _group_title(row)
        line = _spread_line(clean)
        return " ".join(part for part in [team, line] if part).strip()
    if market_type == "moneyline":
        if "draw" in clean:
            return "Draw"
        group_title = _group_title(row)
        if group_title:
            return group_title
        match = re.search(r"will\s+(.+?)\s+(?:win|beat)", text, flags=re.I)
        if match:
            return match.group(1).strip(" ?.")
    return _group_title(row) or str(row.get("question", "") or row.get("market_title", "") or "")


def _group_title(row: pd.Series) -> str:
    raw = _parse_jsonish(row.get("raw_market_json"))
    if isinstance(raw, dict):
        value = raw.get("groupItemTitle") or raw.get("title")
        if value:
            return str(value).strip()
    return ""


def _spread_line(text: str) -> str:
    match = re.search(r"([+-]\s*\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(" ", "")
    match = re.search(r"by more than\s+(\d+(?:\.\d+)?)", text)
    if match:
        return f"-{match.group(1)}"
    return ""


def _team_before_spread(text: str) -> str:
    match = re.search(r"will\s+(.+?)\s+(?:cover|win by|beat)", text, flags=re.I)
    return match.group(1).strip(" ?.") if match else ""


def _normalise_price_cents(value: Any) -> float | pd.NA:
    price = coerce_float(value, float("nan"))
    if pd.isna(price):
        return pd.NA
    if 0.0 <= price <= 1.0:
        return float(price * 100.0)
    if 1.0 < price <= 100.0:
        return float(price)
    return pd.NA


def _price_to_odds(price_cents: Any) -> float | pd.NA:
    price = coerce_float(price_cents, float("nan"))
    if pd.isna(price) or price <= 0:
        return pd.NA
    return 100.0 / price


def _normalise_model_market_rows(df: pd.DataFrame) -> pd.DataFrame:
    model = df.copy() if df is not None else pd.DataFrame()
    for col in ["market", "selection", "model_prob", "model_probability", "fair_odds", "context_probability", "context_reason"]:
        if col not in model.columns:
            model[col] = pd.NA
    if "model_probability" not in model.columns or model["model_probability"].isna().all():
        model["model_probability"] = model["model_prob"]
    model["fair_odds"] = model.apply(
        lambda row: row.get("fair_odds")
        if pd.notna(row.get("fair_odds"))
        else (fair_odds(float(row.get("model_probability"))) if pd.notna(row.get("model_probability")) else pd.NA),
        axis=1,
    )
    return model[["market", "selection", "model_probability", "fair_odds", "context_probability", "context_reason"]].reset_index(drop=True)


def _best_market_match(model_row: pd.Series, pm: pd.DataFrame, home: str, away: str) -> dict[str, Any]:
    if pm.empty:
        return {"mapping_confidence": "low", "mapping_reason": "No Polymarket event markets loaded."}
    expected_type, expected_outcome = _expected_outcome(model_row, home, away)
    if not expected_type:
        return {"mapping_confidence": "low", "mapping_reason": "Unsupported local market for v1 Polymarket join."}
    work = pm.loc[pm["market_type"].astype(str).str.lower() == expected_type].copy()
    if work.empty:
        return {"mapping_confidence": "low", "mapping_reason": f"No {expected_type} market found in resolved event."}
    work["_outcome_key"] = work["outcome_name"].map(_market_key)
    expected_key = _market_key(expected_outcome)
    exact = work.loc[work["_outcome_key"] == expected_key]
    if not exact.empty:
        row = exact.iloc[0].to_dict()
        row["mapping_confidence"] = "high"
        row["mapping_reason"] = f"Matched {expected_type} outcome '{expected_outcome}' exactly."
        return row
    contains = work.loc[work["_outcome_key"].str.contains(expected_key, regex=False, na=False) | work.apply(lambda row: expected_key in _market_key(str(row.get("question", "")) + " " + str(row.get("market_slug", ""))), axis=1)]
    if not contains.empty:
        row = contains.iloc[0].to_dict()
        row["mapping_confidence"] = "medium"
        row["mapping_reason"] = f"Matched {expected_type} outcome '{expected_outcome}' by text."
        return row
    return {"mapping_confidence": "low", "mapping_reason": f"No Polymarket outcome matched '{expected_outcome}'."}


def _expected_outcome(model_row: pd.Series, home: str, away: str) -> tuple[str, str]:
    market = str(model_row.get("market", "") or "")
    selection = str(model_row.get("selection", "") or "")
    if market == "1X2":
        if selection.lower() == "home":
            selection = home
        elif selection.lower() == "away":
            selection = away
        return "moneyline", selection
    if market == "Total":
        return "total", selection
    if market == "BTTS":
        return "btts", selection
    if market == "Handicap":
        return "spread", selection
    if market == "Correct Score":
        return "exact_score", selection
    return "", ""


def _market_key(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+-]+", " ", str(value or "").lower())).strip()


def _clean_text(value: Any) -> str:
    return _market_key(value)


def _score(alpha_gap: Any, confidence: str) -> float:
    if pd.isna(alpha_gap):
        return 0.0
    factor = {"high": 1.0, "manual": 1.0, "medium": 0.6, "low": 0.3}.get(str(confidence).lower(), 0.3)
    return round(abs(float(alpha_gap)) * factor, 2)


def _signal(alpha_gap: Any, confidence: str, has_price: bool) -> str:
    if str(confidence).lower() == "low" or not has_price or pd.isna(alpha_gap):
        return "No signal"
    gap = float(alpha_gap)
    if gap >= 5.0:
        return "Positive model gap"
    if gap <= -5.0:
        return "Negative model gap"
    return "Near fair"


def _context_signal(alpha_gap: Any, confidence: str, has_price: bool) -> str:
    if str(confidence).lower() == "low" or not has_price or pd.isna(alpha_gap):
        return "No signal"
    gap = float(alpha_gap)
    if gap >= 5.0:
        return "Positive context gap"
    if gap <= -5.0:
        return "Negative context gap"
    return "Near fair"


def _group_for_joined_row(row: pd.Series) -> str:
    market = str(row.get("market", ""))
    selection = str(row.get("selection", "")).lower()
    if market == "Total" and "over" in selection:
        return "High Scoring"
    if market == "BTTS" and selection == "yes":
        return "High Scoring"
    if market == "Total" and "under" in selection:
        return "Low Scoring"
    if market == "BTTS" and selection == "no":
        return "Low Scoring"
    if market in {"1X2", "Handicap"}:
        if "draw" in selection and "/" not in selection:
            return "Other Markets"
        if "away" in selection or "/draw" in selection:
            return "Result Away"
        return "Result Home"
    return "Other Markets"


def _display_joined_row(row: pd.Series) -> dict[str, Any]:
    model_prob = coerce_float(row.get("model_probability"), float("nan"))
    fair_odds_value = coerce_float(row.get("fair_odds"), float("nan"))
    fair_price = coerce_float(row.get("fair_price_cents"), float("nan"))
    market_odds = coerce_float(row.get("market_odds_decimal"), float("nan"))
    market_price = coerce_float(row.get("market_price_cents"), float("nan"))
    gap = coerce_float(row.get("alpha_gap_cents"), float("nan"))
    ev = coerce_float(row.get("ev"), float("nan"))
    context_prob = coerce_float(row.get("context_probability"), float("nan"))
    context_fair_price = coerce_float(row.get("context_fair_price_cents"), float("nan"))
    context_gap = coerce_float(row.get("context_alpha_gap_cents"), float("nan"))
    return {
        "Market": f"{row.get('market', '')}: {row.get('selection', '')}".strip(": "),
        "Model probability": "" if pd.isna(model_prob) else f"{100 * model_prob:.1f}%",
        "Fair odds / fair price": "" if pd.isna(fair_odds_value) or pd.isna(fair_price) else f"{fair_odds_value:.2f} / {fair_price:.1f}c",
        "Odds / Price": "" if pd.isna(market_odds) or pd.isna(market_price) else f"{market_odds:.2f} / {market_price:.1f}c",
        "EV / Alpha Gap": "" if pd.isna(gap) or pd.isna(ev) else f"{100 * ev:.1f}% / {gap:+.1f}c",
        "Context probability": "" if pd.isna(context_prob) else f"{100 * context_prob:.1f}%",
        "Context fair price": "" if pd.isna(context_fair_price) else f"{context_fair_price:.1f}c",
        "Context alpha gap": "" if pd.isna(context_gap) else f"{context_gap:+.1f}c",
        "Context signal": row.get("context_signal", "No signal"),
        "Context reason": row.get("context_reason", ""),
        "Score": row.get("score", 0.0),
        "Signal": row.get("signal", "No signal"),
        "Mapping confidence": row.get("mapping_confidence", "low"),
    }
