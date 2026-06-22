from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import DATA_DIR
from src.team_names import normalize_team_name, polymarket_team_codes
from src.utils import coerce_float, today_iso


MATCH_SEARCH_SCORE_COLUMNS = [
    "market_id",
    "question",
    "slug",
    "event_title",
    "category",
    "score",
    "confidence",
    "matched_home",
    "matched_away",
    "matched_competition",
    "matched_sports_terms",
    "rejected",
    "reject_reason",
    "score_breakdown",
]

POLITICAL_TERMS = {
    "approval",
    "bitcoin",
    "congress",
    "crypto",
    "election",
    "fed",
    "government",
    "inflation",
    "interest rate",
    "mayor",
    "minister",
    "poll",
    "president",
    "senate",
    "tariff",
    "vote",
    "war",
}

SPORTS_TERMS = {
    "fifa",
    "football",
    "game",
    "match",
    "soccer",
    "world cup",
}

MATCHUP_TERMS = {"vs", "v", "versus", "beat", "defeat", "draw", "tie"}

TEAM_CODE_ALIASES = {
    "Algeria": ["ALG"],
    "Argentina": ["ARG"],
    "Australia": ["AUS"],
    "Austria": ["AUT"],
    "Belgium": ["BEL"],
    "Bosnia and Herzegovina": ["BIH"],
    "Brazil": ["BRA"],
    "Canada": ["CAN"],
    "Cape Verde": ["CPV"],
    "Colombia": ["COL"],
    "Croatia": ["CRO"],
    "Curaçao": ["CUW", "CUR"],
    "Czechia": ["CZE"],
    "DR Congo": ["CDR", "COD", "DRC", "Congo DR"],
    "Ecuador": ["ECU"],
    "Egypt": ["EGY"],
    "England": ["ENG"],
    "France": ["FRA"],
    "Germany": ["GER"],
    "Ghana": ["GHA"],
    "Haiti": ["HAI"],
    "Iran": ["IRN", "IR Iran"],
    "Iraq": ["IRQ"],
    "Ivory Coast": ["CIV", "Cote d'Ivoire", "Côte d'Ivoire"],
    "Japan": ["JPN"],
    "Jordan": ["JOR"],
    "Mexico": ["MEX"],
    "Morocco": ["MAR"],
    "Netherlands": ["NED", "Holland"],
    "New Zealand": ["NZL"],
    "Norway": ["NOR"],
    "Panama": ["PAN"],
    "Paraguay": ["PAR"],
    "Portugal": ["POR"],
    "Qatar": ["QAT"],
    "Saudi Arabia": ["KSA"],
    "Scotland": ["SCO"],
    "Senegal": ["SEN"],
    "South Africa": ["RSA"],
    "South Korea": ["KOR", "KR", "Korea Republic"],
    "Spain": ["ESP"],
    "Sweden": ["SWE"],
    "Switzerland": ["SUI"],
    "Tunisia": ["TUN"],
    "Turkey": ["TUR", "Turkiye", "Türkiye"],
    "United States": ["USA", "US", "USMNT"],
    "Uruguay": ["URU"],
    "Uzbekistan": ["UZB"],
}


def build_polymarket_match_queries(home: str, away: str, competition: str | None = None) -> list[str]:
    """Build matchup-specific search strings; never team-only by default."""
    home_name = normalize_team_name(home)
    away_name = normalize_team_name(away)
    competition_text = str(competition or "").strip()
    queries: list[str] = []
    if not home_name or not away_name:
        return queries

    home_terms = _query_team_terms(home_name)
    away_terms = _query_team_terms(away_name)
    home_primary = home_terms[0]
    away_primary = away_terms[0]
    home_codes = _code_terms(home_name)
    away_codes = _code_terms(away_name)
    home_code = home_codes[0] if home_codes else ""
    away_code = away_codes[0] if away_codes else ""

    queries.extend(
        [
            f"{home_primary} {away_primary}",
            f"{home_primary} vs {away_primary}",
            f"{home_primary} v {away_primary}",
            f"Will {home_primary} beat {away_primary}",
            f"Will {away_primary} beat {home_primary}",
            f"{home_primary} draw {away_primary}",
        ]
    )
    if home_code and away_code:
        queries.append(f"{home_code} {away_code}")
    for home_alias_code in home_codes[:4]:
        for away_alias_code in away_codes[:4]:
            queries.append(f"{home_alias_code} {away_alias_code}")
            if _is_world_cup_competition(competition_text):
                queries.append(f"fifwc-{home_alias_code.lower()}-{away_alias_code.lower()}")
    if competition_text:
        queries.extend(
            [
                f"{competition_text} {home_primary} {away_primary}",
                f"FIFA {competition_text} {home_primary} {away_primary}",
            ]
        )
    for home_alias in home_terms[1:3]:
        queries.append(f"{home_alias} {away_primary}")
        queries.append(f"{home_alias} vs {away_primary}")
    for away_alias in away_terms[1:3]:
        queries.append(f"{home_primary} {away_alias}")
        queries.append(f"{home_primary} vs {away_alias}")
    return list(dict.fromkeys(query for query in queries if query.strip()))


def score_polymarket_market_for_match(
    market: dict | pd.Series,
    home: str,
    away: str,
    competition: str | None = None,
    fixture_date: str | None = None,
) -> dict[str, Any]:
    """Score one Polymarket market for a selected football matchup."""
    row = market if isinstance(market, pd.Series) else pd.Series(market or {})
    home_name = normalize_team_name(home)
    away_name = normalize_team_name(away)
    text = _market_text(row)
    text_key = _normalise_text(text)
    category_key = _normalise_text(row.get("category", ""))
    score = 0.0
    reasons: list[str] = []

    matched_home_terms = _matched_terms(text_key, _team_terms(home_name))
    matched_away_terms = _matched_terms(text_key, _team_terms(away_name))
    matched_home = bool(matched_home_terms)
    matched_away = bool(matched_away_terms)
    matched_competition = _matched_competition(text_key, competition)
    matched_sports_terms = _matched_keywords(text_key, SPORTS_TERMS)
    matched_matchup_terms = _matched_keywords(text_key, MATCHUP_TERMS)
    political_hits = _matched_keywords(text_key, POLITICAL_TERMS)

    if matched_home and matched_away:
        score += 60.0
        reasons.append(f"both teams matched ({', '.join(matched_home_terms[:2])}; {', '.join(matched_away_terms[:2])})")
    elif matched_home or matched_away:
        score += 12.0
        reasons.append("only one selected team matched")

    if matched_matchup_terms:
        score += 12.0
        reasons.append(f"matchup terms: {', '.join(matched_matchup_terms)}")
    if matched_sports_terms:
        score += 14.0
        reasons.append(f"sports terms: {', '.join(matched_sports_terms)}")
    if _category_is_sports(category_key):
        score += 10.0
        reasons.append("sports category")
    if matched_competition:
        score += 8.0
        reasons.append("competition matched")
    date_score, date_reason = _date_score(row, fixture_date)
    if date_score:
        score += date_score
        reasons.append(date_reason)

    rejected = False
    reject_reason = ""
    if political_hits and not (matched_home and matched_away and matched_sports_terms):
        rejected = True
        reject_reason = f"political/non-sports market: {', '.join(political_hits)}"
        score -= 80.0
    elif _category_is_politics(category_key) and not matched_sports_terms:
        rejected = True
        reject_reason = "political/non-sports category"
        score -= 80.0
    elif not (matched_home and matched_away):
        rejected = True
        reject_reason = "only one or neither selected team matched"
        score -= 35.0
    elif not (matched_sports_terms or matched_matchup_terms or _category_is_sports(category_key)):
        rejected = True
        reject_reason = "both teams matched but no sports or matchup context"
        score -= 20.0

    score = round(max(score, 0.0), 2)
    if rejected:
        confidence = "reject"
    elif matched_home and matched_away and (matched_sports_terms or _category_is_sports(category_key)) and score >= 75.0:
        confidence = "high"
    elif matched_home and matched_away and score >= 55.0:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "market_id": _string_value(row.get("market_id", row.get("id", ""))),
        "question": _string_value(row.get("question", row.get("title", ""))),
        "slug": _string_value(row.get("slug", "")),
        "event_title": _string_value(row.get("event_title", row.get("eventTitle", ""))),
        "category": _string_value(row.get("category", "")),
        "score": score,
        "confidence": confidence,
        "matched_home": matched_home,
        "matched_away": matched_away,
        "matched_competition": bool(matched_competition),
        "matched_sports_terms": bool(matched_sports_terms or _category_is_sports(category_key)),
        "rejected": bool(rejected),
        "reject_reason": reject_reason,
        "score_breakdown": "; ".join(dict.fromkeys(reasons)),
    }


def score_polymarket_markets_for_match(
    markets_df: pd.DataFrame | None,
    home: str,
    away: str,
    fixture_date: str | None = None,
    competition: str | None = "World Cup",
    include_rejected: bool = True,
) -> pd.DataFrame:
    markets = markets_df.copy() if markets_df is not None else pd.DataFrame()
    rows = [
        score_polymarket_market_for_match(row, home, away, competition=competition, fixture_date=fixture_date)
        for _, row in markets.iterrows()
    ]
    out = pd.DataFrame(rows, columns=MATCH_SEARCH_SCORE_COLUMNS)
    if out.empty:
        return out
    if not include_rejected:
        out = out.loc[~out["rejected"].astype(bool)].copy()
    out["_confidence_order"] = out["confidence"].map({"high": 0, "medium": 1, "low": 2, "reject": 3}).fillna(9)
    out = out.sort_values(["rejected", "_confidence_order", "score"], ascending=[True, True, False])
    return out.drop(columns=["_confidence_order"]).reset_index(drop=True)


def find_best_polymarket_match_candidates(
    home: str,
    away: str,
    fixture_date: str | None = None,
    competition: str | None = "World Cup",
    max_candidates: int = 10,
    include_rejected: bool = False,
    retrieval_mode: str = "auto",
    min_confidence: str = "medium",
    active: bool = True,
    closed: bool | None = False,
    market_loader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Run layered market retrieval and score the returned markets locally."""
    from src.polymarket import POLYMARKET_COLUMNS, get_polymarket_markets
    from src.polymarket_sports_discovery import (
        discover_polymarket_sports_event_markets,
        flattened_markets_to_polymarket_rows,
        load_polymarket_markets_cache,
    )

    loader = market_loader or get_polymarket_markets
    queries = build_polymarket_match_queries(home, away, competition)
    layers_tried: list[str] = []
    scored_layers: list[pd.DataFrame] = []
    events_fetched = 0
    markets_flattened = 0
    cache_paths: dict[str, str] = {}

    def add_scored(markets: pd.DataFrame, layer: str, search_query_col: str | None = None) -> None:
        if markets is None or markets.empty:
            return
        scored = score_polymarket_markets_for_match(markets, home, away, fixture_date, competition, include_rejected=True)
        if scored.empty:
            return
        scored["retrieval_layer"] = layer
        if search_query_col and search_query_col in markets.columns and "market_id" in markets.columns:
            query_by_market = markets.groupby("market_id")[search_query_col].apply(lambda values: "; ".join(dict.fromkeys(values.astype(str)))).to_dict()
            scored["search_query"] = scored["market_id"].map(query_by_market)
        elif "search_query" not in scored.columns:
            scored["search_query"] = ""
        scored_layers.append(scored)

    if retrieval_mode in {"query_only", "auto"}:
        layers_tried.append("query_search")
        query_frames = []
        for query in queries:
            try:
                frame = loader(query=query, use_cache=False, force_refresh=True, write_cache=False)
            except TypeError:
                frame = loader(query)
            except Exception:
                frame = pd.DataFrame(columns=POLYMARKET_COLUMNS)
            if frame is not None and not frame.empty:
                frame = frame.copy()
                frame["search_query"] = query
                query_frames.append(frame)
        if query_frames:
            markets = pd.concat(query_frames, ignore_index=True)
            if "market_id" in markets.columns:
                query_by_market = markets.groupby("market_id")["search_query"].apply(lambda values: "; ".join(dict.fromkeys(values.astype(str)))).to_dict()
                markets = markets.drop_duplicates(subset=["market_id"], keep="first").copy()
                markets["search_query"] = markets["market_id"].map(query_by_market)
            add_scored(markets, "query_search", "search_query")

    if retrieval_mode in {"sports_events", "auto"} and not _has_accepted(scored_layers, min_confidence):
        layers_tried.append("sports_events")
        sports_markets, diagnostics = discover_polymarket_sports_event_markets(
            active=active,
            closed=closed,
            retrieval_mode="sports_events",
            write_cache=True,
        )
        events_fetched += int(diagnostics.get("events_fetched", 0))
        markets_flattened += int(diagnostics.get("markets_flattened", 0))
        cache_paths.update(diagnostics.get("cache_paths", {}) or {})
        rows = flattened_markets_to_polymarket_rows(sports_markets)
        add_scored(rows, "sports_events")

    if retrieval_mode in {"all_events", "auto"} and not _has_accepted(scored_layers, min_confidence):
        layers_tried.append("all_events")
        event_markets, diagnostics = discover_polymarket_sports_event_markets(
            active=active,
            closed=closed,
            retrieval_mode="all_events",
            write_cache=True,
        )
        events_fetched += int(diagnostics.get("events_fetched", 0))
        markets_flattened += int(diagnostics.get("markets_flattened", 0))
        cache_paths.update(diagnostics.get("cache_paths", {}) or {})
        rows = flattened_markets_to_polymarket_rows(event_markets)
        add_scored(rows, "all_events")

    if retrieval_mode in {"cache", "auto"} and not _has_accepted(scored_layers, min_confidence):
        layers_tried.append("cache")
        cached = load_polymarket_markets_cache()
        rows = flattened_markets_to_polymarket_rows(cached)
        add_scored(rows, "cache")

    if not scored_layers:
        out = pd.DataFrame(columns=MATCH_SEARCH_SCORE_COLUMNS + ["search_query", "retrieval_layer"])
        _attach_discovery_attrs(out, queries, layers_tried, events_fetched, markets_flattened, 0, cache_paths)
        return out

    scored_all = pd.concat(scored_layers, ignore_index=True)
    if "market_id" in scored_all.columns:
        scored_all = scored_all.drop_duplicates(subset=["market_id"], keep="first").copy()
    filtered = _filter_candidates(scored_all, include_rejected=include_rejected, min_confidence=min_confidence)
    filtered = _sort_candidates(filtered)
    if max_candidates and max_candidates > 0:
        filtered = filtered.head(max_candidates).copy()
    _attach_discovery_attrs(filtered, queries, layers_tried, events_fetched, markets_flattened, len(scored_all), cache_paths)
    if filtered.empty:
        filtered.attrs["no_match_reason"] = _no_match_reason()
    return filtered.reset_index(drop=True)


def write_match_search_audit_report(
    candidates_df: pd.DataFrame,
    queries: list[str],
    home: str,
    away: str,
    competition: str | None,
    report_dir: str | Path = DATA_DIR.parent / "reports",
) -> tuple[Path, Path]:
    report_root = Path(report_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = report_root / f"polymarket_match_search_{today}.csv"
    md_path = report_root / f"polymarket_match_search_{today}.md"
    candidates = candidates_df.copy() if candidates_df is not None else pd.DataFrame(columns=MATCH_SEARCH_SCORE_COLUMNS)
    candidates.to_csv(csv_path, index=False)
    accepted = candidates.loc[~candidates.get("rejected", pd.Series(dtype=bool)).astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    rejected = candidates.loc[candidates.get("rejected", pd.Series(dtype=bool)).astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    best = accepted.head(1)
    md_path.write_text(
        "\n".join(
            [
                f"# Polymarket Match Search Audit - {today}",
                "",
                f"Match: **{home} vs {away}**",
                f"Competition: `{competition or ''}`",
                "",
                "This report audits read-only market discovery. It does not place trades, size positions, or change the football model.",
                "",
                "## Queries Used",
                "",
                "\n".join(f"- `{query}`" for query in queries) or "_No queries._",
                "",
                "## Best Candidate",
                "",
                _to_markdown(best),
                "",
                "## Accepted Candidates",
                "",
                _to_markdown(accepted),
                "",
                "## Rejected Candidates",
                "",
                _to_markdown(rejected),
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _has_accepted(scored_layers: list[pd.DataFrame], min_confidence: str) -> bool:
    if not scored_layers:
        return False
    combined = pd.concat(scored_layers, ignore_index=True)
    accepted = _filter_candidates(combined, include_rejected=False, min_confidence=min_confidence)
    return not accepted.empty


def _filter_candidates(scored: pd.DataFrame, include_rejected: bool, min_confidence: str) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    if not include_rejected:
        out = out.loc[~out["rejected"].astype(bool)].copy()
    minimum = _confidence_value(min_confidence)
    if not include_rejected:
        out = out.loc[out["confidence"].map(_confidence_value) >= minimum].copy()
    return out


def _sort_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    out["_confidence_order"] = out["confidence"].map({"high": 0, "medium": 1, "low": 2, "reject": 3}).fillna(9)
    return out.sort_values(["rejected", "_confidence_order", "score"], ascending=[True, True, False]).drop(columns=["_confidence_order"])


def _confidence_value(confidence: str) -> int:
    return {"reject": 0, "low": 1, "medium": 2, "high": 3}.get(str(confidence or "").lower(), 2)


def _attach_discovery_attrs(
    df: pd.DataFrame,
    queries: list[str],
    layers_tried: list[str],
    events_fetched: int,
    markets_flattened: int,
    markets_scored: int,
    cache_paths: dict[str, str],
) -> None:
    df.attrs["queries_used"] = queries
    df.attrs["retrieval_layers_tried"] = layers_tried
    df.attrs["events_fetched"] = int(events_fetched)
    df.attrs["markets_flattened"] = int(markets_flattened)
    df.attrs["markets_scored"] = int(markets_scored)
    df.attrs["cache_paths"] = cache_paths
    if df.empty:
        df.attrs["no_match_reason"] = _no_match_reason()


def _no_match_reason() -> str:
    return (
        "No Polymarket match market was discovered. Possible reasons: market does not exist yet; "
        "market is nested under an event not returned by active filters; team names differ from aliases; "
        "market is closed/resolved; API cache is stale. Try --retrieval-mode all_events, --include-closed, "
        "or manually inspect Polymarket and add a confirmed row to data/market_mappings.csv."
    )


def candidate_summary(candidates_df: pd.DataFrame | None) -> dict[str, Any]:
    candidates = candidates_df.copy() if candidates_df is not None else pd.DataFrame(columns=MATCH_SEARCH_SCORE_COLUMNS)
    if candidates.empty:
        return {
            "markets_returned": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 0,
            "best_candidate": "",
            "confidence": "",
            "reason": "",
        }
    rejected = candidates["rejected"].astype(bool)
    accepted = candidates.loc[~rejected].copy()
    best = accepted.head(1)
    return {
        "markets_returned": int(len(candidates)),
        "accepted_candidates": int((~rejected).sum()),
        "rejected_candidates": int(rejected.sum()),
        "best_candidate": str(best.iloc[0].get("question", "")) if not best.empty else "",
        "confidence": str(best.iloc[0].get("confidence", "")) if not best.empty else "",
        "reason": str(best.iloc[0].get("score_breakdown", "")) if not best.empty else "",
    }


def _query_team_terms(team: str) -> list[str]:
    canonical = normalize_team_name(team)
    terms = [canonical]
    terms.extend(TEAM_CODE_ALIASES.get(canonical, []))
    terms.extend(polymarket_team_codes(canonical))
    # Add a couple of common display variants from the alias table.
    for alias, target in TEAM_CODE_ALIASES.items():
        if normalize_team_name(alias) == canonical:
            terms.extend(target)
    return list(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))


def _primary_code(team: str) -> str:
    for term in _code_terms(team):
        value = str(term).strip()
        if 2 <= len(value) <= 4 and value.isupper():
            return value
    return ""


def _code_terms(team: str) -> list[str]:
    canonical = normalize_team_name(team)
    terms = []
    terms.extend(polymarket_team_codes(canonical))
    terms.extend(TEAM_CODE_ALIASES.get(canonical, []))
    return list(
        dict.fromkeys(
            str(term).strip().upper()
            for term in terms
            if 2 <= len(str(term).strip()) <= 4 and str(term).strip().upper() == str(term).strip()
        )
    )


def _is_world_cup_competition(competition: str | None) -> bool:
    text = _normalise_text(competition or "")
    return "world cup" in text or "fifa" in text


def _team_terms(team: str) -> list[str]:
    canonical = normalize_team_name(team)
    terms = {canonical}
    terms.update(TEAM_CODE_ALIASES.get(canonical, []))
    terms.update(polymarket_team_codes(canonical))
    if canonical == "United States":
        terms.update(["USA", "United States of America"])
    if canonical == "DR Congo":
        terms.update(["Congo DR", "D.R. Congo", "Democratic Republic of Congo"])
    if canonical == "South Korea":
        terms.update(["Korea Republic", "Republic of Korea"])
    return list(dict.fromkeys(_normalise_text(term) for term in terms if str(term).strip()))


def _market_text(row: pd.Series) -> str:
    parts = [row.get(col, "") for col in ["question", "slug", "event_title", "category"]]
    values = []
    for part in parts:
        try:
            if pd.isna(part):
                continue
        except (TypeError, ValueError):
            pass
        values.append(str(part))
    return " ".join(values)


def _normalise_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_value = ascii_value.replace("’", "'").replace("&", " and ")
    ascii_value = re.sub(r"[^A-Za-z0-9']+", " ", ascii_value)
    return " ".join(ascii_value.lower().split())


def _string_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term and _contains_term(text, term)]


def _matched_keywords(text: str, terms: set[str]) -> list[str]:
    return sorted(term for term in terms if _contains_term(text, _normalise_text(term)))


def _matched_competition(text: str, competition: str | None) -> bool:
    if not competition:
        return False
    comp = _normalise_text(competition)
    if _contains_term(text, comp):
        return True
    return comp == "world cup" and _contains_term(text, "fifa")


def _contains_term(text: str, term: str) -> bool:
    term = _normalise_text(term)
    if not term:
        return False
    if len(term) <= 4 and " " not in term:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def _category_is_sports(category: str) -> bool:
    return any(term in category for term in ["sport", "soccer", "football"])


def _category_is_politics(category: str) -> bool:
    return any(term in category for term in ["politic", "election", "government"])


def _date_score(row: pd.Series, fixture_date: str | None) -> tuple[float, str]:
    if not fixture_date:
        return 0.0, ""
    fixture = pd.to_datetime(fixture_date, errors="coerce")
    if pd.isna(fixture):
        return 0.0, ""
    for col in ["start_date", "end_date"]:
        value = pd.to_datetime(row.get(col), errors="coerce")
        if pd.isna(value):
            continue
        if abs(value.date() - fixture.date()).days <= 2:
            return 6.0, "market date near fixture"
    return 0.0, ""


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
