from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests

from src.cache import (
    cache_last_updated,
    read_dataframe_cache,
    set_source_attrs,
    utc_now_iso,
    write_dataframe_cache,
    write_json_cache,
)
from src.config import DATA_DIR, SOURCE_API, SOURCE_CACHE, SOURCE_LOCAL, get_config
from src.polymarket_match_search import build_polymarket_match_queries
from src.storage import load_csv
from src.team_names import normalize_team_name, polymarket_team_codes
from src.utils import coerce_bool, coerce_float


POLYMARKET_COLUMNS = [
    "market_id",
    "question",
    "slug",
    "event_title",
    "category",
    "start_date",
    "end_date",
    "active",
    "closed",
    "outcomes",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
    "source",
    "last_updated",
]

DEFAULT_POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com"
GAMMA_PAGE_LIMIT = 100
GAMMA_MAX_PAGES = 5
TEAM_SLUG_CODES = {
    "Algeria": "alg",
    "Argentina": "arg",
    "Australia": "aus",
    "Austria": "aut",
    "Belgium": "bel",
    "Bosnia and Herzegovina": "bih",
    "Brazil": "bra",
    "Cape Verde": ["cvi", "cpv"],
    "Canada": "can",
    "Colombia": "col",
    "Croatia": "cro",
    "Curacao": ["cuw", "cur"],
    "Czechia": "cze",
    "DR Congo": ["cdr", "cod", "drc"],
    "Ecuador": "ecu",
    "Egypt": "egy",
    "England": "eng",
    "France": "fra",
    "Germany": "ger",
    "Ghana": "gha",
    "Haiti": "hai",
    "Iran": ["iri", "irn"],
    "Iraq": "irq",
    "Ivory Coast": "civ",
    "Japan": "jpn",
    "Jordan": "jor",
    "Mexico": "mex",
    "Morocco": "mar",
    "Netherlands": "ned",
    "New Zealand": "nzl",
    "Norway": "nor",
    "Panama": "pan",
    "Paraguay": "par",
    "Portugal": "por",
    "Qatar": "qat",
    "Saudi Arabia": "ksa",
    "Scotland": "sco",
    "Senegal": "sen",
    "South Africa": "rsa",
    "South Korea": ["kr", "kor"],
    "Spain": "esp",
    "Switzerland": "sui",
    "Sweden": "swe",
    "Tunisia": "tun",
    "Turkey": "tur",
    "Turkiye": ["tur", "tür"],
    "United States": ["usa", "us"],
    "Uruguay": ["ury", "uru"],
    "Uzbekistan": "uzb",
}

TEAM_SEARCH_ALIASES = {
    "Bosnia and Herzegovina": ["Bosnia"],
    "Cape Verde": ["Cabo Verde"],
    "Curacao": ["Curaçao"],
    "Czechia": ["Czech Republic"],
    "DR Congo": ["Congo", "Congo DR", "CDR", "COD", "DRC"],
    "Ivory Coast": ["Cote d'Ivoire", "Côte d'Ivoire"],
    "South Korea": ["Korea Republic", "Korea"],
    "Turkiye": ["Turkey", "Türkiye"],
    "United States": ["USA", "USMNT"],
}


def get_polymarket_markets(
    query: str | None = None,
    use_cache: bool = False,
    force_refresh: bool = False,
    write_cache: bool = True,
) -> pd.DataFrame:
    """Load Polymarket markets from API, cache, or local CSV.

    The function is intentionally read-only. It never places, previews, or signs
    orders. Empty API settings and unexpected response shapes fall back safely.
    """
    warning = ""

    if use_cache and not force_refresh:
        cached = _load_cached_markets(query)
        if cached is not None and not cached.empty:
            return cached
        if cached is not None and cached.empty:
            warning = "Cached Polymarket market file was empty; refreshing Gamma API."

    api_markets, error = _fetch_markets_from_api(query, write_cache=write_cache)
    if api_markets is not None and not api_markets.empty:
        if write_cache:
            write_dataframe_cache(api_markets, "polymarket_markets_normalized.csv", "Polymarket Gamma API")
        return set_source_attrs(api_markets, SOURCE_API, "Polymarket Gamma API", utc_now_iso(), warning=warning or None)
    warning = "; ".join(part for part in [warning, error or "Polymarket Gamma API returned no usable markets."] if part)

    cached = _load_cached_markets(query, warning=f"Polymarket Gamma API failed; using cache. {warning}")
    if cached is not None and not cached.empty:
        return cached

    local = _normalise_markets(load_csv("polymarket_markets.csv", POLYMARKET_COLUMNS), SOURCE_LOCAL)
    local = _filter_query(local, query)
    return set_source_attrs(
        local,
        SOURCE_LOCAL,
        "data/polymarket_markets.csv",
        _csv_last_modified("polymarket_markets.csv"),
        warning=f"Polymarket API unavailable; using local CSV. {warning}" if warning else None,
    )


def update_polymarket_markets(query: str | None = None) -> dict[str, Any]:
    df = get_polymarket_markets(query=query, force_refresh=True)
    return {
        "status": "success" if not df.empty else "skipped",
        "rows": len(df),
        "source": df.attrs.get("source_label", ""),
        "message": f"Loaded {len(df)} Polymarket market row(s) from {df.attrs.get('source_label', 'unknown source')}.",
        "warning": df.attrs.get("warning", ""),
    }


def get_match_polymarket_markets(home: str, away: str, date_utc: Any | None = None) -> pd.DataFrame:
    """Read-only selected-match market search.

    Broad World Cup searches often return tournament outrights first. Sports
    match pages are event-backed, so try the fixture event slug before falling
    back to team searches. The mapper still validates each market before use.
    """
    event_markets, event_error = _fetch_event_markets_from_api(_match_event_slug_candidates(home, away, date_utc))
    if event_markets is not None and not event_markets.empty:
        return event_markets

    frames = [event_markets]
    frames.extend(
        get_polymarket_markets(query=query, use_cache=False, force_refresh=True, write_cache=False)
        for query in _match_search_queries(home, away)
    )
    out = _combine_market_frames(frames, fallback_source="selected-match Gamma search")
    if out.empty:
        discovered, discovery_warning = _broad_match_market_discovery()
        if not discovered.empty:
            out = discovered
        elif discovery_warning:
            out.attrs["warning"] = discovery_warning
    if event_error and out.empty:
        out.attrs["warning"] = event_error
    return out


def normalize_price(value: Any) -> float | pd.NA:
    price = coerce_float(value, float("nan"))
    if pd.isna(price):
        return pd.NA
    if price > 1.0:
        price = price / 100.0
    if price < 0.0 or price > 1.0:
        return pd.NA
    return float(price)


def _fetch_markets_from_api(query: str | None, write_cache: bool = True) -> tuple[pd.DataFrame | None, str | None]:
    endpoint = _polymarket_endpoint()
    if not endpoint:
        return None, "No Polymarket API URL is configured."

    records: list[dict[str, Any]] = []

    try:
        for page in range(GAMMA_MAX_PAGES):
            params = _gamma_market_params(query=query, offset=page * GAMMA_PAGE_LIMIT)
            response = requests.get(endpoint, params=params, timeout=25)
            response.raise_for_status()
            payload = response.json()
            page_records = _extract_market_records(payload)
            records.extend(page_records)
            if len(page_records) < GAMMA_PAGE_LIMIT:
                break
    except Exception as exc:
        return None, str(exc)

    if write_cache:
        write_json_cache(records, "polymarket_markets_raw.json", "Polymarket Gamma API")
    if not records:
        return None, "Polymarket API payload did not contain market records."

    df = _normalise_markets(pd.DataFrame(records), SOURCE_API)
    df = _filter_query(df, query)
    return df, None


def _load_cached_markets(query: str | None, warning: str | None = None) -> pd.DataFrame | None:
    cached = read_dataframe_cache("polymarket_markets_normalized.csv", max_age_hours=None)
    if cached is None:
        return None
    df = _normalise_markets(cached, SOURCE_CACHE)
    df = _filter_query(df, query)
    return set_source_attrs(
        df,
        SOURCE_CACHE,
        "data/cache/polymarket_markets_normalized.csv",
        cache_last_updated("polymarket_markets_normalized.csv"),
        warning,
    )


def _normalise_markets(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "id": "market_id",
        "conditionId": "market_id",
        "condition_id": "market_id",
        "title": "question",
        "description": "question",
        "event": "event_title",
        "eventTitle": "event_title",
        "startDate": "start_date",
        "startDateIso": "start_date",
        "endDate": "end_date",
        "endDateIso": "end_date",
        "liquidityNum": "liquidity",
        "liquidityClob": "liquidity",
        "volumeNum": "volume",
        "volumeClob": "volume",
        "updatedAt": "last_updated",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    for col in POLYMARKET_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    rows = []
    for _, row in out.iterrows():
        outcomes = _parse_jsonish(row.get("outcomes"))
        outcome_prices = _parse_jsonish(row.get("outcomePrices", row.get("outcome_prices", pd.NA)))
        yes_price, no_price = _derive_yes_no_prices(row, outcomes, outcome_prices)
        event_title = _event_title(row)
        category = _category(row)

        rows.append(
            {
                "market_id": str(row.get("market_id", "") or "").strip(),
                "question": str(row.get("question", "") or "").strip(),
                "slug": str(row.get("slug", "") or "").strip(),
                "event_title": event_title,
                "category": category,
                "start_date": row.get("start_date", pd.NA),
                "end_date": row.get("end_date", pd.NA),
                "active": int(coerce_bool(row.get("active", True))),
                "closed": int(coerce_bool(row.get("closed", False))),
                "outcomes": _stringify_outcomes(outcomes if outcomes else row.get("outcomes", "")),
                "yes_price": normalize_price(yes_price),
                "no_price": normalize_price(no_price),
                "liquidity": coerce_float(row.get("liquidity"), float("nan")),
                "volume": coerce_float(row.get("volume"), float("nan")),
                "source": row.get("source") if pd.notna(row.get("source")) else source_label,
                "last_updated": row.get("last_updated") if pd.notna(row.get("last_updated")) else utc_now_iso(),
            }
        )

    normalised = pd.DataFrame(rows, columns=POLYMARKET_COLUMNS)
    normalised = normalised[normalised["market_id"].astype(str).str.len() > 0]
    return normalised.reset_index(drop=True)


def _derive_yes_no_prices(row: pd.Series, outcomes: Any, outcome_prices: Any) -> tuple[Any, Any]:
    yes_price = row.get("yes_price", row.get("yesPrice", pd.NA))
    no_price = row.get("no_price", row.get("noPrice", pd.NA))
    if pd.notna(yes_price) or pd.notna(no_price):
        return yes_price, no_price

    if isinstance(outcomes, list) and isinstance(outcome_prices, list):
        price_by_outcome = {
            str(outcome).strip().lower(): outcome_prices[idx]
            for idx, outcome in enumerate(outcomes)
            if idx < len(outcome_prices)
        }
        yes_price = price_by_outcome.get("yes", pd.NA)
        no_price = price_by_outcome.get("no", pd.NA)
        if pd.notna(yes_price) or pd.notna(no_price):
            return yes_price, no_price

    if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
        return outcome_prices[0], outcome_prices[1]
    return pd.NA, pd.NA


def _extract_market_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("markets", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _fetch_event_markets_from_api(slugs: list[str]) -> tuple[pd.DataFrame, str | None]:
    if not slugs:
        return pd.DataFrame(columns=POLYMARKET_COLUMNS), None

    endpoint = _polymarket_events_endpoint()
    if not endpoint:
        return pd.DataFrame(columns=POLYMARKET_COLUMNS), "No Polymarket API URL is configured."

    records: list[dict[str, Any]] = []
    errors = []
    try:
        for slug in slugs:
            response = requests.get(
                endpoint,
                params={"slug": slug, "active": "true", "closed": "false"},
                timeout=25,
            )
            response.raise_for_status()
            events = _extract_event_records(response.json())
            records.extend(_event_market_records(events))
            if records:
                break
    except Exception as exc:
        errors.append(str(exc))

    if not records:
        return pd.DataFrame(columns=POLYMARKET_COLUMNS), "; ".join(errors) if errors else None

    df = _normalise_markets(pd.DataFrame(records), SOURCE_API)
    return set_source_attrs(df, SOURCE_API, "Polymarket Gamma events API", utc_now_iso()), None


def _extract_event_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("events", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if isinstance(payload.get("markets"), list):
            return [payload]
    return []


def _event_market_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for event in events:
        event_title = str(event.get("title") or event.get("name") or event.get("slug") or "")
        event_category = _event_category(event)
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            row = market.copy()
            row["event_title"] = row.get("event_title") or row.get("eventTitle") or event_title
            row["category"] = row.get("category") or event_category
            row["source"] = SOURCE_API
            if not row.get("endDate") and event.get("endDate"):
                row["endDate"] = event.get("endDate")
            records.append(row)
    return records


def _event_category(event: dict[str, Any]) -> str:
    sport = event.get("sport")
    if isinstance(sport, str) and sport.strip():
        return sport.strip()
    tags = event.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                label = str(tag.get("label") or tag.get("name") or tag.get("slug") or "").strip()
                if label:
                    return label
            elif str(tag).strip():
                return str(tag).strip()
    series = event.get("series")
    if isinstance(series, list) and series:
        first = series[0]
        if isinstance(first, dict):
            return str(first.get("title") or first.get("slug") or "").strip()
    return ""


def _polymarket_base_url() -> str | None:
    cfg = get_config()
    base = cfg.polymarket_gamma_api_url or cfg.polymarket_api_url or DEFAULT_POLYMARKET_GAMMA_API_URL
    if not base:
        return None
    endpoint = base.rstrip("/")
    for suffix in ("/markets", "/events"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return endpoint


def _polymarket_endpoint() -> str | None:
    base = _polymarket_base_url()
    if not base:
        return None
    return f"{base}/markets"


def _polymarket_events_endpoint() -> str | None:
    base = _polymarket_base_url()
    if not base:
        return None
    return f"{base}/events"


def _match_event_slug_candidates(home: str, away: str, date_utc: Any | None) -> list[str]:
    date_slugs = _date_slug_candidates(date_utc)
    home_codes = _team_slug_codes(home)
    away_codes = _team_slug_codes(away)
    if not date_slugs or not home_codes or not away_codes:
        return []
    candidates = []
    for date_slug in date_slugs:
        for home_code in home_codes:
            for away_code in away_codes:
                candidates.append(f"fifwc-{home_code}-{away_code}-{date_slug}")
                candidates.append(f"fifwc-{away_code}-{home_code}-{date_slug}")
    return list(dict.fromkeys(candidates))


def _date_slug(value: Any | None) -> str:
    candidates = _date_slug_candidates(value)
    return candidates[0] if candidates else ""


def _date_slug_candidates(value: Any | None) -> list[str]:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return []
    dates = [timestamp.date(), (timestamp - pd.Timedelta(days=1)).date()]
    return list(dict.fromkeys(day.isoformat() for day in dates))


def _team_slug_code(team: str) -> str:
    codes = _team_slug_codes(team)
    return codes[0] if codes else ""


def _team_slug_codes(team: str) -> list[str]:
    team_text = normalize_team_name(str(team or "").strip())
    if not team_text:
        return []
    configured = TEAM_SLUG_CODES.get(team_text)
    if isinstance(configured, str):
        codes = [configured]
    elif isinstance(configured, (list, tuple)):
        codes = [str(code) for code in configured if str(code).strip()]
    else:
        codes = []
    codes.extend(code.lower() for code in polymarket_team_codes(team_text))
    letters = re.findall(r"[a-z0-9]+", team_text.lower())
    fallback = "".join(letters)[:3]
    if fallback:
        codes.append(fallback)
    return list(dict.fromkeys(code.lower().strip() for code in codes if code))


def _gamma_market_params(query: str | None, offset: int = 0) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": GAMMA_PAGE_LIMIT,
        "offset": offset,
        "active": "true",
        "closed": "false",
    }
    if query:
        # Gamma may ignore search params on some deployments; local filtering
        # below remains the source of truth.
        params["search"] = query
    return params


def _filter_query(df: pd.DataFrame, query: str | None) -> pd.DataFrame:
    if df.empty or not query:
        return df
    query_l = str(query).lower().strip()
    if not query_l:
        return df
    haystack = (
        df["question"].astype(str)
        + " "
        + df["slug"].astype(str)
        + " "
        + df["event_title"].astype(str)
        + " "
        + df["category"].astype(str)
    ).str.lower()
    phrase_match = haystack.str.contains(query_l, regex=False, na=False)
    tokens = _query_tokens(query_l)
    if not tokens:
        return df.loc[phrase_match].reset_index(drop=True)
    token_match = pd.Series(True, index=df.index)
    for token in tokens:
        token_match &= haystack.str.contains(token, regex=False, na=False)
    return df.loc[phrase_match | token_match].reset_index(drop=True)


def _query_tokens(query: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "for",
        "game",
        "match",
        "of",
        "or",
        "the",
        "to",
        "v",
        "vs",
        "will",
    }
    return [token for token in re.findall(r"[a-z0-9]+", query.lower()) if token not in stopwords]


def _match_search_queries(home: str, away: str) -> list[str]:
    return build_polymarket_match_queries(home, away, competition="World Cup")


def _team_search_terms(team: str) -> list[str]:
    team_text = str(team or "").strip()
    if not team_text:
        return []
    terms = [team_text]
    terms.extend(TEAM_SEARCH_ALIASES.get(team_text, []))
    return list(dict.fromkeys(term for term in terms if term))


def _combine_market_frames(frames: list[pd.DataFrame], fallback_source: str) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    warnings = list(
        dict.fromkeys(
            str(frame.attrs.get("warning", "")).strip()
            for frame in frames
            if frame is not None and str(frame.attrs.get("warning", "")).strip()
        )
    )
    source_labels = list(
        dict.fromkeys(
            str(frame.attrs.get("source_label", "")).strip()
            for frame in frames
            if frame is not None and str(frame.attrs.get("source_label", "")).strip()
        )
    )

    if not valid_frames:
        out = pd.DataFrame(columns=POLYMARKET_COLUMNS)
    else:
        out = pd.concat(valid_frames, ignore_index=True)
        for col in POLYMARKET_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        out = out[POLYMARKET_COLUMNS].drop_duplicates(subset=["market_id"]).reset_index(drop=True)

    out.attrs["source_label"] = " + ".join(source_labels) if source_labels else fallback_source
    out.attrs["last_updated"] = utc_now_iso()
    if out.empty and warnings:
        out.attrs["warning"] = "; ".join(warnings)
    return out


def _broad_match_market_discovery() -> tuple[pd.DataFrame, str]:
    from src.polymarket_sports_discovery import (
        discover_polymarket_sports_event_markets,
        flattened_markets_to_polymarket_rows,
        load_polymarket_markets_cache,
    )

    diagnostics = []
    layers_tried = []
    events_fetched = 0
    markets_flattened = 0
    for mode in ["sports_events", "all_events"]:
        layers_tried.append(mode)
        markets, meta = discover_polymarket_sports_event_markets(retrieval_mode=mode, write_cache=True)
        events_fetched += int(meta.get("events_fetched", 0))
        markets_flattened += int(meta.get("markets_flattened", 0))
        diagnostics.append(f"{mode}: events={meta.get('events_fetched', 0)}, markets={meta.get('markets_flattened', 0)}")
        rows = flattened_markets_to_polymarket_rows(markets)
        if not rows.empty:
            rows.attrs["source_label"] = f"Gamma {mode}"
            rows.attrs["last_updated"] = utc_now_iso()
            rows.attrs["discovery_diagnostics"] = "; ".join(diagnostics)
            rows.attrs["retrieval_layers_tried"] = layers_tried
            rows.attrs["events_fetched"] = events_fetched
            rows.attrs["markets_flattened"] = markets_flattened
            return rows, ""

    layers_tried.append("cache")
    cached = load_polymarket_markets_cache()
    rows = flattened_markets_to_polymarket_rows(cached)
    if not rows.empty:
        rows.attrs["source_label"] = "data/polymarket_markets_cache.csv"
        rows.attrs["last_updated"] = cached.attrs.get("last_updated", "")
        rows.attrs["discovery_diagnostics"] = "; ".join(diagnostics + ["cache"])
        rows.attrs["retrieval_layers_tried"] = layers_tried
        rows.attrs["events_fetched"] = events_fetched
        rows.attrs["markets_flattened"] = markets_flattened
        return rows, ""
    return pd.DataFrame(columns=POLYMARKET_COLUMNS), "; ".join(diagnostics)


def _event_title(row: pd.Series) -> str:
    value = row.get("event_title", "")
    if isinstance(value, dict):
        return str(value.get("title") or value.get("name") or "")
    if isinstance(row.get("events"), list) and row.get("events"):
        event = row.get("events")[0]
        if isinstance(event, dict):
            return str(event.get("title") or event.get("name") or event.get("slug") or "")
    if pd.isna(value):
        return ""
    return str(value)


def _category(row: pd.Series) -> str:
    value = row.get("category", "")
    if isinstance(value, dict):
        return str(value.get("name") or value.get("slug") or "")
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    if isinstance(row.get("events"), list) and row.get("events"):
        event = row.get("events")[0]
        if isinstance(event, dict):
            event_category = event.get("category")
            if isinstance(event_category, str):
                return event_category.strip()
            if isinstance(event_category, dict):
                return str(event_category.get("name") or event_category.get("slug") or "")
    return ""


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.split(",") if part.strip()]


def _stringify_outcomes(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _csv_last_modified(filename: str) -> str:
    path = DATA_DIR / filename
    if not path.exists():
        return ""
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
