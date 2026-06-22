from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from src.config import SOURCE_API, SOURCE_CACHE, get_config
from src.polymarket_sports_discovery import (
    EVENTS_CACHE_PATH,
    GAMMA_EVENT_CACHE_COLUMNS,
    flatten_gamma_events_to_markets,
)
from src.team_names import normalize_polymarket_team_code, normalize_team_name, polymarket_team_codes
from src.utils import coerce_bool, coerce_float, read_csv_with_columns, utc_now_iso


DEFAULT_POLYMARKET_WEB_URL = "https://polymarket.com"
DEFAULT_POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com"

POLYMARKET_URL_MARKET_COLUMNS = [
    "event_id",
    "event_slug",
    "event_title",
    "event_category",
    "event_start_date",
    "event_end_date",
    "market_id",
    "market_slug",
    "question",
    "market_title",
    "market_type",
    "outcomes",
    "yes_price",
    "no_price",
    "liquidity",
    "volume",
    "source_url",
    "source",
    "last_updated",
    "extraction_method",
]


def parse_polymarket_url_or_slug(value: str) -> dict[str, Any]:
    raw = str(value or "").strip()
    parsed = urlparse(raw if re.match(r"^https?://", raw, flags=re.I) else "")
    path_parts = [part for part in parsed.path.split("/") if part] if parsed.scheme else []
    slug = path_parts[-1] if path_parts else raw.strip("/")
    category_path = "/".join(path_parts[:-1])
    competition_hint = ""
    if path_parts and path_parts[0] == "sports" and len(path_parts) > 1:
        competition_hint = path_parts[1]
    elif slug.startswith("fifwc-"):
        competition_hint = "world-cup"

    team_code_1 = ""
    team_code_2 = ""
    date_hint = ""
    match = re.search(r"(?:^|-)fifwc-([a-z0-9]{2,4})-([a-z0-9]{2,4})-(\d{4}-\d{2}-\d{2})(?:$|-)", slug, flags=re.I)
    if match:
        team_code_1 = match.group(1).upper()
        team_code_2 = match.group(2).upper()
        date_hint = match.group(3)

    is_sports_url = bool(path_parts and path_parts[0] == "sports") or slug.startswith("fifwc-")
    warning = ""
    if not raw:
        status = "empty"
        warning = "No URL or slug provided."
    elif not slug:
        status = "invalid"
        warning = "Could not parse a Polymarket slug."
    elif not (team_code_1 and team_code_2 and date_hint):
        status = "partial"
        warning = "Slug did not match the expected sports event pattern."
    else:
        status = "parsed"

    return {
        "url": raw if parsed.scheme else "",
        "slug": slug,
        "category_path": category_path,
        "competition_hint": competition_hint,
        "team_code_1": team_code_1,
        "team_code_2": team_code_2,
        "date_hint": date_hint,
        "is_sports_url": bool(is_sports_url),
        "parse_status": status,
        "warning": warning,
    }


def fetch_polymarket_event_by_slug(slug: str) -> dict | None:
    event, _diagnostics = fetch_polymarket_event_by_slug_with_diagnostics(slug)
    return event


def fetch_polymarket_event_by_slug_with_diagnostics(slug: str, source_url: str | None = None) -> tuple[dict | None, dict[str, Any]]:
    slug = str(slug or "").strip()
    diagnostics: dict[str, Any] = {
        "slug": slug,
        "methods_tried": [],
        "method": "",
        "warnings": [],
    }
    if not slug:
        diagnostics["warnings"].append("Missing slug.")
        return None, diagnostics

    event = _fetch_event_by_slug_param(slug, diagnostics)
    if event:
        diagnostics["method"] = "gamma_events_slug_filter"
        return event, diagnostics

    event = _fetch_event_by_slug_path(slug, diagnostics)
    if event:
        diagnostics["method"] = "gamma_events_slug_path"
        return event, diagnostics

    event = _fetch_event_by_exact_search(slug, diagnostics)
    if event:
        diagnostics["method"] = "gamma_events_search"
        return event, diagnostics

    event = _fetch_event_from_cache(slug, diagnostics)
    if event:
        diagnostics["method"] = "events_cache"
        return event, diagnostics

    event = _fetch_event_from_html(slug, source_url, diagnostics)
    if event:
        diagnostics["method"] = "polymarket_html"
        return event, diagnostics

    diagnostics["warnings"].append("No event found for slug.")
    return None, diagnostics


def resolve_polymarket_event_markets_from_url_or_slug(value: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    parsed = parse_polymarket_url_or_slug(value)
    event, fetch_diagnostics = fetch_polymarket_event_by_slug_with_diagnostics(parsed.get("slug", ""), parsed.get("url") or None)
    diagnostics = {
        "parsed": parsed,
        "fetch": fetch_diagnostics,
        "event_title": "",
        "markets_extracted": 0,
        "market_types_found": [],
        "warning": "",
    }
    if not event:
        diagnostics["warning"] = "; ".join(fetch_diagnostics.get("warnings", []))
        out = pd.DataFrame(columns=POLYMARKET_URL_MARKET_COLUMNS)
        out.attrs.update(diagnostics)
        return out, diagnostics

    flat = flatten_gamma_events_to_markets([event])
    rows = []
    source_url = parsed.get("url") or _event_url(parsed.get("slug", ""), parsed.get("competition_hint", "world-cup"))
    extraction_method = fetch_diagnostics.get("method", "")
    for _, row in flat.iterrows():
        market_type = classify_polymarket_market_type(row)
        rows.append(
            {
                "event_id": row.get("event_id", ""),
                "event_slug": row.get("event_slug", parsed.get("slug", "")),
                "event_title": row.get("event_title", ""),
                "event_category": row.get("event_category", ""),
                "event_start_date": row.get("event_start_date", pd.NA),
                "event_end_date": row.get("event_end_date", pd.NA),
                "market_id": row.get("market_id", ""),
                "market_slug": row.get("market_slug", ""),
                "question": row.get("question", ""),
                "market_title": row.get("market_title", ""),
                "market_type": market_type,
                "outcomes": row.get("outcomes", ""),
                "yes_price": row.get("yes_price", pd.NA),
                "no_price": row.get("no_price", pd.NA),
                "liquidity": row.get("liquidity", pd.NA),
                "volume": row.get("volume", pd.NA),
                "source_url": source_url,
                "source": row.get("source", SOURCE_API),
                "last_updated": row.get("last_updated", utc_now_iso()),
                "extraction_method": extraction_method,
            }
        )
    out = pd.DataFrame(rows, columns=POLYMARKET_URL_MARKET_COLUMNS)
    diagnostics["event_title"] = str(event.get("title") or event.get("name") or event.get("slug") or "")
    diagnostics["markets_extracted"] = int(len(out))
    diagnostics["market_types_found"] = sorted(out["market_type"].dropna().astype(str).unique().tolist()) if not out.empty else []
    if out.empty:
        diagnostics["warning"] = "Event resolved, but no nested markets were extracted."
    out.attrs.update(diagnostics)
    out.attrs["source_label"] = "Polymarket URL resolver"
    return out, diagnostics


def score_polymarket_event_slug_for_fixture(
    slug: str,
    home: str,
    away: str,
    fixture_date: str | None = None,
) -> dict[str, Any]:
    parsed = parse_polymarket_url_or_slug(slug)
    home_name = normalize_team_name(home)
    away_name = normalize_team_name(away)
    slug_team_1 = normalize_polymarket_team_code(parsed.get("team_code_1", ""))
    slug_team_2 = normalize_polymarket_team_code(parsed.get("team_code_2", ""))
    slug_teams = {slug_team_1, slug_team_2}
    matched_home = bool(home_name and home_name in slug_teams)
    matched_away = bool(away_name and away_name in slug_teams)
    matched_date = _dates_match(parsed.get("date_hint", ""), fixture_date)
    matched_competition = parsed.get("competition_hint") in {"world-cup", "fifa-world-cup"} or str(parsed.get("slug", "")).startswith("fifwc-")

    score = 0.0
    reasons = []
    if matched_home:
        score += 35.0
        reasons.append("home team code matched")
    if matched_away:
        score += 35.0
        reasons.append("away team code matched")
    if matched_date:
        score += 20.0
        reasons.append("fixture date matched")
    if matched_competition:
        score += 10.0
        reasons.append("World Cup slug/path matched")

    rejected = False
    warnings = []
    if not matched_home:
        warnings.append("home mismatch")
    if not matched_away:
        warnings.append("away mismatch")
    if fixture_date and not matched_date:
        warnings.append("date mismatch")
    if not (matched_home and matched_away):
        rejected = True
    confidence = "high" if not rejected and matched_date and matched_competition else "medium" if not rejected else "reject"

    return {
        "slug": parsed.get("slug", ""),
        "home": home_name,
        "away": away_name,
        "slug_team_1": slug_team_1,
        "slug_team_2": slug_team_2,
        "matched_home": matched_home,
        "matched_away": matched_away,
        "matched_date": matched_date,
        "matched_competition": bool(matched_competition),
        "score": round(score, 2),
        "confidence": confidence,
        "rejected": rejected,
        "warning": "; ".join(warnings),
        "score_breakdown": "; ".join(reasons),
    }


def resolved_url_markets_to_polymarket_rows(markets_df: pd.DataFrame) -> pd.DataFrame:
    from src.polymarket import POLYMARKET_COLUMNS

    markets = markets_df.copy() if markets_df is not None else pd.DataFrame(columns=POLYMARKET_URL_MARKET_COLUMNS)
    for col in POLYMARKET_URL_MARKET_COLUMNS:
        if col not in markets.columns:
            markets[col] = pd.NA
    rows = []
    for _, row in markets.iterrows():
        rows.append(
            {
                "market_id": row.get("market_id", ""),
                "question": row.get("question", row.get("market_title", "")),
                "slug": row.get("market_slug", ""),
                "event_title": row.get("event_title", ""),
                "category": row.get("event_category", ""),
                "start_date": row.get("event_start_date", pd.NA),
                "end_date": row.get("event_end_date", pd.NA),
                "active": 1,
                "closed": 0,
                "outcomes": row.get("outcomes", ""),
                "yes_price": row.get("yes_price", pd.NA),
                "no_price": row.get("no_price", pd.NA),
                "liquidity": row.get("liquidity", pd.NA),
                "volume": row.get("volume", pd.NA),
                "source": row.get("source", SOURCE_API),
                "last_updated": row.get("last_updated", utc_now_iso()),
            }
        )
    out = pd.DataFrame(rows, columns=POLYMARKET_COLUMNS)
    out.attrs["source_label"] = "Polymarket URL resolver"
    return out


def classify_polymarket_market_type(market: dict | pd.Series) -> str:
    row = market if isinstance(market, pd.Series) else pd.Series(market or {})
    text = _normalise(
        " ".join(
            str(row.get(col, "") or "")
            for col in ["question", "market_title", "market_slug", "slug", "groupItemTitle", "description"]
        )
    )
    if "both teams to score" in text or "btts" in text:
        return "btts"
    if "first team to score" in text or "score first" in text:
        return "first_team_to_score"
    if "exact score" in text or "correct score" in text or re.search(r"(?<![\d-])\d{1,2}\s*-\s*\d{1,2}(?![\d-])", text):
        return "exact_score"
    if "corner" in text:
        return "corners"
    if "assist" in text:
        return "assists"
    if "shot" in text:
        return "shots"
    if "spread" in text or "handicap" in text or "by more than" in text:
        return "spread"
    if "total" in text or "over " in text or "under " in text:
        return "total"
    if "half" in text:
        return "half_result"
    if "goal" in text and not ("win" in text or "draw" in text):
        return "goals"
    if "draw" in text or " win " in f" {text} " or text.startswith("will ") and " win" in text:
        return "moneyline"
    return "unknown"


def _fetch_event_by_slug_param(slug: str, diagnostics: dict[str, Any]) -> dict | None:
    diagnostics["methods_tried"].append("gamma_events_slug_filter")
    try:
        response = requests.get(_gamma_endpoint("events"), params={"slug": slug}, timeout=25)
        response.raise_for_status()
        for event in _extract_event_records(response.json()):
            if str(event.get("slug", "")).strip() == slug:
                return event
    except Exception as exc:
        diagnostics["warnings"].append(f"slug filter failed: {exc}")
    return None


def _fetch_event_by_slug_path(slug: str, diagnostics: dict[str, Any]) -> dict | None:
    diagnostics["methods_tried"].append("gamma_events_slug_path")
    try:
        response = requests.get(f"{_gamma_endpoint('events')}/slug/{slug}", timeout=25)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and str(payload.get("slug", "")).strip() == slug:
            return payload
    except Exception as exc:
        diagnostics["warnings"].append(f"slug path failed: {exc}")
    return None


def _fetch_event_by_exact_search(slug: str, diagnostics: dict[str, Any]) -> dict | None:
    diagnostics["methods_tried"].append("gamma_events_search")
    try:
        response = requests.get(_gamma_endpoint("events"), params={"search": slug, "limit": 25}, timeout=25)
        response.raise_for_status()
        for event in _extract_event_records(response.json()):
            if str(event.get("slug", "")).strip() == slug:
                return event
    except Exception as exc:
        diagnostics["warnings"].append(f"event search failed: {exc}")
    return None


def _fetch_event_from_cache(slug: str, diagnostics: dict[str, Any]) -> dict | None:
    diagnostics["methods_tried"].append("events_cache")
    cache = read_csv_with_columns(EVENTS_CACHE_PATH, GAMMA_EVENT_CACHE_COLUMNS)
    if cache.empty or "event_slug" not in cache.columns:
        return None
    matches = cache.loc[cache["event_slug"].astype(str) == slug]
    for _, row in matches.iterrows():
        raw = row.get("raw_event_json", "")
        try:
            event = json.loads(str(raw or ""))
        except json.JSONDecodeError:
            event = {}
        if event:
            return event
        return {
            "id": row.get("event_id", ""),
            "slug": row.get("event_slug", ""),
            "title": row.get("event_title", ""),
            "category": row.get("event_category", ""),
            "startDate": row.get("event_start_date", pd.NA),
            "endDate": row.get("event_end_date", pd.NA),
            "active": coerce_bool(row.get("event_active", True)),
            "closed": coerce_bool(row.get("event_closed", False)),
            "markets": [],
        }
    return None


def _fetch_event_from_html(slug: str, source_url: str | None, diagnostics: dict[str, Any]) -> dict | None:
    diagnostics["methods_tried"].append("polymarket_html")
    url = source_url or _event_url(slug, "world-cup")
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html = response.text
    except Exception as exc:
        diagnostics["warnings"].append(f"HTML fetch failed: {exc}")
        return None
    event = _extract_event_from_next_data(html, slug)
    if event:
        return event
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else slug
    return {"slug": slug, "title": title, "markets": []}


def _extract_event_from_next_data(html: str, slug: str) -> dict | None:
    match = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, flags=re.I | re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    for record in _walk_dicts(data):
        if str(record.get("slug", "")).strip() == slug and isinstance(record.get("markets"), list):
            return record
    return None


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _extract_event_records(payload: Any) -> list[dict]:
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


def _gamma_endpoint(kind: str) -> str:
    cfg = get_config()
    base = cfg.polymarket_gamma_api_url or cfg.polymarket_api_url or DEFAULT_POLYMARKET_GAMMA_API_URL
    endpoint = str(base).rstrip("/")
    for suffix in ("/markets", "/events", "/tags", "/series", "/sports"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    return f"{endpoint}/{kind}"


def _event_url(slug: str, competition_hint: str | None) -> str:
    competition = str(competition_hint or "world-cup").strip("/") or "world-cup"
    return f"{DEFAULT_POLYMARKET_WEB_URL}/sports/{competition}/{slug}"


def _dates_match(slug_date: str, fixture_date: str | None) -> bool:
    if not slug_date or not fixture_date:
        return False
    slug_ts = pd.to_datetime(slug_date, errors="coerce")
    fixture_ts = pd.to_datetime(fixture_date, errors="coerce")
    if pd.isna(slug_ts) or pd.isna(fixture_ts):
        return False
    return slug_ts.date() == fixture_ts.date()


def _normalise(value: Any) -> str:
    text = str(value or "").lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9.+ -]+", " ", text)
    return " ".join(text.split())
